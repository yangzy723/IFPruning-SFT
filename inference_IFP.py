import os
import sys
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from transformers.models.gemma.modeling_gemma import GemmaMLP
from safetensors.torch import load_file

# ==============================================================================
# 全局日志配置
# ==============================================================================
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] -> %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S", 
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ==============================================================================
# 模块 1: 核心组件 (绝对数值稳定版)
# ==============================================================================
class SparsityPredictor(nn.Module):
    def __init__(self, target_num_layers: int, target_ffn_dim: int, extractor_path: str):
        super().__init__()
        self.num_layers = target_num_layers
        self.ffn_dim = target_ffn_dim
        
        self.feature_extractor = AutoModel.from_pretrained(
            extractor_path, torch_dtype=torch.bfloat16, local_files_only=True
        )
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        config = self.feature_extractor.config
        extractor_hidden_dim = getattr(config, "hidden_size", None) or \
                               getattr(config, "d_model", None) or \
                               getattr(config, "n_embd", None)    
        if extractor_hidden_dim is None:
            extractor_hidden_dim = self.feature_extractor.get_input_embeddings().weight.shape[1]

        self.mlp = nn.Sequential(
            nn.Linear(extractor_hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, target_num_layers * target_ffn_dim)
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.feature_extractor(input_ids=input_ids, attention_mask=attention_mask)
        seq_lengths = attention_mask.sum(dim=1) - 1
        last_token_states = outputs.last_hidden_state[torch.arange(input_ids.shape[0]), seq_lengths]
        return self.mlp(last_token_states).view(-1, self.num_layers, self.ffn_dim)


class GemmaDynamicMaskedFFN_Inference(nn.Module):
    def __init__(self, original_mlp: GemmaMLP, target_ffn_dim: int):
        super().__init__()
        self.gate_proj = original_mlp.gate_proj
        self.up_proj = original_mlp.up_proj
        self.down_proj = original_mlp.down_proj
        self.act_fn = original_mlp.act_fn 
        self.target_ffn_dim = target_ffn_dim
        self.layer_scores = None 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_out = self.act_fn(self.gate_proj(x))
        up_out = self.up_proj(x)
        
        if self.layer_scores is not None:
            scores = self.layer_scores.to(device=x.device, dtype=x.dtype)
            
            # 🛡️ 核心修复：抛弃带有核爆风险的 Softmax 放大系数，直接提取 Hard Top-K
            # 这保证了被选中的神经元权重为绝对的 1.0，屏蔽的为 0.0，绝不改变原模型的激活值规模！
            _, topk_idx = torch.topk(scores, self.target_ffn_dim, dim=-1)
            indicator = torch.zeros_like(scores).scatter_(-1, topk_idx, 1.0)
            
            mask = indicator.unsqueeze(1) 
            activated_hidden = (gate_out * up_out) * mask
        else:
            logger.error("推理期间未检测到锁定的层掩码！模型退化为 Dense 模式。")
            activated_hidden = gate_out * up_out
            
        return self.down_proj(activated_hidden)

# ==============================================================================
# 模块 2: 推理架构代理与掩码锁定机制
# ==============================================================================
class GemmaIFPruningWrapper(nn.Module):
    def __init__(self, base_model, target_ffn_dim: int, extractor_path: str):
        super().__init__()
        self.base_model = base_model
        cfg = getattr(base_model.config, "text_config", base_model.config)
        
        self.predictor = SparsityPredictor(cfg.num_hidden_layers, cfg.intermediate_size, extractor_path)
        self.predictor.to(device=base_model.device, dtype=base_model.dtype)
        
        self.llm_layers = [m for n, m in self.base_model.named_modules() if isinstance(m, nn.ModuleList) and hasattr(m[0], 'mlp')][0]
        for layer in self.llm_layers:
            layer.mlp = GemmaDynamicMaskedFFN_Inference(layer.mlp, target_ffn_dim)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)

    def compute_and_lock_mask(self, predictor_input_ids: torch.Tensor, predictor_attention_mask: torch.Tensor):
        with torch.no_grad():
            all_layer_scores = self.predictor(predictor_input_ids, predictor_attention_mask)
            for i, layer in enumerate(self.llm_layers):
                layer.mlp.layer_scores = all_layer_scores[:, i, :]
        logger.info(f"✅ 掩码已成功计算并锁定！(Target Dim: 4096，模式: Hard Mask 防爆)")

# ==============================================================================
# 主推理管线
# ==============================================================================
def main():
    base_model_path = os.path.abspath("./gemma-4-12b")
    predictor_model_path = os.path.abspath("./Qwen3.5-0.8b")
    checkpoint_path = os.path.abspath("./gemma-12b-ifpruning-output/checkpoint-7000/model.safetensors") 
    target_dim = 4096

    logger.info("1. 加载 Tokenizer...")
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_path, local_files_only=True)
    predictor_tokenizer = AutoTokenizer.from_pretrained(predictor_model_path, local_files_only=True)

    logger.info("2. 加载 Gemma-12B 底座并启动多卡切分 (device_map)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )

    logger.info("3. 包装 IFPruning 动态推理架构...")
    model = GemmaIFPruningWrapper(base_model, target_dim, predictor_model_path)

    # ==========================================================================
    # 🌟 核心修复：既然内存够大，直接使用原生 API，确保所有 Tied Weights 完美加载
    # ==========================================================================
    logger.info(f"4. 正在从硬盘加载 46GB 权重进内存，请稍候...")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"找不到权重文件: {checkpoint_path}")
        
    state_dict = load_file(checkpoint_path)
    logger.info(f"5. 正在将权重分发至 GPU (确保指针无损)...")
    model.load_state_dict(state_dict, strict=False)
    
    # 释放内存中的庞大字典
    del state_dict 
    torch.cuda.empty_cache()
    # ==========================================================================

    model.eval()
    logger.info("🚀 模型环境就绪，启动推理！\n")
    
    instruction = "用 Python 写一个快速排序算法，并加好注释。"
    prompt = f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n"
    
    base_inputs = base_tokenizer(prompt, return_tensors="pt").to(base_model.device)
    pred_inputs = predictor_tokenizer(prompt, return_tensors="pt").to(base_model.device)
    
    model.compute_and_lock_mask(
        predictor_input_ids=pred_inputs["input_ids"],
        predictor_attention_mask=pred_inputs["attention_mask"]
    )

    logger.info("开始流式生成...")
    with torch.no_grad():
        outputs = model.base_model.generate(
            **base_inputs, 
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.05,
            do_sample=True
        )
    
    input_length = base_inputs["input_ids"].shape[1]
    response = base_tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    
    print("\n" + "=" * 60)
    print(f"😎 用户输入: {instruction}")
    print("=" * 60)
    print(f"🤖 稀疏模型响应:\n{response}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()