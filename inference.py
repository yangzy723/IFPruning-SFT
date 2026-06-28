#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IFPruning SFT Inference Pipeline
Supports interactive terminal chat and exports routing scores for visualization.
"""

import os
import sys
import logging
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from transformers.models.gemma.modeling_gemma import GemmaMLP
from safetensors.torch import load_file

# ==============================================================================
# Global Configuration
# ==============================================================================
BASE_MODEL_PATH = "./gemma-4-12B"
PREDICTOR_MODEL_PATH = "./Qwen3.5-0.8B"
CHECKPOINT_DIR = "./gemma-12B-ifpruning" 
TARGET_SPARSE_DIM = 4096

# 用于保存 Score
SCORE_DUMP_DIR = Path("./routing_scores")
SCORE_DUMP_DIR.mkdir(exist_ok=True, parents=True)

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] [%(process)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S", 
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("ifpruning_inference")

# ==============================================================================
# Module 1 & 2: Architecture & Wrapper
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
            _, topk_idx = torch.topk(scores, self.target_ffn_dim, dim=-1)
            indicator = torch.zeros_like(scores).scatter_(-1, topk_idx, 1.0)
            mask = indicator.unsqueeze(1) 
            activated_hidden = (gate_out * up_out) * mask
        else:
            activated_hidden = gate_out * up_out
        return self.down_proj(activated_hidden)

class GemmaIFPruningWrapper(nn.Module):
    def __init__(self, base_model, target_ffn_dim: int, extractor_path: str):
        super().__init__()
        self.base_model = base_model
        cfg = getattr(base_model.config, "text_config", base_model.config)
        self.predictor = SparsityPredictor(cfg.num_hidden_layers, cfg.intermediate_size, extractor_path)
        
        target_device = next(base_model.parameters()).device
        self.predictor.to(device=target_device, dtype=base_model.dtype)
        
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
            target_device = next(self.predictor.parameters()).device
            p_ids = predictor_input_ids.to(target_device)
            p_mask = predictor_attention_mask.to(target_device)
            
            all_layer_scores = self.predictor(p_ids, p_mask)
            for i, layer in enumerate(self.llm_layers):
                layer.mlp.layer_scores = all_layer_scores[:, i, :]
            
            return all_layer_scores.cpu()

# ==============================================================================
# Execution Entry Point
# ==============================================================================
def main():
    checkpoint_dir = Path(CHECKPOINT_DIR)
    predictor_model_path = Path(PREDICTOR_MODEL_PATH)
    base_model_path = Path(BASE_MODEL_PATH)
    predictor_weights_path = checkpoint_dir / "predictor_mlp.safetensors"
    
    if not checkpoint_dir.exists() or not predictor_weights_path.exists():
        raise FileNotFoundError(f"Missing checkpoint or predictor weights in {checkpoint_dir}")

    logger.info("Initializing tokenizers...")
    base_tokenizer = AutoTokenizer.from_pretrained(str(base_model_path), local_files_only=True)
    predictor_tokenizer = AutoTokenizer.from_pretrained(str(predictor_model_path), local_files_only=True)

    logger.info("Loading custom base model from checkpoint...")
    base_model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint_dir), 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        local_files_only=True
    )

    try:
        embed_weight = base_model.model.language_model.embed_tokens.weight
        lm_head_weight = base_model.lm_head.weight
        if embed_weight.data_ptr() != lm_head_weight.data_ptr():
            base_model.lm_head.weight = embed_weight
    except Exception:
        pass

    logger.info("Injecting dynamic activation sparsity architecture...")
    model = GemmaIFPruningWrapper(base_model, TARGET_SPARSE_DIM, str(predictor_model_path))

    logger.info("Restoring decoupled predictor parameters from safetensors...")
    pred_state_dict = load_file(str(predictor_weights_path))
    model.predictor.mlp.load_state_dict(pred_state_dict, strict=True)
    del pred_state_dict
    
    model.eval()
    print("\n" + "=" * 60)
    print("--- IFPruning Interactive Shell ---")
    print("Type 'quit' or 'exit' to stop.")
    print("=" * 60 + "\n")
    
    chat_tpl = (
        "{% for m in messages %}"
        "{{'<|turn>' + m['role'] + '\\n' + m['content'] + '<turn|>\\n'}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{'<|turn>model\\n'}}{% endif %}"
    )
    
    input_device = next(base_model.parameters()).device
    prompt_idx = 1

    while True:
        try:
            instruction = input(f"\n[User (Prompt #{prompt_idx})] ❯ ")
            if not instruction.strip():
                continue
            if instruction.strip().lower() in ['quit', 'exit']:
                print("Exiting...")
                break
                
            messages = [{"role": "user", "content": instruction}]
            base_prompt = base_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, chat_template=chat_tpl)
            base_inputs = base_tokenizer(base_prompt, return_tensors="pt", add_special_tokens=False)
            base_inputs = {k: v.to(input_device) for k, v in base_inputs.items()}
            
            pred_inputs = predictor_tokenizer(instruction, return_tensors="pt", add_special_tokens=True)
            
            # 计算掩码，并提取 Score 矩阵
            scores_tensor = model.compute_and_lock_mask(
                predictor_input_ids=pred_inputs["input_ids"],
                predictor_attention_mask=pred_inputs["attention_mask"]
            )

            safe_name = "".join(x for x in instruction[:15] if x.isalnum() or x.isspace()).replace(" ", "_")
            file_name = f"score_{prompt_idx:02d}_{safe_name}.pt"
            save_path = SCORE_DUMP_DIR / file_name
            
            torch.save({
                "prompt": instruction,
                "scores": scores_tensor.squeeze(0).float() # 保存为 [L, D] 的 float32 矩阵
            }, save_path)
            
            print(f"[*] Routing scores saved to: ./routing_scores/{file_name}")

            print("[IFP Model] ❯ ", end="", flush=True)
            with torch.no_grad():
                outputs = model.base_model.generate(
                    **base_inputs, 
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.05,
                    do_sample=True,
                    pad_token_id=base_tokenizer.eos_token_id
                )
            
            input_length = base_inputs["input_ids"].shape[1]
            response = base_tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
            print(response)
            
            prompt_idx += 1

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            logger.error(f"Generation error: {e}")

if __name__ == "__main__":
    main()