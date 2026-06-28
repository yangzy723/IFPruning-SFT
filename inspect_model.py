import torch
from transformers import AutoModelForCausalLM, AutoConfig

# 本地模型路径
MODEL_PATH = "./gemma-4-12B"

def inspect_model_structure(model_path):
    try:
        config = AutoConfig.from_pretrained(model_path)
        print("\n========== 1. 核心 Config 配置 ==========")
        print(f"模型架构 (Architectures): {config.architectures}")
        print(f"模型类型 (Model Type): {config.model_type}")
        
        if hasattr(config, "text_config"):
            print("多模态嵌套！真正的文本配置在 'text_config' 中:")
            print(f"   - 隐藏层维度 (hidden_size): {config.text_config.hidden_size}")
            print(f"   - 网络层数 (num_hidden_layers): {config.text_config.num_hidden_layers}")
            print(f"   - FFN维度 (intermediate_size): {config.text_config.intermediate_size}")
        else:
            print(f"标准结构:")
            print(f"   - 隐藏层维度 (hidden_size): {getattr(config, 'hidden_size', '未知')}")
    except Exception as e:
        print(f"读取配置失败: {e}")

    print("\n正在以低内存模式加载模型物理结构...")
    # low_cpu_mem_usage=True 是关键，在 CPU 内存中快速建构模型骨架
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )

    print("\n========== 2. 宏观物理结构树 ==========")
    # 直接 print 模型，可看到类似系统文件夹的嵌套结构
    print(model)

    print("\n========== 3. 模块绝对路径 ==========")
    # 遍历并过滤出关心的模块（比如 layers 和 mlp）
    layer_count = 0
    mlp_paths = []
    
    for name, module in model.named_modules():
        # 寻找 Transformer 层堆叠的列表
        if isinstance(module, torch.nn.ModuleList):
            if len(module) > 0 and hasattr(module[0], "mlp"):
                print(f"找到 Transformer 堆叠层，绝对路径为: '{name}'")
                print(f"   - 该路径下共有 {len(module)} 层网络。")
                layer_count = len(module)
        
        # 收集几个 MLP 的路径作为例子
        if "mlp" in name.lower() and isinstance(module, torch.nn.Module) and not isinstance(module, torch.nn.ModuleList):
            mlp_paths.append((name, module.__class__.__name__))

    if mlp_paths:
        print(f"\nMLP (前馈神经网络) 模块示例:")
        # 只打印前 3 个和最后 1 个，避免刷屏
        for path, cls_name in mlp_paths[:3]:
            print(f"   - 路径: {path} ---> 类型: {cls_name}")
        print("   - ... (中间省略) ...")
        print(f"   - 路径: {mlp_paths[-1][0]} ---> 类型: {mlp_paths[-1][1]}")
    else:
        print("\nError! 未能按常规命名找到 mlp 模块，请检查上方宏观结构树。")

if __name__ == "__main__":
    inspect_model_structure(MODEL_PATH)