import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    # 使用原厂基座模型路径
    base_model_path = "./gemma-4-12b"

    print("1. 加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    print("2. 加载基座模型 (BF16, 自动分配多卡)...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto" 
    )
    model.eval()

    print("\n基座模型加载完毕！开始测试...")
    
    instruction = "用 Python 写一个快速排序算法，并加好注释。"
    prompt = f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=0.95
        )
    
    # 截取新生成的部分并解码
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    
    print(f"\n用户提问: {instruction}")
    print(f"模型回答:\n{response}")

if __name__ == "__main__":
    main()