# model.py
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch
import os

def load_model_and_tokenizer(base_model_name,lora_dir,cache_dir="./models"):
    """加载基础模型、LoRA适配器和分词器，返回 (model, tokenizer)"""
    # 1. 配置路径（与训练代码一致）

   # offload_folder = "./model_offload"
   # os.makedirs(offload_folder, exist_ok=True)

    # 2. 验证LoRA目录完整性
    if not os.path.exists(lora_dir):
        raise FileNotFoundError(f"LoRA目录不存在：{lora_dir}，请先训练")
    required_files = ["adapter_config.json", "adapter_model.bin"]
    for file in required_files:
        if not os.path.exists(os.path.join(lora_dir, file)):
            raise FileNotFoundError(f"LoRA缺少文件：{file}，请重新训练")

    # 3. 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        cache_dir=cache_dir
    )
    tokenizer.pad_token = tokenizer.eos_token  # 设置pad_token

    # 4. 加载量化配置（与训练一致）
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    # 5. 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        cache_dir=cache_dir,
        #offload_folder=offload_folder,
        trust_remote_code=True
    )

    # 6. 加载LoRA适配器
    model = PeftModel.from_pretrained(
        base_model,
        lora_dir,
        device_map="auto"
    )
    model.eval()  # 推理模式

    print("模型和分词器加载完成！")
    return model, tokenizer


def generate_answer(model, tokenizer, question, max_new_tokens=1024, temperature=0.7):
    """使用加载好的模型和分词器生成回答"""
    # 输入格式必须与训练一致
    input_text = f"{question}"
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 简单的答案提取逻辑
    if "Output:" in input_text:
        answer = generated_text[len(input_text):].strip()
    else:
        answer = generated_text.strip()
        
    return answer


# 主程序：加载模型并测试推理
if __name__ == "__main__":

    lora_dir = "./deepseek_lora_adapter"
    base_model_name = "deepseek-ai/deepseek-llm-7b-base"
   # base_model_name = "deepseek-ai/Qwen2.5-7B-Instruct"
   # base_model_name = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
    try:
        model, tokenizer = load_model_and_tokenizer(base_model_name,lora_dir)
    except Exception:
        exit(1)

    while True:
        question = input("请输入问题（输入'退出'结束）：")
        if question.lower() == "退出":
            break
        try:
            answer = generate_answer(model, tokenizer, question)
            print(f"\n回答：{answer}\n")
        except Exception as e:
            print(f"推理失败：{e}")