from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch
import os


def load_model_and_tokenizer(base_model_name, lora_dir, cache_dir="./models"):
    """加载基础模型、LoRA适配器和分词器，返回 (model, tokenizer)"""
    # 1. 验证LoRA目录完整性
    if not os.path.exists(lora_dir):
        raise FileNotFoundError(f"LoRA目录不存在：{lora_dir}，请先训练")
    required_files = ["adapter_config.json", "adapter_model.bin"]
    for file in required_files:
        if not os.path.exists(os.path.join(lora_dir, file)):
            raise FileNotFoundError(f"LoRA缺少文件：{file}，请重新训练")

    # 2. 加载分词器（与训练保持一致，补充padding_side和trust_remote_code）
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        cache_dir=cache_dir,
        padding_side="right",  # 必须与训练一致，避免有效文本被截断
        trust_remote_code=True  # DeepSeek模型需要此参数
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # 确保pad_token存在

    # 3. 加载量化配置（与训练完全一致）
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    # 4. 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        cache_dir=cache_dir,
        trust_remote_code=True,  # DeepSeek模型需要此参数
        torch_dtype=torch.float16
    )

    # 5. 加载LoRA适配器
    model = PeftModel.from_pretrained(
        base_model,
        lora_dir,
        device_map="auto"
    )
    model.eval()  # 推理模式（禁用dropout等训练特有层）

    print("模型和分词器加载完成！")
    return model, tokenizer


def generate_answer(model, tokenizer, question, max_new_tokens=1024, temperature=0.7):

    input_text = f"Instruction: {question}\nOutput:"

    # 分词（与训练保持一致的参数）
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=1024,  # 与训练的max_length一致
        padding="max_length" if len(input_text) < 1024 else False  # 保持格式统一
    ).to(model.device)

    # 推理生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_return_sequences=1  # 只生成1个回答
        )

    # 解码生成结果
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 只取"Output:"后面的内容
    if "Output:" in generated_text:
        # 截取"Output:"之后的文本作为最终回答
        answer = generated_text.split("Output:")[-1].strip()
        # 处理极端情况：如果模型没生成有效内容，返回提示
        if not answer:
            answer = "未生成有效回答，请尝试调整问题或参数。"
    else:
        # 兜底：如果格式异常，返回完整生成结果（便于调试）
        answer = generated_text.strip()

    return answer


# 主程序：加载模型并测试推理
if __name__ == "__main__":
    lora_dir = "./finetune_results/lora_adapter"

    base_model_name = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"

    try:
        model, tokenizer = load_model_and_tokenizer(base_model_name, lora_dir)
    except Exception as e:
        print(f"模型加载失败：{e}")
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