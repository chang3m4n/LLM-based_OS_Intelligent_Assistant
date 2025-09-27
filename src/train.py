# 环境安装：pip install transformers peft datasets accelerate bitsandbytes
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import torch
import os
from datetime import datetime

# 1. 加载模型和分词器
model_name = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
cache_dir = "../models"
output_dir = "./finetune_results"

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 初始化分词器
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    padding_side="right",
    trust_remote_code=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 量化配置（24G显卡适用）
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 4bit量化释放显存，保障训练稳定性
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",  # 高精度量化格式，效果损失极小
    bnb_4bit_compute_dtype=torch.float16
)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    cache_dir=cache_dir
)

# 准备模型用于量化训练
model = prepare_model_for_kbit_training(model)

# 2. 配置LoRA（单卡优化参数）
peft_config = LoraConfig(
    r=16,  # 适中的秩，平衡效果和计算量
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
    bias="none"
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()  # 打印可训练参数占比

# 3. 加载数据集
dataset = load_dataset("json", data_files="../data/output.jsonl")  # 替换为你的数据路径
dataset = dataset["train"].train_test_split(test_size=0.1)  # 划分训练/验证集


# 4. 数据预处理
def tokenize_function(examples):
    # 从messages列表中提取user的问题和assistant的回答
    questions = []
    answers = []
    for msg_list in examples["messages"]:
        # 遍历单条数据的messages，分离user和assistant内容
        user_content = ""
        assistant_content = ""
        for msg in msg_list:
            if msg["role"] == "user":
                user_content = msg["content"].strip()  # 提取用户问题
            elif msg["role"] == "assistant":
                assistant_content = msg["content"].strip()  # 提取助手回答

        # 过滤无效样本（缺少问题或回答）
        if user_content and assistant_content:
            questions.append(user_content)
            answers.append(assistant_content)
        else:
            # 打印无效样本提示（可选，用于数据校验）
            print(f"跳过无效样本：缺少user/assistant内容 | messages: {msg_list[:50]}...")

    # 沿用原有的"Instruction-Output"格式
    texts = [f"Instruction: {q}\nOutput: {a}" for q, a in zip(questions, answers)]

    # 分词处理（与原逻辑一致）
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=1024,  # 24G显卡可支持的长度
        padding="max_length",
        add_special_tokens=True
    )

    # 构建标签（忽略Instruction部分，仅对Output计算损失）
    labels = []
    for input_ids in tokenized["input_ids"]:
        text = tokenizer.decode(input_ids, skip_special_tokens=True)
        output_pos = text.find("Output:")

        if output_pos >= 0:
            # 计算"Output:"前的前缀长度，用-100标记（不参与损失计算）
            prefix = text[:output_pos + len("Output:")]
            prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
            label = [-100] * len(prefix_ids) + input_ids[len(prefix_ids):]
        else:
            label = input_ids.copy()  # 回退方案（极少触发）

        labels.append(label)

    tokenized["labels"] = labels
    return tokenized


# --------------------------------------------------------------------------------


# 应用预处理
tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
    num_proc=4  # 适配服务器CPU核心数
)

# 数据整理器
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # 因果语言模型不需要掩码语言建模
)

# 5. 训练配置
training_args = TrainingArguments(
    output_dir=output_dir,
    # 批次设置
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,  # 累计梯度，模拟更大批次
    # 学习率与轮次
    learning_rate=2e-5,
    num_train_epochs=10,
    # 评估策略
    eval_strategy="epoch",
    # 混合精度
    fp16=True,  # 启用FP16加速
    # 日志与保存
    logging_dir=f"{output_dir}/logs",
    logging_steps=20,
    save_strategy="epoch",
    save_total_limit=2,  # 只保留2个最佳模型
    # 优化器
    optim="paged_adamw_8bit",  # 节省内存的优化器
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.01,
    # 加载最佳模型
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)

# 6. 启动训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator
)

print("开始训练...")
trainer.train()

# 7. 保存模型
model.save_pretrained(f"{output_dir}/lora_adapter")
tokenizer.save_pretrained(f"{output_dir}/lora_adapter")
print(f"模型已保存至 {output_dir}/lora_adapter")