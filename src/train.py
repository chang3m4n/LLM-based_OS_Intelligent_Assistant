from transformers import (
    AutoModelForCausalLM, # 因果模型
    AutoTokenizer, # 分词
    Trainer, # 封装训练逻辑，只用写参数
    TrainingArguments,
    BitsAndBytesConfig, # 量化
    DataCollatorForSeq2Seq # padding+mask -> tensor
)
# 微调
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import torch
import os
import re


# -------------------------- 配置参数 --------------------------
class TrainingConfig:
    MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
    CACHE_DIR = "../models"
    OUTPUT_DIR = "./finetune_results"
    DATA_FILE = "../data/output.jsonl"

    # 训练参数
    BATCH_SIZE = 4
    # 梯度累加，读两份数据才更新
    GRADIENT_ACCUMULATION_STEPS = 2
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 15
    MAX_LENGTH = 1024

'''
要微调大模型，首先需要一个标准格式的数据集，
因此第一步就是对原始数据集转换
第二步，把数据tokenize，padding，mask
第三步，准备好模型框架，比如lora，写好配置，
其实也就是准备好数据通路，告诉模型沿着什么路径传播
梯度沿着什么方向传播，如何传播。
第四步，设置训练参数，知道训练什么时候结束
第五步，推理验证
'''
# -------------------------- 数据处理 --------------------------
def tokenize_function(batch):
    SYSTEM_PROMPT = "你是操作系统课程智能问答助手，名字是比艾特。专注回答和操作系统知识相关的问题；回答简明扼要，专业准确。"

    traindata = []
    trainlabel = []

    '''
    不能对整条数据使用template的原因是找不到label从哪里开始
    label找不到从哪里开始的原因是粘连，所以我们才不希望因为粘连的原因导致训练出错
    比如[I am a student.<|Im_end|><|Im_start|>assistant\n我是一个学生...]
    假如贪恋编码的策略将assistant后面的"\n我" 编码成一个token了
    你如果用\n的id或者我的id去定位，可能会找不到这个id
    看了huggingface的源码，新版本为了解决这个问题已经在原有chat_template模板中
    加入了{%generation%}作为label的区间
    解决方法是：
    assistant的回答”我是一个学生“是放在generation里的，并且会记住他的字符串起始和
    结束位置，然后在string -> id的过程中，将token和string的映射关系存在offset mapping
    里，检查，如果某个token在generation的string范围内，就设置label -100
    哦，不同模型的模板和字典不一样，他不是硬编码，那他妈也没解决粘连啊
    不行了我先不管了
    反正qwen的模型没有这个功能
    我也只能硬编码

    造成掩码位置错误，学习到无效信息（system和user之间也可能发生粘连，但是不用于训练，影响不大）
    方案1：（麻烦，不通用，但是绝对正确）
    1. system部分+user开始标签应用模板,构成system_id
    2. 用户输入user字符串
    3. user字符串转user_id，根据maxlength截断处理
    4. system_id 拼接 user_id
    5. 拼接结束符号和answer开始符号
    6. 拼接answer_id + 结束符号
    
    方案2：如果不需要处理截断问题（训练的时候如果数据都是干净的，感觉可以使用这个方案）
    1. 对system+user的部分直接应用模板+特殊符号
    2. 拼接answer_id + 结束符号
    
    方案3：
    1.先提取用户输入的部分内容截断，然后转回字符串（转回可能导致语义偏差）
    2.使用方案2
    
    对于推理的时候，不需要label，system部分的长度是已知的，因此可以直接计算出user部分的预算
    对两个部分应用模板，用户的输入如果超出预算，直接截断，最后再补结束标识符和assistant的开头
    '''

    '''
    以下采用方案二
    messages 格式输入：
    [
        {"role": "system", "content": ""},
        {"role": "user", "content": ""},
        {"role": "assistant", "content": ""}
    ]
    '''
    for messages in batch["messages"]:
        question = messages[:-1]
        answer = messages[-1]['content']
        question_id = tokenizer.apply_chat_template(
            question,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors=None
        )
        answer_id = tokenizer.encode(
            answer,
            add_special_tokens=False,
            return_tensors=None
        )
        end_id = tokenizer.encode(
            '<|im_end|>',
            add_special_tokens=False,
            return_tensors=None
        )
        full_id = question_id + answer_id +end_id
        labels = [-100] * len(question_id) + answer_id +end_id

        traindata.append(full_id)
        trainlabel.append(labels)

    return {
        "input_ids": traindata,
        "labels": trainlabel,
        "attention_mask": [[1] * len(ids) for ids in traindata]
    }
    # 先是要对齐样本，填充padding   --> 应该是使用dataco。。啥的
    # 接下来还需要attention mask   --> 这个是告诉模型哪些是padding，没用的
    # 这个玩意应该是在哪里作用来着
    # 然后还需要因果掩码，这个应该是model内部的机制，我直接设置参数应该是
    # 有了这些直接丢进trainer，应该就结束了

# -------------------------- 模型设置 --------------------------
def load_model_and_tokenizer():
    """设置模型和分词器"""
    print("初始化分词器...")
    tokenizer = AutoTokenizer.from_pretrained(
        TrainingConfig.MODEL_NAME,
        cache_dir=TrainingConfig.CACHE_DIR,
        padding_side="right",
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("配置量化...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    print("加载基础模型...")
    model = AutoModelForCausalLM.from_pretrained(
        TrainingConfig.MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=TrainingConfig.CACHE_DIR,
        torch_dtype=torch.float16
    )

    model = prepare_model_for_kbit_training(model)
    return model, tokenizer


# -------------------------- 训练执行 --------------------------
def main():
    global tokenizer

    try:
        # 创建目录
        os.makedirs(TrainingConfig.OUTPUT_DIR, exist_ok=True)
        os.makedirs(TrainingConfig.CACHE_DIR, exist_ok=True)

        # 1. 加载数据
        print("加载数据集...")
        dataset = load_dataset("json", data_files=TrainingConfig.DATA_FILE)
        dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
        print(f"训练集: {len(dataset['train'])}, 验证集: {len(dataset['test'])}")

        # 2. 设置模型和分词器
        model, tokenizer = load_model_and_tokenizer()

        # 3. 配置LoRA
        print("配置LoRA...")
        peft_config = LoraConfig(
            r=32,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
            bias="none"
        )

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        # 4. 处理全部数据
        print("处理训练数据...")
        tokenized_datasets = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names,
            num_proc=1
        )

        # 5. 设置训练参数
        training_args = TrainingArguments(
            output_dir=TrainingConfig.OUTPUT_DIR,
            per_device_train_batch_size=TrainingConfig.BATCH_SIZE,
            per_device_eval_batch_size=TrainingConfig.BATCH_SIZE,
            gradient_accumulation_steps=TrainingConfig.GRADIENT_ACCUMULATION_STEPS,
            learning_rate=TrainingConfig.LEARNING_RATE,
            num_train_epochs=TrainingConfig.NUM_EPOCHS,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            fp16=True,
            logging_dir=f"{TrainingConfig.OUTPUT_DIR}/logs",
            logging_steps=20,
            save_total_limit=2,
            optim="paged_adamw_8bit",
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            weight_decay=0.01,
            report_to=None,
        )

        # 6. 数据整理器
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            label_pad_token_id=-100,
            return_tensors="pt"
        )

        # 7. 创建Trainer并训练
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
            data_collator=data_collator,
        )

        print("开始训练...")
        trainer.train()

        # 8. 保存模型
        final_output_dir = f"{TrainingConfig.OUTPUT_DIR}/final_model"
        model.save_pretrained(final_output_dir)
        tokenizer.save_pretrained(final_output_dir)

        print(f"训练完成！模型保存在: {final_output_dir}")

    except Exception as e:
        print(f"训练失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
