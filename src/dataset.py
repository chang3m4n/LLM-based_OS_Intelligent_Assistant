import json

# 读取原始数据
with open("data/train4.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 清洗并统一格式
cleaned_data = []
for item in data:
    # 确保问题和答案存在
    if "question" not in item or "answer" not in item:
        continue
    # 统一前缀格式（去除多余空格、替换中文冒号）
    question = item["question"].strip()
    answer = item["answer"].strip().replace("Output：", "Output: ")  # 替换中文冒号
    # 重新拼接
    cleaned_item = {
        "question": question,
        "answer": answer
    }
    cleaned_data.append(cleaned_item)

# 保存清洗后的数据
with open("data/train4.json", "w", encoding="utf-8") as f:
    json.dump(cleaned_data, f, ensure_ascii=False, indent=2)