import json
import argparse


def convert_to_jsonl(input_file, output_file):
    """
    将输入的JSON文件转换为符合要求的JSONL文件

    参数:
    input_file: 输入JSON文件路径
    output_file: 输出JSONL文件路径
    """
    try:
        # 读取输入JSON文件
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 确保输入数据是列表
        if not isinstance(data, list):
            data = [data]

        # 处理每条数据并写入JSONL文件
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in data:
                # 验证必要字段是否存在
                if 'question' not in item or 'answer' not in item:
                    print(f"跳过无效条目: {item} - 缺少question或answer字段")
                    continue

                # 构建符合要求的messages结构
                messages = [
                    {"role": "user", "content": item['question']},
                    {"role": "assistant", "content": item['answer']}
                ]

                # 创建符合要求的JSON对象
                jsonl_item = {"messages": messages}

                # 写入JSONL文件，每行一个JSON对象
                f.write(json.dumps(jsonl_item, ensure_ascii=False) + '\n')

        print(f"转换完成，已保存到 {output_file}")

    except Exception as e:
        print(f"转换过程中出错: {str(e)}")


if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='将包含question和answer的JSON文件转换为符合要求的JSONL文件')
    parser.add_argument('input', help='../data/train.json')
    parser.add_argument('output', help='../data/output.jsonl')

    args = parser.parse_args()

    # 执行转换
    convert_to_jsonl(args.input, args.output)
