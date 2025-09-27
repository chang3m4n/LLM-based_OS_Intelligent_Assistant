from openai import OpenAI

# API配置
API_KEY = "sk-tlzpmectzllvacxvbukrbjseqeixpgappvragvigouhniqqg"
BASE_URL = "https://api.siliconflow.cn/v1"
API_MODEL_NAME = "ft:LoRA/Qwen/Qwen2.5-72B-Instruct:d26sf01719ns73a81sog:os:kuyejtcllpdczgbmqror"

# 初始化客户端
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

def get_api_answer(messages, max_tokens=2048, temperature=0.7):
    """
    调用API模型并返回生成的回答。

    :param messages: 发送给API的消息列表
    :param max_tokens: 最大生成令牌数
    :param temperature: 温度参数
    :return: API返回的回答字符串
    """
    try:
        response = client.chat.completions.create(
            model=API_MODEL_NAME,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False  # 在app.py中我们处理非流式响应
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"调用API失败: {e}")
        raise  # 重新引发异常，让调用者处理

if __name__ == '__main__':
    # 测试代码
    test_messages = [
        {"role": "system", "content": "你是操作系统课程智能问答助手；你的名字是小维；你只能回答和操作系统知识有关的问题；用户如果问和操作系统无关的问题，你要回答暂不支持此功能；你的回答语气要活泼"},
        {"role": "user", "content": "什么是进程？"}
    ]
    try:
        answer = get_api_answer(test_messages)
        print("API测试回答:")
        print(answer)
    except Exception as e:
        print(f"测试失败: {e}")
