import os
import random
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import requests
import urllib.parse
from bs4 import BeautifulSoup
# 确保 model.py 中的函数能正确导入（若不在同一目录，需调整路径）
from inference import generate_answer, load_model_and_tokenizer
from api import get_api_answer

app = Flask(__name__, template_folder='templates')
CORS(app)

# 本地模型缓存
local_model = None
local_tokenizer = None

# 系统提示词
SYSTEM_PROMPT = """你是操作系统课程智能问答助手，名字是比艾特。
- 仅回答和操作系统知识相关的问题；
- 若问题无关，直接回答"暂不支持此功能"，不额外解释；
- 回答语气友好"""

# 判断搜索结果相关性的提示词
RELEVANCE_PROMPT = """请判断以下问题是否与操作系统知识相关，只能回答"相关"或"不相关"，不允许其他任何回答：
{question}"""

# 不支持回答的标识文本
UNSUPPORTED_RESPONSE = "暂不支持此功能"

# 默认模型模式
DEFAULT_MODEL_MODE = os.environ.get('MODEL_MODE', 'api')
BASE_MODEL_NAME = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
LORA_DIR = "./finetune_results/lora_adapter"  # 训练生成的LoRA路径
CACHE_DIR = '../models'  # 模型缓存目录

# 搜索用户代理池
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0'
]


def get_random_headers():
    """获取随机请求头，避免被反爬"""
    return {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept': 'application/json, text/html, */*',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        'Connection': 'keep-alive'
    }

def preload_local_model():
    """预加载本地模型（与 model.py 参数对齐）"""
    base_model_name = BASE_MODEL_NAME
    lora_dir = LORA_DIR
    cache_dir = CACHE_DIR
    print("开始预加载本地模型...")
    try:
        # 调用 model.py 的加载函数，确保参数一致
        local_model, local_tokenizer = load_model_and_tokenizer(
            base_model_name=base_model_name,
            lora_dir=lora_dir,
            cache_dir=cache_dir
        )
        print("本地模型预加载完成。")
        return local_model, local_tokenizer
    except Exception as e:
        print(f"预加载本地模型失败: {e}")
        return None, None


def search_baidu_with_site(query, site, num_results=2):
    """在百度中使用site指令搜索指定单个网站的内容"""
    results = []
    try:
        full_query = f"{query} site:{site}"
        url = f"https://www.baidu.com/s?wd={urllib.parse.quote(full_query)}"

        response = requests.get(url, headers=get_random_headers(), timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')

        for result in soup.select('.result.c-container')[:num_results]:
            title_tag = result.select_one('h3.t a')
            if not title_tag:
                continue

            title = title_tag.get_text(strip=True)
            url = title_tag['href'] if 'href' in title_tag.attrs else ''
            source = site.split('.')[0].capitalize()

            results.append({
                'title': title,
                'url': url,
                'source': source
            })

        print(f"百度搜索 {site} 完成，找到 {len(results)} 条结果")
    except Exception as e:
        print(f"百度搜索 {site} 失败: {e}")

    return results


def search(query, max_results_per_site=2):
    """执行搜索，分别获取知乎和CSDN的结果并合并"""
    print(f"执行搜索: {query}")
    try:
        zhihu_results = search_baidu_with_site(query, 'zhihu.com', max_results_per_site)
        csdn_results = search_baidu_with_site(query, 'csdn.net', max_results_per_site)
        all_results = zhihu_results + csdn_results

        results_text = ""
        for i, result in enumerate(all_results, 1):
            results_text += f"结果 {i}（来自{result['source']}）:\n"
            results_text += f"标题: {result['title']}\n"
            results_text += f"链接: {result['url']}\n\n"

        if not all_results:
            return "未找到相关搜索结果，将基于内置知识库回答。", []

        return results_text, all_results

    except Exception as e:
        print(f"网络搜索错误: {str(e)}")
        return f"网络搜索过程中出现错误: {str(e)}", []


def is_relevant_to_os(question, model_mode):
    """判断问题是否与操作系统相关（适配 model.py 的输入格式）"""
    try:
        response_text = ""
        if model_mode == 'local':
            if not local_model or not local_tokenizer:
                print("本地模型未加载")
                return False

            # 按 model.py 要求的格式构造输入（Instruction + 问题 + Output:）
            prompt = f"Instruction: {RELEVANCE_PROMPT.format(question=question)}\nOutput:"
            response_text = generate_answer(
                local_model,
                local_tokenizer,
                prompt,
                max_new_tokens=10,
                temperature=0.0  # 强制确定性输出，避免模糊结果
            )
        else:
            messages = [
                {"role": "system",
                 "content": "你只需要判断问题是否与操作系统相关，只能回答'相关'或'不相关'，不允许其他任何回答"},
                {"role": "user", "content": question}
            ]
            response_text = get_api_answer(messages, max_tokens=10, temperature=0.0)

        cleaned_response = response_text.strip().lower()
        print(f"问题相关性判断原始结果: '{response_text}'，清理后: '{cleaned_response}'")

        if cleaned_response == "相关":
            return True
        elif cleaned_response == "不相关":
            return False
        else:
            print(f"收到非标准的相关性判断结果: {response_text}，默认视为不相关")
            return False

    except Exception as e:
        print(f"判断相关性时出错: {e}")
        return False


@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()

        user_message = data.get('message', '')
        max_tokens = data.get('max_tokens', 2048)
        temperature = data.get('temperature', 0.7)
        enable_web_search = data.get('enable_web_search', False)
        model_mode = data.get('model_mode', DEFAULT_MODEL_MODE)

        search_links = []
        system_prompt = SYSTEM_PROMPT
        user_content = user_message

        # 1. 先判断问题相关性
        is_relevant = is_relevant_to_os(user_message, model_mode)
        print(
            f"问题: {user_message}，相关性判断结果: {'相关' if is_relevant else '不相关'}，是否启用搜索: {enable_web_search}")

        # 2. 处理搜索逻辑
        if enable_web_search:
            if is_relevant:
                print("满足搜索条件，执行搜索...")
                web_search_results, search_links = search(user_message)
                # 构造带搜索结果的用户内容
                user_content = f"问题：{user_message}\n\n网络搜索结果：\n{web_search_results}\n\n请基于以上搜索结果回答问题。"
            else:
                print("问题不相关，不执行搜索")
                user_content = f"问题：{user_message}\n\n该问题与操作系统无关，请直接回答'暂不支持此功能'。"
                search_links = []
        else:
            print("未启用搜索功能")
            user_content = user_message  # 直接用原始问题

        assistant_response = ""
        if model_mode == 'local':
            # 构造最终输入：system_prompt（背景） + Instruction: 用户内容 + Output:（引导生成）
            # 完全适配 model.py 中训练的格式
            final_prompt = f"{system_prompt}\n\nInstruction: {user_content}\nOutput:"
            try:
                if not local_model or not local_tokenizer:
                    return jsonify({'success': False, 'error': '本地模型未加载成功'}), 500

                # 调用 model.py 的 generate_answer，无需额外处理格式
                assistant_response = generate_answer(
                    local_model,
                    local_tokenizer,
                    final_prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature
                )
            except Exception as e:
                print(f"本地模型推理失败: {e}")
                return jsonify({'success': False, 'error': '本地模型推理失败'}), 500
        else:
            # API模式保持不变
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
            try:
                assistant_response = get_api_answer(messages, max_tokens=max_tokens, temperature=temperature)
            except Exception as e:
                print(f"API模型推理失败: {e}")
                return jsonify({'success': False, 'error': 'API模型推理失败'}), 500

        # 过滤不相关回答的链接
        if UNSUPPORTED_RESPONSE in assistant_response:
            search_links = []

        return jsonify({
            'success': True,
            'response': assistant_response,
            'search_links': search_links if enable_web_search else [],
            'parameters': {
                'max_tokens': max_tokens,
                'temperature': temperature,
                'web_search_enabled': enable_web_search,
                'model_mode': model_mode
            }
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    # 启动时预加载本地模型（若为local模式）
    if DEFAULT_MODEL_MODE == "local":
        local_model, local_tokenizer = preload_local_model()
    app.run(host='0.0.0.0', port=5000, debug=True)