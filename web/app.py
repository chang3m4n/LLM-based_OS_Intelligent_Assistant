import os
import random
import sys
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import requests
import urllib.parse
from bs4 import BeautifulSoup

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from inference import generate_answer, load_model_and_tokenizer
from api import get_api_answer

app = Flask(__name__, template_folder='templates')
CORS(app)

# 本地模型缓存
local_model = None
local_tokenizer = None

# 系统提示词
SYSTEM_PROMPT = """你是操作系统课程智能问答助手，名字是比艾特。
- 专注回答和操作系统知识相关的问题；
- 回答语气活泼友好，内容专业易懂"""

# 不支持回答的标识文本
UNSUPPORTED_RESPONSE = "暂不支持此功能"

# 默认模型模式
DEFAULT_MODEL_MODE = os.environ.get('MODEL_MODE', 'api') # 选择api或local模式问答
BASE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct" # 本地模型
LORA_DIR = "../src/finetune_results/final_model"  # 训练生成的LoRA路径
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


def load_local_model():
    """加载本地模型"""
    global local_model, local_tokenizer

    if local_model is not None and local_tokenizer is not None:
        print("模型已加载，跳过重复加载")
        return True

    base_model_name = BASE_MODEL_NAME
    lora_dir = LORA_DIR
    cache_dir = CACHE_DIR
    print("开始加载本地模型...")
    try:
        local_model, local_tokenizer = load_model_and_tokenizer(
            base_model_name=base_model_name,
            lora_dir=lora_dir,
            cache_dir=cache_dir
        )
        print("本地模型加载完成。")
        return True
    except Exception as e:
        print(f"加载本地模型失败: {e}")
        return False


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


@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()

        user_message = data.get('message', '')
        max_tokens = data.get('max_tokens', 1024)
        temperature = data.get('temperature', 0.5)
        enable_web_search = data.get('enable_web_search', False)
        model_mode = data.get('model_mode', DEFAULT_MODEL_MODE)

        search_links = []
        system_prompt = SYSTEM_PROMPT
        user_content = user_message

        # 处理搜索逻辑
        if enable_web_search:
            print("启用搜索功能，执行搜索...")
            web_search_results, search_links = search(user_message)
            # 构造带搜索结果的用户内容
            user_content = f"问题：{user_message}\n\n网络搜索结果：\n{web_search_results}\n\n请基于以上搜索结果回答问题。"
        else:
            print("未启用搜索功能")
            user_content = user_message

        assistant_response = ""
        if model_mode == 'local':
            # 检查模型是否已加载
            if local_model is None or local_tokenizer is None:
                # 如果模型未加载，尝试加载
                if not load_local_model():
                    return jsonify({'success': False, 'error': '本地模型加载失败'}), 500

            try:
                # 调用本地模型推理
                assistant_response = generate_answer(
                    local_model,
                    local_tokenizer,
                    user_content,
                    temperature=temperature,
                    max_new_tokens=max_tokens
                )
            except Exception as e:
                print(f"本地模型推理失败: {e}")
                return jsonify({'success': False, 'error': '本地模型推理失败'}), 500
        else:
            # API模式调用
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


# 初始化时加载模型
if DEFAULT_MODEL_MODE == "local":
    print("应用初始化，尝试加载本地模型...")
    load_local_model()


if __name__ == '__main__':
    # 关闭 Flask 自动重载功能，避免重复加载
    use_reloader = os.environ.get('FLASK_USE_RELOADER', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=use_reloader)
