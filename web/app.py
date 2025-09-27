import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import requests
import urllib.parse
from inference import generate_answer,load_model_and_tokenizer
from api import get_api_answer

app = Flask(__name__, template_folder='templates')
CORS(app)

# 本地模型缓存
local_model = None
local_tokenizer = None

# 系统提示词
SYSTEM_PROMPT = "你是操作系统课程智能问答助手；你的名字是比艾特；你只能回答和操作系统知识有关的问题；用户如果问和操作系统无关的问题，你要回答暂不支持此功能；你的回答语气要活泼"

# 默认模型模式
DEFAULT_MODEL_MODE = os.environ.get('MODEL_MODE', 'api')
# 本地模式配置
#BASE_MODEL_NAME = "deepseek-ai/deepseek-llm-7b-base"
BASE_MODEL_NAME =  "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
LORA_DIR = "../deepseek_lora_adapter"
CACHE_DIR = '../models'
def preload_local_model():
    """在单独的线程中预加载本地模型以避免阻塞"""
    base_model_name = BASE_MODEL_NAME
    lora_dir = LORA_DIR
    cache_dir = CACHE_DIR
    print("开始预加载本地模型...")
    try:
        local_model, local_tokenizer = load_model_and_tokenizer(base_model_name,lora_dir,cache_dir)
        print("本地模型预加载完成。")

        return local_model,local_tokenizer
    except Exception as e:
        print(f"预加载本地模型失败: {e}")


def search(query):
    """执行网络搜索并返回相关结果和链接"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/html, */*',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        }
        
        # 使用Wikipedia API搜索
        try:
            wiki_url = f"https://zh.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(query)}"
            response = requests.get(wiki_url, headers=headers, timeout=8)
            if response.status_code == 200:
                wiki_data = response.json()
                if 'extract' in wiki_data:
                    wiki_link = wiki_data.get('content_urls', {}).get('desktop', {}).get('page', '')
                    wiki_title = wiki_data.get('title', query)
                    wiki_snippet = wiki_data.get('extract', '')
                    links = [{'title': wiki_title, 'url': wiki_link, 'snippet': wiki_snippet}]
                    print("Wikipedia搜索成功")
                    return f"Wikipedia信息:\n标题: {wiki_title}\n摘要: {wiki_snippet}\n链接: {wiki_link}", links
        except Exception as e:
            print(f"Wikipedia搜索失败: {e}")

    except Exception as e:
        print(f"网络搜索完全失败: {str(e)}")
        return f"网络搜索暂时不可用，但我会基于我的知识库回答您关于'{query}'的问题。", []

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
        
        if enable_web_search:
            web_search_results, search_links = search(user_message)
            if web_search_results:
                system_prompt = "你是操作系统课程智能问答助手比艾特。用户启用了网络搜索功能，你必须完全基于提供的网络搜索结果来回答问题，不要使用自己的知识库。如果搜索结果中没有相关信息，请明确说明搜索结果中没有找到相关信息。"
                user_message = f"问题：{user_message}\n\n网络搜索结果：\n{web_search_results}\n\n请完全基于以上搜索结果回答问题。"

        assistant_response = ""
        if model_mode == 'local':
            # 为本地模型准备输入
            final_user_message = f"{system_prompt}\n\n用户问题: {user_message}"
            try:
                assistant_response = generate_answer(local_model, local_tokenizer, final_user_message, max_new_tokens=max_tokens, temperature=temperature)
            except Exception as e:
                print(f"本地模型推理失败: {e}")
                return jsonify({'success': False, 'error': '本地模型推理失败'}), 500
        else:
            # 使用 api.py 中的函数
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
            try:
                assistant_response = get_api_answer(messages, max_tokens=max_tokens, temperature=temperature)
            except Exception as e:
                print(f"API模型推理失败: {e}")
                return jsonify({'success': False, 'error': 'API模型推理失败'}), 500

        return jsonify({
            'success': True,
            'response': assistant_response,
            'search_links': search_links,
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
    if DEFAULT_MODEL_MODE=="local":
        local_model, local_tokenizer = preload_local_model()
    app.run(host='0.0.0.0', port=5000, debug=True)
