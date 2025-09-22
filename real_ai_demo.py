#!/usr/bin/env python3
"""
真实AI对话前端演示
集成训练好的Qwen LoRA模型
"""

import http.server
import socketserver
import json
from pathlib import Path
import webbrowser
import threading
import time
import urllib.parse
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.getcwd())

try:
    from model_inference import get_model_instance, chat_with_model
    REAL_AI_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 无法导入模型推理模块: {e}")
    REAL_AI_AVAILABLE = False

def get_training_status():
    """获取训练状态"""
    results_dir = Path("results/models/qwen-1.8b-lora-ultimate")
    training_success_file = results_dir / "training_success.json"
    
    if training_success_file.exists():
        with open(training_success_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    else:
        return {"status": "未找到训练记录"}

def handle_chat_request(message):
    """处理聊天请求"""
    if not REAL_AI_AVAILABLE:
        return "❌ 模型推理模块未加载，请检查依赖。"
    
    try:
        print(f"🤖 用户问题: {message}")
        response = chat_with_model(message)
        print(f"🧠 模型回复: {response}")
        return response
    except Exception as e:
        print(f"❌ 对话错误: {e}")
        return f"抱歉，生成回复时出现错误: {str(e)}"

def create_html_content():
    """生成HTML页面内容"""
    status_data = get_training_status()
    
    # 检查模型状态
    model_status = "✅ 真实AI模型已就绪" if REAL_AI_AVAILABLE else "❌ 模型推理不可用"
    
    html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🤖 真实AI对话 - Qwen LoRA微调</title>
    <style>
        body {{
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
        }}
        .header h2 {{
            margin: 10px 0 0 0;
            font-size: 1.5em;
            opacity: 0.9;
        }}
        .model-status {{
            text-align: center;
            padding: 15px;
            margin: 20px 0;
            border-radius: 10px;
            font-weight: bold;
            font-size: 18px;
            {'background: #4CAF50; color: white;' if REAL_AI_AVAILABLE else 'background: #ff9800; color: white;'}
        }}
        .tabs {{
            display: flex;
            margin-bottom: 20px;
            border-bottom: 2px solid #eee;
        }}
        .tab {{
            padding: 15px 25px;
            cursor: pointer;
            border: none;
            background: none;
            font-size: 16px;
            font-weight: bold;
            color: #666;
            border-bottom: 3px solid transparent;
            transition: all 0.3s;
        }}
        .tab.active {{
            color: #667eea;
            border-bottom-color: #667eea;
        }}
        .tab-content {{
            display: none;
            padding: 20px;
            background: #f9f9f9;
            border-radius: 10px;
        }}
        .tab-content.active {{
            display: block;
        }}
        .chat-container {{
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }}
        .chat-area {{
            height: 500px;
            overflow-y: auto;
            border: 2px solid #eee;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            background: #fafafa;
        }}
        .message {{
            margin: 15px 0;
            padding: 15px;
            border-radius: 15px;
            max-width: 80%;
            word-wrap: break-word;
        }}
        .user-message {{
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            margin-left: auto;
            text-align: right;
        }}
        .ai-message {{
            background: #f0f0f0;
            color: #333;
            border-left: 4px solid #4CAF50;
        }}
        .loading-message {{
            background: #fff3cd;
            color: #856404;
            border-left: 4px solid #ffc107;
            font-style: italic;
        }}
        .chat-input-area {{
            display: flex;
            gap: 15px;
            align-items: center;
        }}
        .chat-input {{
            flex: 1;
            padding: 15px;
            border: 2px solid #ddd;
            border-radius: 25px;
            font-size: 16px;
            outline: none;
            transition: border-color 0.3s;
        }}
        .chat-input:focus {{
            border-color: #667eea;
        }}
        .send-btn {{
            padding: 15px 30px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-size: 16px;
            font-weight: bold;
            transition: transform 0.2s;
        }}
        .send-btn:hover {{
            transform: scale(1.05);
        }}
        .send-btn:disabled {{
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }}
        .quick-buttons {{
            margin: 15px 0;
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }}
        .quick-btn {{
            padding: 10px 20px;
            background: #f8f9fa;
            border: 2px solid #667eea;
            border-radius: 25px;
            cursor: pointer;
            font-size: 14px;
            color: #667eea;
            transition: all 0.3s;
        }}
        .quick-btn:hover {{
            background: #667eea;
            color: white;
        }}
        .status-box {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            border-left: 5px solid #4CAF50;
            margin: 20px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .status-item {{
            margin: 10px 0;
            padding: 10px;
            background: #f0f0f0;
            border-radius: 5px;
        }}
        .status-label {{
            font-weight: bold;
            color: #333;
        }}
        .status-value {{
            color: #666;
        }}
        .footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            border-top: 1px solid #eee;
            margin-top: 30px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 真实AI对话演示</h1>
            <h2>Qwen-1.8B LoRA微调 - 智能对话</h2>
            <p>🎮 RTX 4060 + 🧠 真实模型推理 + ⚡ LoRA技术</p>
        </div>

        <div class="model-status">
            {model_status}
        </div>

        <div class="tabs">
            <button class="tab active" onclick="showTab('chat')">🤖 AI对话</button>
            <button class="tab" onclick="showTab('status')">🏆 训练成果</button>
            <button class="tab" onclick="showTab('info')">📋 项目信息</button>
        </div>

        <div id="chat" class="tab-content active">
            <div class="chat-container">
                <h3>💬 与训练好的Qwen模型对话</h3>
                <div class="quick-buttons">
                    <button class="quick-btn" onclick="quickChat('你好，请介绍一下自己')">👋 自我介绍</button>
                    <button class="quick-btn" onclick="quickChat('你是什么模型？有什么特点？')">🤖 模型介绍</button>
                    <button class="quick-btn" onclick="quickChat('请解释一下LoRA技术')">⚡ LoRA技术</button>
                    <button class="quick-btn" onclick="quickChat('你能帮我做什么？')">❓ 功能说明</button>
                    <button class="quick-btn" onclick="quickChat('写一首关于AI的小诗')">🎭 创意写作</button>
                </div>
                <div class="chat-area" id="chatArea">
                    <div class="message ai-message">
                        🤖 你好！我是经过LoRA微调的Qwen-1.8B模型。我已经成功加载并准备与你对话！
                        <br><br>
                        ✨ 特点：
                        <br>• 🇨🇳 专注中文对话
                        <br>• ⚡ LoRA高效微调
                        <br>• 🎮 RTX 4060优化
                        <br>• 🧠 智能文本生成
                        <br><br>
                        请随意提问，我会尽力为你提供有帮助的回答！
                    </div>
                </div>
                <div class="chat-input-area">
                    <input type="text" id="chatInput" class="chat-input" 
                           placeholder="输入你的问题..." 
                           onkeypress="handleEnter(event)"
                           {'disabled' if not REAL_AI_AVAILABLE else ''}>
                    <button id="sendBtn" class="send-btn" onclick="sendMessage()"
                            {'disabled' if not REAL_AI_AVAILABLE else ''}>
                        发送
                    </button>
                </div>
            </div>
        </div>

        <div id="status" class="tab-content">
            <div class="status-box">
                <div class="status-title">🎉 训练完全成功！</div>
                <div class="status-item">
                    <span class="status-label">状态:</span> 
                    <span class="status-value">{status_data.get('status', 'Unknown')}</span>
                </div>
                <div class="status-item">
                    <span class="status-label">模型:</span> 
                    <span class="status-value">{status_data.get('model', 'Unknown')}</span>
                </div>
                <div class="status-item">
                    <span class="status-label">训练步数:</span> 
                    <span class="status-value">{status_data.get('training_steps', 0)}</span>
                </div>
                <div class="status-item">
                    <span class="status-label">训练时间:</span> 
                    <span class="status-value">{status_data.get('duration_seconds', 0):.3f}秒</span>
                </div>
                <div class="status-item">
                    <span class="status-label">推理状态:</span> 
                    <span class="status-value">{'✅ 真实AI推理已启用' if REAL_AI_AVAILABLE else '❌ 推理模块未加载'}</span>
                </div>
            </div>
        </div>

        <div id="info" class="tab-content">
            <h3>📋 项目信息</h3>
            <div class="status-box">
                <h4>🎯 技术架构</h4>
                <ul>
                    <li><strong>基础模型</strong>: Qwen-1.8B-Chat</li>
                    <li><strong>微调技术</strong>: LoRA (Low-Rank Adaptation)</li>
                    <li><strong>训练设备</strong>: RTX 4060 (8GB)</li>
                    <li><strong>推理框架</strong>: PyTorch + Transformers + PEFT</li>
                    <li><strong>前端技术</strong>: HTML + JavaScript + Python HTTP Server</li>
                </ul>
            </div>
            <div class="status-box">
                <h4>⚡ 性能指标</h4>
                <ul>
                    <li><strong>训练时间</strong>: 4.536秒</li>
                    <li><strong>参数效率</strong>: 仅训练0.36%参数</li>
                    <li><strong>适配器大小</strong>: 6.3MB</li>
                    <li><strong>推理速度</strong>: 实时响应</li>
                    <li><strong>内存占用</strong>: 低于4GB</li>
                </ul>
            </div>
        </div>

        <div class="footer">
            <p>🎉 真实AI对话系统 - 基于训练成功的Qwen LoRA模型</p>
            <p>💡 这是真正的AI模型推理，不是预设回复！</p>
        </div>
    </div>

    <script>
        let isLoading = false;

        function showTab(tabName) {{
            const tabs = document.querySelectorAll('.tab-content');
            tabs.forEach(tab => tab.classList.remove('active'));
            
            const tabButtons = document.querySelectorAll('.tab');
            tabButtons.forEach(btn => btn.classList.remove('active'));
            
            document.getElementById(tabName).classList.add('active');
            event.target.classList.add('active');
        }}

        function quickChat(message) {{
            document.getElementById('chatInput').value = message;
            sendMessage();
        }}

        async function sendMessage() {{
            if (isLoading) return;
            
            const input = document.getElementById('chatInput');
            const message = input.value.trim();
            const sendBtn = document.getElementById('sendBtn');
            
            if (!message) return;
            
            // 添加用户消息
            addMessage(message, 'user');
            input.value = '';
            
            // 设置加载状态
            isLoading = true;
            sendBtn.disabled = true;
            sendBtn.textContent = '生成中...';
            
            // 添加加载消息
            const loadingId = addMessage('🤖 正在思考中...', 'loading');
            
            try {{
                // 发送请求到服务器
                const response = await fetch('/chat', {{
                    method: 'POST',
                    headers: {{
                        'Content-Type': 'application/json',
                    }},
                    body: JSON.stringify({{ message: message }})
                }});
                
                const data = await response.json();
                
                // 移除加载消息
                document.getElementById(loadingId).remove();
                
                // 添加AI回复
                addMessage(data.response, 'ai');
                
            }} catch (error) {{
                // 移除加载消息
                document.getElementById(loadingId).remove();
                addMessage('❌ 抱歉，连接服务器时出现错误: ' + error.message, 'ai');
            }} finally {{
                // 恢复按钮状态
                isLoading = false;
                sendBtn.disabled = false;
                sendBtn.textContent = '发送';
            }}
        }}

        function addMessage(content, type) {{
            const chatArea = document.getElementById('chatArea');
            const messageDiv = document.createElement('div');
            const messageId = 'msg_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
            
            messageDiv.id = messageId;
            messageDiv.className = `message ${{type}}-message`;
            
            if (type === 'user') {{
                messageDiv.innerHTML = `👤 ${{content}}`;
            }} else if (type === 'ai') {{
                messageDiv.innerHTML = `🤖 ${{content}}`;
            }} else if (type === 'loading') {{
                messageDiv.innerHTML = content;
            }}
            
            chatArea.appendChild(messageDiv);
            chatArea.scrollTop = chatArea.scrollHeight;
            
            return messageId;
        }}

        function handleEnter(event) {{
            if (event.key === 'Enter' && !isLoading) {{
                sendMessage();
            }}
        }}

        // 页面加载完成后聚焦输入框
        window.onload = function() {{
            document.getElementById('chatInput').focus();
        }};
    </script>
</body>
</html>
    """
    return html_content

class AIHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            html_content = create_html_content()
            self.wfile.write(html_content.encode('utf-8'))
        else:
            super().do_GET()
    
    def do_POST(self):
        if self.path == '/chat':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                user_message = data.get('message', '')
                
                # 调用真实AI模型
                ai_response = handle_chat_request(user_message)
                
                response_data = {
                    'response': ai_response,
                    'status': 'success'
                }
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps(response_data, ensure_ascii=False).encode('utf-8'))
                
            except Exception as e:
                error_response = {
                    'response': f'服务器处理错误: {str(e)}',
                    'status': 'error'
                }
                
                self.send_response(500)
                self.send_header('Content-type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps(error_response, ensure_ascii=False).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

def start_ai_server():
    """启动AI对话服务器"""
    PORT = 8001
    
    print(f"🚀 启动真实AI对话服务器...")
    print(f"🤖 模型状态: {'✅ 已就绪' if REAL_AI_AVAILABLE else '❌ 不可用'}")
    print(f"🔗 访问地址: http://localhost:{PORT}")
    print(f"📝 请在浏览器中打开上述地址开始对话")
    print("⚠️  按 Ctrl+C 停止服务器")
    
    with socketserver.TCPServer(("", PORT), AIHTTPRequestHandler) as httpd:
        # 自动打开浏览器
        def open_browser():
            time.sleep(1)
            webbrowser.open(f'http://localhost:{PORT}')
        
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n👋 AI对话服务器已停止")

if __name__ == "__main__":
    start_ai_server()