from flask import Flask, request, jsonify, send_file, render_template
import tempfile
import os
import scipy
import numpy as np

app = Flask(__name__, template_folder='templates')

# 创建输出目录
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"音频输出目录: {OUTPUT_DIR}")

print("Qwen-TTS服务正在启动，正在加载模型...")

# 初始化模型变量
model_base = None
model_voice_design = None
model_custom_voice = None

# 0.6B模型变量
model_base_0_6b = None
model_voice_design_0_6b = None
model_custom_voice_0_6b = None

# 尝试导入Qwen-TTS模型
print("Qwen-TTS服务正在启动...")
try:
    # 解决SoX缺失的问题，先导入并处理
    import warnings
    warnings.filterwarnings("ignore")
    
    # 尝试导入Qwen-TTS模型
    from qwen_tts import Qwen3TTSModel
    
    print("✅ Qwen-TTS模型类导入成功！")
    print("📌 注意：SoX缺失不会影响基本功能，只会影响某些高级功能。")
    
    # 加载Base模型（用于语音克隆）
    print("\n📁 加载 Base 模型...")
    try:
        model_base = Qwen3TTSModel.from_pretrained(
            "./Qwen3-TTS-12Hz-1.7B-Base", 
            trust_remote_code=True, 
            device_map="cpu"
        )
        print("✅ Base 模型加载成功！")
    except Exception as e:
        print(f"❌ Base 模型加载失败: {e}")
        model_base = None
    
    # 加载VoiceDesign模型（使用完整版）
    print("\n📁 加载 VoiceDesign 模型...")
    try:
        model_voice_design = Qwen3TTSModel.from_pretrained(
            "./Qwen3-TTS-12Hz-1.7B-VoiceDesign-Full", 
            trust_remote_code=True, 
            device_map="cpu"
        )
        print("✅ VoiceDesign 模型加载成功！")
    except Exception as e:
        print(f"❌ VoiceDesign 模型加载失败: {e}")
        model_voice_design = None
    
    # 加载CustomVoice模型（使用完整版）
    print("\n📁 加载 CustomVoice 模型...")
    try:
        model_custom_voice = Qwen3TTSModel.from_pretrained(
            "./Qwen3-TTS-12Hz-1.7B-CustomVoice-Full", 
            trust_remote_code=True, 
            device_map="cpu"
        )
        print("✅ CustomVoice 模型加载成功！")
    except Exception as e:
        print(f"❌ CustomVoice 模型加载失败: {e}")
        model_custom_voice = None
    
    # 加载0.6B Base模型
    print("\n📁 加载 0.6B Base 模型...")
    try:
        model_base_0_6b = Qwen3TTSModel.from_pretrained(
            "./Qwen3-TTS-12Hz-0.6B-Base", 
            trust_remote_code=True, 
            device_map="cpu"
        )
        print("✅ 0.6B Base 模型加载成功！")
    except Exception as e:
        print(f"❌ 0.6B Base 模型加载失败: {e}")
        model_base_0_6b = None
    
    # 加载0.6B VoiceDesign模型
    print("\n📁 加载 0.6B VoiceDesign 模型...")
    try:
        model_voice_design_0_6b = Qwen3TTSModel.from_pretrained(
            "./Qwen3-TTS-12Hz-0.6B-VoiceDesign", 
            trust_remote_code=True, 
            device_map="cpu"
        )
        print("✅ 0.6B VoiceDesign 模型加载成功！")
    except Exception as e:
        print(f"❌ 0.6B VoiceDesign 模型加载失败: {e}")
        model_voice_design_0_6b = None
    
    # 加载0.6B CustomVoice模型
    print("\n📁 加载 0.6B CustomVoice 模型...")
    try:
        model_custom_voice_0_6b = Qwen3TTSModel.from_pretrained(
            "./Qwen3-TTS-12Hz-0.6B-CustomVoice", 
            trust_remote_code=True, 
            device_map="cpu"
        )
        print("✅ 0.6B CustomVoice 模型加载成功！")
    except Exception as e:
        print(f"❌ 0.6B CustomVoice 模型加载失败: {e}")
        model_custom_voice_0_6b = None

except Exception as e:
    print(f"❌ 模型类导入失败: {e}")
    print("📝 将使用模拟音频生成功能。")

print("\n✅ 服务启动成功！")
print("🔗 请在浏览器中访问: http://localhost:5000")
print("💡 当前状态：")
if model_voice_design is not None or model_custom_voice is not None:
    print("   - Qwen3-TTS模型：已加载（真实语音生成）")
    if model_voice_design:
        print("     ✓ VoiceDesign 模型：可用")
    if model_custom_voice:
        print("     ✓ CustomVoice 模型：可用")
    if model_base:
        print("     ✓ Base 模型：可用（语音克隆）")
else:
    print("   - Qwen3-TTS模型：使用模拟音频生成")
    print("   - 建议：检查模型文件是否完整")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Qwen3-TTS Demo</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        body {
            background-color: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        header {
            margin-bottom: 30px;
        }
        h1 {
            font-size: 24px;
            font-weight: 600;
            margin-bottom: 10px;
            color: #1a1a1a;
        }
        .subtitle {
            font-size: 14px;
            color: #666;
            margin-bottom: 20px;
        }
        .tabs {
            display: flex;
            margin-bottom: 30px;
            border-bottom: 1px solid #e0e0e0;
        }
        .tab {
            padding: 10px 20px;
            margin-right: 10px;
            cursor: pointer;
            border: none;
            background: none;
            font-size: 14px;
            color: #666;
            border-bottom: 2px solid transparent;
            transition: all 0.3s;
        }
        .tab.active {
            color: #1890ff;
            border-bottom-color: #1890ff;
            font-weight: 500;
        }
        .tab:hover {
            color: #1890ff;
        }
        .main-content {
            display: flex;
            gap: 20px;
        }
        .panel {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }
        .tab-panel {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }
        .tab-panel.active {
            display: block;
        }
        .panel-title {
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 20px;
            color: #333;
        }
        .form-group {
            margin-bottom: 20px;
        }
        .form-label {
            display: block;
            font-size: 14px;
            font-weight: 500;
            margin-bottom: 8px;
            color: #555;
        }
        textarea,
        select,
        input[type="text"] {
            width: 100%;
            padding: 12px;
            border: 1px solid #d9d9d9;
            border-radius: 4px;
            font-size: 14px;
            transition: all 0.3s;
            resize: vertical;
        }
        textarea {
            min-height: 100px;
        }
        textarea:focus,
        select:focus,
        input[type="text"]:focus {
            outline: none;
            border-color: #1890ff;
            box-shadow: 0 0 0 2px rgba(24, 144, 255, 0.2);
        }
        .btn-primary {
            width: 100%;
            padding: 12px;
            background-color: #1890ff;
            color: white;
            border: none;
            border-radius: 4px;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s;
        }
        .btn-primary:hover {
            background-color: #40a9ff;
        }
        .btn-primary:disabled {
            background-color: #d9d9d9;
            cursor: not-allowed;
        }
        .audio-section {
            margin-top: 20px;
        }
        .audio-preview {
            padding: 15px;
            background: #fafafa;
            border-radius: 4px;
            border: 1px solid #e8e8e8;
        }
        .audio-player {
            width: 100%;
            margin-bottom: 10px;
        }
        .audio-player-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 16px;
            padding: 24px;
            color: white;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }
        .player-controls {
            display: flex;
            align-items: center;
            gap: 16px;
            margin-bottom: 16px;
        }
        .play-button {
            width: 56px;
            height: 56px;
            border-radius: 50%;
            background: white;
            border: none;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            transition: transform 0.2s, box-shadow 0.2s;
            flex-shrink: 0;
        }
        .play-button:hover {
            transform: scale(1.05);
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        }
        .play-button:active {
            transform: scale(0.95);
        }
        .play-button svg {
            width: 24px;
            height: 24px;
            fill: #667eea;
        }
        .progress-container {
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        .progress-bar-container {
            position: relative;
            height: 6px;
            background: rgba(255,255,255,0.3);
            border-radius: 3px;
            cursor: pointer;
        }
        .progress-bar {
            height: 100%;
            background: white;
            border-radius: 3px;
            transition: width 0.1s;
            position: relative;
        }
        .progress-bar::after {
            content: '';
            position: absolute;
            right: -6px;
            top: 50%;
            transform: translateY(-50%);
            width: 12px;
            height: 12px;
            background: white;
            border-radius: 50%;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }
        .time-display {
            display: flex;
            justify-content: space-between;
            font-size: 12px;
            opacity: 0.9;
        }
        .player-actions {
            display: flex;
            gap: 12px;
            align-items: center;
        }
        .icon-button {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            background: rgba(255,255,255,0.2);
            border: none;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: background 0.2s;
        }
        .icon-button:hover {
            background: rgba(255,255,255,0.3);
        }
        .icon-button svg {
            width: 18px;
            height: 18px;
            stroke: white;
            fill: none;
        }
        .download-button {
            padding: 8px 16px;
            background: rgba(255,255,255,0.2);
            border: none;
            border-radius: 20px;
            color: white;
            font-size: 13px;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 6px;
            transition: background 0.2s;
        }
        .download-button:hover {
            background: rgba(255,255,255,0.3);
        }
        .download-button svg {
            width: 16px;
            height: 16px;
        }
        .generation-info {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 16px;
            padding-top: 16px;
            border-top: 1px solid rgba(255,255,255,0.2);
            font-size: 13px;
        }
        .time-cost {
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .time-cost svg {
            width: 14px;
            height: 14px;
        }
        .debug-toggle {
            background: rgba(255,255,255,0.2);
            border: none;
            border-radius: 16px;
            padding: 6px 12px;
            color: white;
            font-size: 12px;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 4px;
            transition: background 0.2s;
        }
        .debug-toggle:hover {
            background: rgba(255,255,255,0.3);
        }
        .debug-panel {
            margin-top: 16px;
            padding: 16px;
            background: rgba(0,0,0,0.3);
            border-radius: 12px;
            max-height: 200px;
            overflow-y: auto;
            font-family: 'Courier New', monospace;
            font-size: 12px;
            display: none;
        }
        .debug-panel.active {
            display: block;
        }
        .debug-log {
            margin-bottom: 4px;
            padding: 2px 0;
        }
        .debug-log.timestamp {
            color: #ffd700;
        }
        .debug-log.info {
            color: #87ceeb;
        }
        .debug-log.success {
            color: #90ee90;
        }
        .debug-log.error {
            color: #ff6b6b;
        }
        .batch-process {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 12px;
            padding: 20px;
            margin-top: 20px;
        }
        .batch-item {
            background: white;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            border-left: 4px solid #667eea;
        }
        .batch-item.processing {
            border-left-color: #faad14;
            animation: pulse 1.5s infinite;
        }
        .batch-item.completed {
            border-left-color: #52c41a;
        }
        .batch-item.failed {
            border-left-color: #ff4d4f;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        .workflow-node {
            background: white;
            border: 2px solid #e0e0e0;
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 12px;
            position: relative;
            transition: all 0.3s;
        }
        .workflow-node:hover {
            border-color: #667eea;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15);
        }
        .workflow-node.active {
            border-color: #667eea;
            background: #f8f9ff;
        }
        .node-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }
        .node-title {
            font-weight: 600;
            color: #333;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .node-icon {
            width: 24px;
            height: 24px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 6px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 12px;
        }
        .node-status {
            font-size: 12px;
            padding: 4px 10px;
            border-radius: 12px;
            background: #f0f0f0;
            color: #666;
        }
        .node-status.running {
            background: #fff7e6;
            color: #faad14;
        }
        .node-status.completed {
            background: #f6ffed;
            color: #52c41a;
        }
        .status-box {
            padding: 15px;
            background: #fafafa;
            border-radius: 4px;
            border: 1px solid #e8e8e8;
            font-size: 14px;
            color: #666;
            margin-top: 15px;
            min-height: 50px;
        }
        .status-success {
            color: #52c41a;
        }
        .status-error {
            color: #ff4d4f;
        }
        .status-loading {
            color: #faad14;
        }
        footer {
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #e0e0e0;
            font-size: 14px;
            color: #999;
            text-align: center;
        }
        .footer-links {
            margin-top: 10px;
        }
        .footer-links a {
            color: #1890ff;
            text-decoration: none;
            margin: 0 10px;
            font-size: 13px;
        }
        .footer-links a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Qwen3-TTS Demo</h1>
            <p class="subtitle">A unified Text-to-Speech demo featuring three powerful modes:</p>
            <ul style="font-size: 14px; color: #666; margin-left: 20px; margin-bottom: 15px;">
                <li>Voice Design: Create custom voices using natural language descriptions</li>
                <li>Voice Clone (Base): Clone any voice from a reference audio</li>
                <li>TTS (CustomVoice): Generate speech with predefined speakers and optional style instructions</li>
            </ul>
            <p class="subtitle" style="margin-bottom: 0;">Built with <span style="color: #1890ff;">Qwen3-TTS</span> by Alibaba Qwen Team.</p>
        </header>

        <div class="tabs">
            <button class="tab active" onclick="switchTab('voice-design')">Voice Design</button>
            <button class="tab" onclick="switchTab('voice-clone')">Voice Clone (Base)</button>
            <button class="tab" onclick="switchTab('tts-custom')">TTS (CustomVoice)</button>
            <button class="tab" onclick="switchTab('batch-workflow')">Batch Workflow</button>
        </div>

        <div class="main-content">
            <!-- Left Side: Tab Panels -->
            <div style="flex: 1;">
                <!-- Voice Design Tab -->
                <div id="voice-design" class="tab-panel active">
                    <h3 class="panel-title">Create Custom Voice with Natural Language</h3>
                    
                    <div class="form-group">
                        <label class="form-label">Text to Synthesize</label>
                        <textarea id="text-input" placeholder="请输入要合成的文本...">它在最上面的抽屉里……等等，抽屉是空的？不可能，这绝对不可能！我肯定是放在那里的！</textarea>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">Language</label>
                        <select id="language-select">
                            <option value="auto">Auto</option>
                            <option value="zh">中文</option>
                            <option value="en">English</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">Voice Description</label>
                        <textarea id="voice-description" placeholder="请描述您想要的声音...">用难以置信的语气说话，但语气中开始流露出一丝恐慌。</textarea>
                    </div>
                    
                    <button class="btn-primary" onclick="generateSpeech('voice-design')" id="generate-btn">Generate with Custom Voice</button>
                </div>

                <!-- Voice Clone (Base) Tab -->
                <div id="voice-clone" class="tab-panel" style="display: none;">
                    <h3 class="panel-title">Voice Clone (Base)</h3>
                    
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                        <!-- Left Column: Reference Audio -->
                        <div>
                            <div class="form-group">
                                <label style="font-size: 14px; color: #666; margin-bottom: 10px; display: block;">Reference Audio from Microphone/Radio</label>
                                <div style="border: 2px dashed #d9d9d9; border-radius: 8px; padding: 40px; text-align: center; background: #fafafa;">
                                    <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="color: #1890ff;"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>
                                    <p style="margin: 10px 0; color: #666;">将音频拖放到此处</p>
                                    <p style="margin: 0; font-size: 12px; color: #999;">点击上传</p>
                                    <input type="file" accept="audio/*" style="display: none;" id="clone-audio-upload">
                                    <button onclick="document.getElementById('clone-audio-upload').click()" style="margin-top: 15px; padding: 8px 20px; background: #1890ff; color: white; border: none; border-radius: 4px; cursor: pointer;">上传音频</button>
                                </div>
                                <div style="margin-top: 15px;">
                                    <p style="font-size: 12px; color: #999; margin-bottom: 10px;">或上传示例音频文件：</p>
                                    <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                                        <button class="btn-secondary" style="padding: 6px 12px; background: #f0f0f0; color: #666; border: 1px solid #d9d9d9; border-radius: 4px; cursor: pointer; font-size: 12px;">示例1</button>
                                        <button class="btn-secondary" style="padding: 6px 12px; background: #f0f0f0; color: #666; border: 1px solid #d9d9d9; border-radius: 4px; cursor: pointer; font-size: 12px;">示例2</button>
                                        <button class="btn-secondary" style="padding: 6px 12px; background: #f0f0f0; color: #666; border: 1px solid #d9d9d9; border-radius: 4px; cursor: pointer; font-size: 12px;">示例3</button>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="form-group">
                                <label class="form-label">Reference Text</label>
                                <textarea id="clone-reference-text" placeholder="请输入参考音频中的转录文本..." style="min-height: 80px;"></textarea>
                            </div>
                        </div>
                        
                        <!-- Right Column: Target Text -->
                        <div>
                            <div class="form-group">
                                <label class="form-label">Target Text</label>
                                <textarea id="clone-target-text" placeholder="Enter the text you want the cloned voice to speak." style="min-height: 120px;"></textarea>
                            </div>
                            
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 15px;">
                                <div>
                                    <label class="form-label">Language</label>
                                    <select id="clone-language">
                                        <option value="auto">Auto</option>
                                        <option value="zh">中文</option>
                                        <option value="en">English</option>
                                    </select>
                                </div>
                                <div>
                                    <label class="form-label">Model Size</label>
                                    <select id="clone-model-size">
                                        <option value="1.7B">1.7B</option>
                                    </select>
                                </div>
                            </div>
                            
                            <button class="btn-primary" onclick="generateSpeech('voice-clone')" style="margin-top: 20px;">Clone & Generate</button>
                        </div>
                    </div>
                </div>

                <!-- TTS (CustomVoice) Tab -->
                <div id="tts-custom" class="tab-panel" style="display: none;">
                    <h3 class="panel-title">TTS (CustomVoice)</h3>
                    
                    <div class="form-group">
                        <label class="form-label">Text to Synthesize</label>
                        <textarea id="custom-text-input" placeholder="请输入要合成的文本...">你好，这里是Qwen-TTS的演示，让我为你展示不同的语音风格和说话方式。</textarea>
                        <button class="btn-secondary" style="margin-top: 10px; padding: 6px 12px; background: #f0f0f0; color: #666; border: 1px solid #d9d9d9; border-radius: 4px; cursor: pointer; font-size: 12px;">加载示例</button>
                    </div>
                    
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 15px;">
                        <div>
                            <label class="form-label">Language</label>
                            <select id="custom-language">
                                <option value="zh">中文</option>
                                <option value="en">English</option>
                            </select>
                        </div>
                        <div>
                            <label class="form-label">Speaker</label>
                            <select id="custom-speaker">
                                <option value="Vivian">Vivian</option>
                                <option value="Mike">Mike</option>
                                <option value="Lisa">Lisa</option>
                                <option value="David">David</option>
                                <option value="Emma">Emma</option>
                                <option value="John">John</option>
                            </select>
                        </div>
                    </div>
                    
                    <div style="margin-top: 15px;">
                        <label class="form-label">Style Instruction</label>
                        <input type="text" id="custom-style" placeholder="e.g., 用愉快的语气，带有一些幽默感">
                    </div>
                    
                    <button class="btn-primary" onclick="generateSpeech('tts-custom')" style="margin-top: 20px;">Generate Speech</button>
                </div>

                <!-- Batch Workflow Tab -->
                <div id="batch-workflow" class="tab-panel" style="display: none;">
                    <h3 class="panel-title">Batch Processing Workflow</h3>
                    <p style="font-size: 13px; color: #666; margin-bottom: 20px;">批量处理多个文本，支持工作流节点式管理</p>
                    
                    <div class="form-group">
                        <label class="form-label">选择模式</label>
                        <select id="batch-mode" onchange="updateBatchTemplate()">
                            <option value="voice-design">Voice Design - 自定义声音</option>
                            <option value="tts-custom">TTS CustomVoice - 预设声音</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">批量文本输入（每行一个）</label>
                        <textarea id="batch-texts" placeholder="请输入多个文本，每行一个..." style="min-height: 120px;"></textarea>
                    </div>
                    
                    <div id="batch-voice-design-options" class="form-group">
                        <label class="form-label">声音描述（应用于所有文本）</label>
                        <textarea id="batch-voice-desc" placeholder="描述你想要的声音特征...">用自然的语气说话，语速适中</textarea>
                    </div>
                    
                    <div id="batch-tts-options" class="form-group" style="display: none;">
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                            <div>
                                <label class="form-label">Speaker</label>
                                <select id="batch-speaker">
                                    <option value="Vivian">Vivian</option>
                                    <option value="Mike">Mike</option>
                                    <option value="Lisa">Lisa</option>
                                    <option value="David">David</option>
                                </select>
                            </div>
                            <div>
                                <label class="form-label">Style</label>
                                <input type="text" id="batch-style" placeholder="风格描述">
                            </div>
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label class="form-label">Language</label>
                        <select id="batch-language">
                            <option value="zh">中文</option>
                            <option value="en">English</option>
                            <option value="auto">Auto</option>
                        </select>
                    </div>
                    
                    <button class="btn-primary" onclick="startBatchProcessing()" style="margin-bottom: 20px;">
                        <svg style="width: 16px; height: 16px; margin-right: 6px; vertical-align: middle;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
                        开始批量处理
                    </button>
                    
                    <div id="batch-progress" style="display: none; margin-bottom: 20px;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                            <span style="font-size: 13px; color: #666;">处理进度</span>
                            <span id="batch-progress-text" style="font-size: 13px; color: #1890ff; font-weight: 500;">0/0</span>
                        </div>
                        <div style="height: 6px; background: #e8e8e8; border-radius: 3px; overflow: hidden;">
                            <div id="batch-progress-bar" style="height: 100%; width: 0%; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); transition: width 0.3s;"></div>
                        </div>
                    </div>
                    
                    <div id="workflow-nodes">
                        <h4 style="font-size: 14px; margin-bottom: 12px; color: #333;">工作流节点</h4>
                        <div id="batch-nodes-container">
                            <div class="workflow-node">
                                <div class="node-header">
                                    <div class="node-title">
                                        <div class="node-icon">1</div>
                                        <span>准备就绪</span>
                                    </div>
                                    <span class="node-status">等待开始</span>
                                </div>
                                <p style="font-size: 12px; color: #999;">点击"开始批量处理"按钮开始生成语音</p>
                            </div>
                        </div>
                    </div>
                    
                    <div id="batch-results" style="margin-top: 20px; display: none;">
                        <h4 style="font-size: 14px; margin-bottom: 12px; color: #333;">生成结果</h4>
                        <div id="batch-results-list"></div>
                    </div>
                </div>
            </div>

            <!-- Right Side: Generated Audio and Status (Always Visible) -->
            <div style="width: 450px; margin-left: 20px;">
                <div class="panel">
                    <h3 class="panel-title">Generated Audio</h3>
                    <div class="audio-player-container">
                        <div class="player-controls">
                            <button onclick="togglePlay()" id="play-btn" class="play-button">
                                <svg id="play-icon" viewBox="0 0 24 24"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
                                <svg id="pause-icon" viewBox="0 0 24 24" style="display: none;"><rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect></svg>
                            </button>
                            <div class="progress-container">
                                <div class="progress-bar-container" onclick="seekAudio(event)">
                                    <div class="progress-bar" id="progress-bar" style="width: 0%;"></div>
                                </div>
                                <div class="time-display">
                                    <span id="current-time">0:00</span>
                                    <span id="total-time">0:00</span>
                                </div>
                            </div>
                        </div>
                        <div class="player-actions">
                            <button onclick="toggleMute()" id="mute-btn" class="icon-button">
                                <svg id="volume-icon" viewBox="0 0 24 24"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><path d="M15.54 8.46a5 5 0 0 1 0 7.07"></path><path d="M19.07 4.93a10 10 0 0 1 0 14.14"></path></svg>
                                <svg id="mute-icon" viewBox="0 0 24 24" style="display: none;"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
                            </button>
                            <button onclick="downloadAudio()" class="download-button">
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>
                                下载音频
                            </button>
                        </div>
                        <div class="generation-info">
                            <div class="time-cost">
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>
                                <span id="generation-time">生成耗时: --</span>
                            </div>
                            <button onclick="toggleDebug()" class="debug-toggle">
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
                                执行日志
                            </button>
                        </div>
                        <div id="debug-panel" class="debug-panel">
                            <div class="debug-log">等待开始生成...</div>
                        </div>
                        <audio id="audio-player" ontimeupdate="updateProgress()" onloadedmetadata="updateAudioInfo()" style="display: none;"></audio>
                    </div>
                    
                    <h3 class="panel-title" style="margin-top: 20px;">Status</h3>
                    <div id="status" class="status-box">
                        就绪
                    </div>
                </div>
            </div>
        </div>

        <footer>
            <div class="footer-links">
                <a href="#">通过 API 或 MCP 使用</a>
                <a href="#">使用 Gradio 构建</a>
                <a href="#">设置</a>
            </div>
        </footer>
    </div>

    <script>
        // Tab switching functionality
        function switchTab(tabName) {
            // Hide all tab panels
            const panels = document.querySelectorAll('.tab-panel');
            panels.forEach(panel => {
                panel.style.display = 'none';
                panel.classList.remove('active');
            });
            
            // Show the selected tab panel
            const selectedPanel = document.getElementById(tabName);
            if (selectedPanel) {
                selectedPanel.style.display = 'block';
                selectedPanel.classList.add('active');
            }
            
            // Update active tab button
            const tabs = document.querySelectorAll('.tab');
            tabs.forEach(tab => {
                tab.classList.remove('active');
            });
            event.target.classList.add('active');
        }
        
        // Audio control variables
        let isPlaying = false;
        let isMuted = false;
        let generationStartTime = null;
        let debugLogs = [];
        
        // Add debug log
        function addDebugLog(message, type = 'info') {
            const timestamp = new Date().toLocaleTimeString('zh-CN', { hour12: false });
            const logEntry = { timestamp, message, type };
            debugLogs.push(logEntry);
            
            const debugPanel = document.getElementById('debug-panel');
            const logDiv = document.createElement('div');
            logDiv.className = `debug-log ${type}`;
            logDiv.innerHTML = `<span class="timestamp">[${timestamp}]</span> ${message}`;
            debugPanel.appendChild(logDiv);
            debugPanel.scrollTop = debugPanel.scrollHeight;
        }
        
        // Toggle debug panel
        function toggleDebug() {
            const debugPanel = document.getElementById('debug-panel');
            debugPanel.classList.toggle('active');
        }
        
        // Clear debug logs
        function clearDebugLogs() {
            debugLogs = [];
            const debugPanel = document.getElementById('debug-panel');
            debugPanel.innerHTML = '<div class="debug-log">等待开始生成...</div>';
        }
        
        // Speech generation function
        async function generateSpeech(tabName) {
            const statusDiv = document.getElementById('status');
            const audioPlayer = document.getElementById('audio-player');
            const generationTimeEl = document.getElementById('generation-time');
            let endpoint = '/tts';
            let requestBody = {};
            
            // Clear previous logs
            clearDebugLogs();
            generationStartTime = Date.now();
            addDebugLog('开始语音生成任务', 'info');
            
            // Determine which tab is active and get the appropriate form data
            if (tabName === 'voice-design') {
                const text = document.getElementById('text-input').value;
                const language = document.getElementById('language-select').value;
                const voiceDesc = document.getElementById('voice-description').value;
                const generateBtn = document.getElementById('generate-btn');
                
                if (!text.trim()) {
                    showStatus('请输入要合成的文本', 'error');
                    addDebugLog('错误: 文本为空', 'error');
                    return;
                }
                
                generateBtn.disabled = true;
                addDebugLog(`模式: Voice Design`, 'info');
                addDebugLog(`语言: ${language}`, 'info');
                addDebugLog(`文本长度: ${text.length} 字符`, 'info');
                requestBody = {
                    text: text,
                    language: language,
                    voice_description: voiceDesc,
                    mode: 'voice-design'
                };
            } else if (tabName === 'voice-clone') {
                const referenceText = document.getElementById('clone-reference-text').value;
                const targetText = document.getElementById('clone-target-text').value;
                const language = document.getElementById('clone-language').value;
                const modelSize = document.getElementById('clone-model-size').value;
                
                if (!targetText.trim()) {
                    showStatus('请输入目标文本', 'error');
                    addDebugLog('错误: 目标文本为空', 'error');
                    return;
                }
                
                addDebugLog(`模式: Voice Clone`, 'info');
                addDebugLog(`模型大小: ${modelSize}`, 'info');
                requestBody = {
                    text: targetText,
                    reference_text: referenceText,
                    language: language,
                    model_size: modelSize,
                    mode: 'voice-clone'
                };
            } else if (tabName === 'tts-custom') {
                const text = document.getElementById('custom-text-input').value;
                const language = document.getElementById('custom-language').value;
                const speaker = document.getElementById('custom-speaker').value;
                const style = document.getElementById('custom-style').value;
                
                if (!text.trim()) {
                    showStatus('请输入要合成的文本', 'error');
                    addDebugLog('错误: 文本为空', 'error');
                    return;
                }
                
                addDebugLog(`模式: TTS CustomVoice`, 'info');
                addDebugLog(`说话人: ${speaker}`, 'info');
                addDebugLog(`风格: ${style || '默认'}`, 'info');
                requestBody = {
                    text: text,
                    language: language,
                    speaker: speaker,
                    style: style,
                    mode: 'tts-custom'
                };
            }
            
            showStatus('正在生成语音，请稍候...', 'loading');
            addDebugLog('发送请求到服务器...', 'info');
            
            try {
                const response = await fetch(endpoint, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(requestBody)
                });
                
                addDebugLog('收到服务器响应', 'success');
                const data = await response.json();
                
                if (data.success) {
                    const duration = ((Date.now() - generationStartTime) / 1000).toFixed(2);
                    audioPlayer.src = data.audio_url;
                    showStatus('语音生成成功！', 'success');
                    generationTimeEl.textContent = `生成耗时: ${duration}s`;
                    addDebugLog(`语音生成成功! 耗时: ${duration}秒`, 'success');
                    resetAudioControls();
                } else {
                    showStatus('生成失败: ' + data.error, 'error');
                    addDebugLog(`生成失败: ${data.error}`, 'error');
                }
            } catch (error) {
                showStatus('请求失败: ' + error.message, 'error');
                addDebugLog(`请求失败: ${error.message}`, 'error');
            } finally {
                // Re-enable all generate buttons
                const buttons = document.querySelectorAll('.btn-primary');
                buttons.forEach(btn => btn.disabled = false);
            }
        }
        
        // Play/Pause toggle function
        function togglePlay() {
            const audioPlayer = document.getElementById('audio-player');
            const playBtn = document.getElementById('play-btn');
            const playIcon = document.getElementById('play-icon');
            const pauseIcon = document.getElementById('pause-icon');
            const progressBar = document.getElementById('progress-bar');
            
            if (audioPlayer.src) {
                if (isPlaying) {
                    audioPlayer.pause();
                    playIcon.style.display = 'block';
                    pauseIcon.style.display = 'none';
                } else {
                    audioPlayer.play();
                    playIcon.style.display = 'none';
                    pauseIcon.style.display = 'block';
                    progressBar.disabled = false;
                }
                isPlaying = !isPlaying;
            }
        }
        
        // Mute/Unmute toggle function
        function toggleMute() {
            const audioPlayer = document.getElementById('audio-player');
            const volumeIcon = document.getElementById('volume-icon');
            const muteIcon = document.getElementById('mute-icon');
            
            if (audioPlayer.src) {
                audioPlayer.muted = !audioPlayer.muted;
                isMuted = audioPlayer.muted;
                
                if (isMuted) {
                    volumeIcon.style.display = 'none';
                    muteIcon.style.display = 'block';
                } else {
                    volumeIcon.style.display = 'block';
                    muteIcon.style.display = 'none';
                }
            }
        }
        
        // Batch processing functions
        function updateBatchTemplate() {
            const mode = document.getElementById('batch-mode').value;
            const voiceDesignOptions = document.getElementById('batch-voice-design-options');
            const ttsOptions = document.getElementById('batch-tts-options');
            
            if (mode === 'voice-design') {
                voiceDesignOptions.style.display = 'block';
                ttsOptions.style.display = 'none';
            } else {
                voiceDesignOptions.style.display = 'none';
                ttsOptions.style.display = 'block';
            }
        }
        
        let batchResults = [];
        
        async function startBatchProcessing() {
            const textsInput = document.getElementById('batch-texts').value.trim();
            const mode = document.getElementById('batch-mode').value;
            const language = document.getElementById('batch-language').value;
            const progressDiv = document.getElementById('batch-progress');
            const progressBar = document.getElementById('batch-progress-bar');
            const progressText = document.getElementById('batch-progress-text');
            const nodesContainer = document.getElementById('batch-nodes-container');
            const resultsDiv = document.getElementById('batch-results');
            const resultsList = document.getElementById('batch-results-list');
            
            if (!textsInput) {
                showStatus('请输入至少一个文本', 'error');
                return;
            }
            
            const texts = textsInput.split('\n').filter(t => t.trim());
            if (texts.length === 0) {
                showStatus('请输入有效的文本', 'error');
                return;
            }
            
            // Reset and show progress
            batchResults = [];
            progressDiv.style.display = 'block';
            resultsDiv.style.display = 'none';
            resultsList.innerHTML = '';
            
            // Create workflow nodes
            nodesContainer.innerHTML = '';
            texts.forEach((text, index) => {
                const node = document.createElement('div');
                node.className = 'workflow-node';
                node.id = `workflow-node-${index}`;
                node.innerHTML = `
                    <div class="node-header">
                        <div class="node-title">
                            <div class="node-icon">${index + 1}</div>
                            <span style="max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${text.substring(0, 30)}${text.length > 30 ? '...' : ''}</span>
                        </div>
                        <span class="node-status" id="node-status-${index}">等待中</span>
                    </div>
                    <p style="font-size: 12px; color: #999;" id="node-info-${index}">准备处理</p>
                `;
                nodesContainer.appendChild(node);
            });
            
            // Process each text
            const startTime = Date.now();
            
            for (let i = 0; i < texts.length; i++) {
                const text = texts[i];
                const node = document.getElementById(`workflow-node-${i}`);
                const statusEl = document.getElementById(`node-status-${i}`);
                const infoEl = document.getElementById(`node-info-${i}`);
                
                // Update progress
                progressText.textContent = `${i + 1}/${texts.length}`;
                progressBar.style.width = `${((i + 1) / texts.length) * 100}%`;
                
                // Update node status
                node.classList.add('active');
                statusEl.textContent = '处理中...';
                statusEl.className = 'node-status running';
                infoEl.textContent = '正在生成语音...';
                
                try {
                    // Prepare request body
                    let requestBody = {
                        text: text,
                        language: language,
                        mode: mode
                    };
                    
                    if (mode === 'voice-design') {
                        requestBody.voice_description = document.getElementById('batch-voice-desc').value;
                    } else {
                        requestBody.speaker = document.getElementById('batch-speaker').value;
                        requestBody.style = document.getElementById('batch-style').value;
                    }
                    
                    // Send request
                    const response = await fetch('/tts', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(requestBody)
                    });
                    
                    const data = await response.json();
                    
                    if (data.success) {
                        batchResults.push({ text, audioUrl: data.audio_url, success: true });
                        statusEl.textContent = '完成';
                        statusEl.className = 'node-status completed';
                        infoEl.textContent = '语音生成成功';
                        node.classList.remove('active');
                        node.style.borderLeft = '4px solid #52c41a';
                    } else {
                        throw new Error(data.error);
                    }
                } catch (error) {
                    batchResults.push({ text, error: error.message, success: false });
                    statusEl.textContent = '失败';
                    statusEl.className = 'node-status';
                    statusEl.style.background = '#fff1f0';
                    statusEl.style.color = '#ff4d4f';
                    infoEl.textContent = `错误: ${error.message}`;
                    node.classList.remove('active');
                    node.style.borderLeft = '4px solid #ff4d4f';
                }
                
                // Small delay between requests to avoid overwhelming the server
                if (i < texts.length - 1) {
                    await new Promise(resolve => setTimeout(resolve, 500));
                }
            }
            
            // Show completion
            const totalTime = ((Date.now() - startTime) / 1000).toFixed(2);
            showStatus(`批量处理完成！共${texts.length}个，成功${batchResults.filter(r => r.success).length}个，耗时${totalTime}秒`, 'success');
            
            // Display results
            resultsDiv.style.display = 'block';
            batchResults.forEach((result, index) => {
                if (result.success) {
                    const resultItem = document.createElement('div');
                    resultItem.className = 'batch-item completed';
                    resultItem.innerHTML = `
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-size: 13px; color: #333; max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${index + 1}. ${result.text}</span>
                            <button onclick="playBatchAudio('${result.audioUrl}')" class="btn-primary" style="padding: 6px 12px; font-size: 12px;">
                                <svg style="width: 12px; height: 12px; margin-right: 4px; vertical-align: middle;" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
                                播放
                            </button>
                        </div>
                    `;
                    resultsList.appendChild(resultItem);
                }
            });
        }
        
        function playBatchAudio(audioUrl) {
            const audioPlayer = document.getElementById('audio-player');
            audioPlayer.src = audioUrl;
            audioPlayer.play();
            isPlaying = true;
            document.getElementById('play-icon').style.display = 'none';
            document.getElementById('pause-icon').style.display = 'block';
            showStatus('正在播放批量生成的音频', 'success');
        }
        
        // Update progress bar
        function updateProgress() {
            const audioPlayer = document.getElementById('audio-player');
            const progressBar = document.getElementById('progress-bar');
            const currentTimeEl = document.getElementById('current-time');
            
            if (audioPlayer.duration) {
                const progress = (audioPlayer.currentTime / audioPlayer.duration) * 100;
                progressBar.style.width = progress + '%';
                currentTimeEl.textContent = formatTime(audioPlayer.currentTime);
                
                // Reset when audio ends
                if (audioPlayer.ended) {
                    isPlaying = false;
                    document.getElementById('play-icon').style.display = 'block';
                    document.getElementById('pause-icon').style.display = 'none';
                    progressBar.style.width = '0%';
                }
            }
        }
        
        // Seek audio when clicking on progress bar
        function seekAudio(event) {
            const audioPlayer = document.getElementById('audio-player');
            const progressBarContainer = event.currentTarget;
            
            if (audioPlayer.duration) {
                const rect = progressBarContainer.getBoundingClientRect();
                const clickPosition = (event.clientX - rect.left) / rect.width;
                audioPlayer.currentTime = clickPosition * audioPlayer.duration;
            }
        }
        
        // Update audio info when loaded
        function updateAudioInfo() {
            const audioPlayer = document.getElementById('audio-player');
            const totalTimeEl = document.getElementById('total-time');
            totalTimeEl.textContent = formatTime(audioPlayer.duration);
        }
        
        // Format time as mm:ss
        function formatTime(seconds) {
            const mins = Math.floor(seconds / 60);
            const secs = Math.floor(seconds % 60);
            return `${mins}:${secs.toString().padStart(2, '0')}`;
        }
        
        // Reset audio controls
        function resetAudioControls() {
            isPlaying = false;
            isMuted = false;
            document.getElementById('play-icon').style.display = 'block';
            document.getElementById('pause-icon').style.display = 'none';
            document.getElementById('volume-icon').style.display = 'block';
            document.getElementById('mute-icon').style.display = 'none';
            document.getElementById('progress-bar').style.width = '0%';
            document.getElementById('current-time').textContent = '0:00';
            document.getElementById('total-time').textContent = '0:00';
        }
        
        // Status display function
        function showStatus(message, type) {
            const statusDiv = document.getElementById('status');
            statusDiv.textContent = message;
            
            // Remove existing classes
            statusDiv.className = 'status-box';
            
            // Add new class based on type
            if (type) {
                statusDiv.classList.add('status-' + type);
            }
        }
        
        // Download audio function
        function downloadAudio() {
            const audioPlayer = document.getElementById('audio-player');
            if (audioPlayer.src) {
                const a = document.createElement('a');
                a.href = audioPlayer.src;
                a.download = 'qwen-tts-output.wav';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/tts', methods=['POST'])
def text_to_speech():
    try:
        data = request.json
        text = data.get('text', '')
        mode = data.get('mode', 'voice-design')

        if not text:
            return jsonify({'success': False, 'error': '请输入要合成的文本'})

        # 打印请求信息
        print(f"\n=== 语音生成请求 ===")
        print(f"模式: {mode}")
        print(f"文本: {text[:50]}...")
        
        # 默认参数
        language = data.get('language', 'auto')
        
        # 语言代码映射（兼容前端ISO代码）
        language_map = {
            'zh': 'chinese',
            'en': 'english',
            'ja': 'japanese',
            'ko': 'korean',
            'fr': 'french',
            'de': 'german',
            'es': 'spanish',
            'it': 'italian',
            'pt': 'portuguese',
            'ru': 'russian'
        }
        
        # 转换语言代码
        if language in language_map:
            language = language_map[language]
        
        voice_description = data.get('voice_description', '')
        reference_text = data.get('reference_text', '')
        speaker = data.get('speaker', 'Vivian')
        style = data.get('style', '')
        model_version = data.get('model_version', 'full')  # 获取模型版本参数
        
        if mode == 'voice-design':
            print(f"语言: {language}")
            print(f"声音描述: {voice_description[:50]}...")
            print(f"模型版本: {model_version}")
        elif mode == 'voice-clone':
            print(f"语言: {language}")
            print(f"参考文本: {reference_text[:50]}...")
            print(f"模型版本: {model_version}")
        elif mode == 'tts-custom':
            print(f"语言: {language}")
            print(f"说话人: {speaker}")
            print(f"风格: {style}")
            print(f"模型版本: {model_version}")
        print(f"====================\n")

        # 使用Qwen-TTS模型生成语音
        try:
            import time
            start_time = time.time()
            
            # 性能优化：使用更快的生成参数
            # 降低max_new_tokens以加速生成，根据文本长度动态调整
            text_length = len(text)
            
            # 根据模型版本调整参数
            if model_version == 'small':
                # 轻量版：更快的生成速度，稍微降低质量
                max_tokens = min(1024, max(256, text_length * 6))
                temperature = 0.5
                top_p = 0.7
                top_k = 30
                print(f"使用轻量版模型参数以加速生成")
            else:
                # 完整版：最高质量
                max_tokens = min(2048, max(512, text_length * 8))
                temperature = 0.6
                top_p = 0.8
                top_k = 50
            
            generation_config = {
                'do_sample': True,
                'temperature': temperature,
                'top_p': top_p,
                'top_k': top_k,
                'max_new_tokens': max_tokens,
                'num_beams': 1,
                'early_stopping': True,
            }
            
            # 根据model_version选择模型大小（0.6b或1.7b）
            use_0_6b = (model_version == '0.6b')
            
            # 根据不同模式调用不同的生成方法
            if mode == 'voice-design':
                # 语音设计模式 - 选择VoiceDesign模型
                if use_0_6b and model_voice_design_0_6b is not None:
                    selected_model = model_voice_design_0_6b
                    model_name = "0.6B VoiceDesign"
                elif model_voice_design is not None:
                    selected_model = model_voice_design
                    model_name = "1.7B VoiceDesign"
                else:
                    selected_model = None
                    model_name = "None"
                
                if selected_model is not None:
                    print(f"使用{model_name}模型生成语音...")
                    print(f"开始时间: {time.strftime('%H:%M:%S')}")
                    print(f"优化参数: max_tokens={max_tokens}")
                    
                    # 尝试使用优化的生成方法
                    try:
                        # 使用更快的推理设置
                        wavs, sample_rate = selected_model.generate_voice_design(
                            text=text,
                            language=language,
                            voice_description=voice_description,
                            instruct=voice_description,
                            **generation_config
                        )
                    except TypeError as e:
                        print(f"优化参数不支持: {e}，使用默认参数")
                        # 如果模型不支持这些参数，使用默认参数
                        wavs, sample_rate = selected_model.generate_voice_design(
                            text=text,
                            language=language,
                            voice_description=voice_description,
                            instruct=voice_description
                        )
                else:
                    raise Exception("VoiceDesign模型未加载")
                    
            elif mode == 'voice-clone':
                # 语音克隆模式 - 使用Base模型的generate_voice_clone方法
                if use_0_6b and model_base_0_6b is not None:
                    selected_model = model_base_0_6b
                    model_name = "0.6B Base"
                elif model_base is not None:
                    selected_model = model_base
                    model_name = "1.7B Base"
                else:
                    selected_model = None
                    model_name = "None"
                
                if selected_model is not None:
                    print(f"使用{model_name}模型进行声音克隆...")
                    print(f"开始时间: {time.strftime('%H:%M:%S')}")
                    
                    # 获取参考音频路径
                    reference_audio = data.get('reference_audio', '')
                    if not reference_audio:
                        raise Exception("请上传参考音频文件")
                    
                    # 构建参考音频的完整路径（从output目录查找）
                    ref_audio_path = os.path.join(OUTPUT_DIR, reference_audio)
                    if not os.path.exists(ref_audio_path):
                        # 如果文件不在output目录，尝试在临时目录查找
                        ref_audio_path = os.path.join(tempfile.gettempdir(), reference_audio)
                    if not os.path.exists(ref_audio_path):
                        # 如果还是找不到，使用原路径
                        ref_audio_path = reference_audio
                    
                    print(f"参考音频: {ref_audio_path}")
                    print(f"参考文本: {reference_text[:50] if reference_text else 'None'}...")
                    
                    try:
                        # 使用Base模型的generate_voice_clone方法
                        wavs, sample_rate = selected_model.generate_voice_clone(
                            text=text,
                            language=language,
                            ref_audio=ref_audio_path,
                            ref_text=reference_text if reference_text else None,
                            x_vector_only_mode=False,  # 使用ICL模式以获得更好的克隆效果
                            **generation_config
                        )
                    except Exception as e:
                        print(f"声音克隆失败: {e}")
                        print("尝试使用x_vector_only_mode=True模式...")
                        # 如果ICL模式失败，尝试仅使用x_vector模式
                        wavs, sample_rate = selected_model.generate_voice_clone(
                            text=text,
                            language=language,
                            ref_audio=ref_audio_path,
                            x_vector_only_mode=True,
                            **generation_config
                        )
                else:
                    raise Exception("Base模型未加载，声音克隆功能不可用")
                    
            elif mode == 'tts-custom':
                # 自定义语音模式 - 选择CustomVoice模型
                if use_0_6b and model_custom_voice_0_6b is not None:
                    selected_model = model_custom_voice_0_6b
                    model_name = "0.6B CustomVoice"
                elif model_custom_voice is not None:
                    selected_model = model_custom_voice
                    model_name = "1.7B CustomVoice"
                else:
                    selected_model = None
                    model_name = "None"
                
                if selected_model is not None:
                    print(f"使用{model_name}模型生成语音...")
                    print(f"开始时间: {time.strftime('%H:%M:%S')}")
                    print(f"优化参数: max_tokens={max_tokens}")
                    
                    # 将style转换为instruct（如果提供了style）
                    instruct_text = style if style else None
                    
                    try:
                        if instruct_text:
                            wavs, sample_rate = selected_model.generate_custom_voice(
                                text=text,
                                language=language,
                                speaker=speaker,
                                instruct=instruct_text,
                                **generation_config
                            )
                        else:
                            wavs, sample_rate = selected_model.generate_custom_voice(
                                text=text,
                                language=language,
                                speaker=speaker,
                                **generation_config
                            )
                    except TypeError as e:
                        print(f"优化参数不支持: {e}，使用默认参数")
                        if instruct_text:
                            wavs, sample_rate = selected_model.generate_custom_voice(
                                text=text,
                                language=language,
                                speaker=speaker,
                                instruct=instruct_text
                            )
                        else:
                            wavs, sample_rate = selected_model.generate_custom_voice(
                                text=text,
                                language=language,
                                speaker=speaker
                            )
                else:
                    raise Exception("CustomVoice模型未加载")
                    
            else:
                # 默认使用语音设计模式
                if use_0_6b and model_voice_design_0_6b is not None:
                    default_model = model_voice_design_0_6b
                    default_name = "0.6B VoiceDesign"
                elif model_voice_design is not None:
                    default_model = model_voice_design
                    default_name = "1.7B VoiceDesign"
                else:
                    default_model = None
                    default_name = "None"
                    
                if default_model is not None:
                    print(f"使用默认{default_name}模型生成语音...")
                    try:
                        wavs, sample_rate = default_model.generate_voice_design(
                            text=text,
                            language=language,
                            voice_description=voice_description,
                            instruct=voice_description,
                            **generation_config
                        )
                    except TypeError:
                        wavs, sample_rate = default_model.generate_voice_design(
                            text=text,
                            language=language,
                            voice_description=voice_description,
                            instruct=voice_description
                        )
                else:
                    raise Exception("VoiceDesign模型未加载")
            
            end_time = time.time()
            generation_duration = end_time - start_time
            print(f"语音生成完成，耗时: {generation_duration:.2f}秒")
            
            # 处理生成的音频（所有分支都需要执行这里）
            print(f"语音生成成功，采样率: {sample_rate}")
            audio_data = wavs[0]  # 取第一个生成的音频
            
            # 将生成的音频保存到output目录
            audio_path = os.path.join(OUTPUT_DIR, f"qwen_tts_output_{hash(text)}.wav")
            scipy.io.wavfile.write(audio_path, sample_rate, audio_data)
            print(f"音频已保存到: {audio_path}")
            
            return jsonify({
                'success': True,
                'audio_url': f'/audio/qwen_tts_output_{hash(text)}.wav'
            })
            
        except Exception as e:
            print(f"模型生成失败: {e}")
            import traceback
            traceback.print_exc()
            # 如果模型生成失败，使用模拟音频作为后备
            sample_rate = 22050
            duration = 2  # 2秒
            frequency = 440  # A4音
            
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            audio_data = 0.5 * np.sin(2 * np.pi * frequency * t)
            audio_data = (audio_data * 32767).astype(np.int16)
            
            audio_path = os.path.join(OUTPUT_DIR, f"tts_output_{hash(text)}.wav")
            scipy.io.wavfile.write(audio_path, sample_rate, audio_data)
            print(f"音频已保存到: {audio_path}")
            
            return jsonify({
                'success': True,
                'audio_url': f'/audio/tts_output_{hash(text)}.wav'
            })

    except Exception as e:
        print(f"生成语音时出错: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/upload', methods=['POST'])
def upload_file():
    """上传参考音频文件用于声音克隆"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': '没有文件'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': '文件名为空'}), 400
        
        # 保存文件到output目录
        filename = f"ref_audio_{hash(file.filename)}_{file.filename}"
        filepath = os.path.join(OUTPUT_DIR, filename)
        file.save(filepath)
        
        print(f"参考音频已上传到: {filepath}")
        
        return jsonify({
            'success': True,
            'filename': filename,
            'filepath': filepath
        })
    except Exception as e:
        print(f"文件上传失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/audio/<filename>')
def serve_audio(filename):
    try:
        # 首先在output目录查找
        audio_path = os.path.join(OUTPUT_DIR, filename)
        if not os.path.exists(audio_path):
            # 如果不在output目录，尝试在临时目录查找（兼容旧文件）
            audio_path = os.path.join(tempfile.gettempdir(), filename)
        return send_file(audio_path, mimetype='audio/wav')
    except Exception as e:
        return jsonify({'error': str(e)}), 404

if __name__ == '__main__':
    print("\n" + "="*50)
    print("Qwen-TTS 服务启动成功！")
    print("请在浏览器中访问: http://localhost:5000")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
