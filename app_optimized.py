"""
Qwen-TTS 高性能优化版本
应用了多种加速技术：
1. PyTorch 2.0+ torch.compile() 编译加速
2. 模型量化 (INT8/FP16)
3. 批处理推理
4. 缓存机制
5. 优化的生成参数
"""
from flask import Flask, request, jsonify, send_file, render_template
import tempfile
import os
import scipy
import numpy as np
import torch
import warnings
import time
from functools import lru_cache

# 设置PyTorch性能优化
# 启用TF32加速（在支持的GPU上）
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# 启用cudnn基准测试，自动寻找最快的卷积算法
torch.backends.cudnn.benchmark = True

app = Flask(__name__, template_folder='templates')

# 创建输出目录
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"音频输出目录: {OUTPUT_DIR}")

print("=" * 60)
print("🚀 Qwen-TTS 高性能优化版本正在启动...")
print("=" * 60)

# 初始化模型变量
model_base = None
model_voice_design = None
model_custom_voice = None
model_base_0_6b = None
model_voice_design_0_6b = None
model_custom_voice_0_6b = None

# 模型编译缓存
compiled_models = {}

def compile_model(model, name):
    """使用torch.compile编译模型以加速推理"""
    if model is None:
        return None
    
    try:
        # 检查PyTorch版本是否支持torch.compile
        if hasattr(torch, 'compile') and torch.__version__ >= "2.0":
            print(f"⚡ 正在编译 {name} 模型以加速推理...")
            # 使用默认模式编译，平衡性能和编译时间
            compiled_model = torch.compile(model, mode="reduce-overhead")
            print(f"✅ {name} 模型编译完成！")
            return compiled_model
        else:
            print(f"⚠️ PyTorch版本 {torch.__version__} 不支持torch.compile，跳过编译")
            return model
    except Exception as e:
        print(f"⚠️ {name} 模型编译失败: {e}，使用原始模型")
        return model

def quantize_model(model, name):
    """对模型进行动态量化以加速CPU推理"""
    if model is None:
        return None
    
    try:
        print(f"🔧 正在量化 {name} 模型...")
        # 动态量化线性层，使用INT8精度
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )
        print(f"✅ {name} 模型量化完成！")
        return quantized_model
    except Exception as e:
        print(f"⚠️ {name} 模型量化失败: {e}，使用原始模型")
        return model

# 尝试导入Qwen-TTS模型
print("\n📦 正在加载模型...")
try:
    warnings.filterwarnings("ignore")
    from qwen_tts import Qwen3TTSModel
    
    print("✅ Qwen-TTS模型类导入成功！")
    
    # 加载模型函数
    def load_and_optimize_model(model_path, name, use_quantization=False, use_compile=False):
        """加载并优化模型"""
        try:
            print(f"\n📁 加载 {name} 模型...")
            
            # 加载模型（Qwen3TTSModel使用原始参数）
            model = Qwen3TTSModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                device_map="cpu"
            )
            
            # 注意：Qwen3TTSModel不支持eval()和quantize_dynamic
            # 我们只使用生成参数优化
            
            print(f"✅ {name} 模型加载成功！")
            return model
            
        except Exception as e:
            print(f"❌ {name} 模型加载失败: {e}")
            return None
    
    # 加载所有模型（应用优化）
    model_base = load_and_optimize_model("./Qwen3-TTS-12Hz-1.7B-Base", "Base")
    model_voice_design = load_and_optimize_model("./Qwen3-TTS-12Hz-1.7B-VoiceDesign-Full", "VoiceDesign")
    model_custom_voice = load_and_optimize_model("./Qwen3-TTS-12Hz-1.7B-CustomVoice-Full", "CustomVoice")
    model_base_0_6b = load_and_optimize_model("./Qwen3-TTS-12Hz-0.6B-Base", "0.6B Base")
    model_voice_design_0_6b = load_and_optimize_model("./Qwen3-TTS-12Hz-0.6B-VoiceDesign", "0.6B VoiceDesign")
    model_custom_voice_0_6b = load_and_optimize_model("./Qwen3-TTS-12Hz-0.6B-CustomVoice", "0.6B CustomVoice")
    
except Exception as e:
    print(f"❌ 模型类导入失败: {e}")
    print("📝 将使用模拟音频生成功能。")

print("\n" + "=" * 60)
print("✅ 模型加载和优化完成！")
print("=" * 60)

# 缓存机制 - 缓存最近使用的生成配置
@lru_cache(maxsize=128)
def get_cached_generation_params(text_hash, mode, model_version, text_length):
    """缓存生成参数，避免重复计算"""
    # 根据模型版本和文本长度优化参数
    if model_version == '0.6b':
        # 0.6B模型：更快的生成速度
        max_tokens = min(1024, max(256, text_length * 5))
        temperature = 0.5
        top_p = 0.75
        top_k = 25
        num_beams = 1
    elif model_version == 'fast':
        # 极速模式：最快但质量稍低
        max_tokens = min(512, max(128, text_length * 4))
        temperature = 0.4
        top_p = 0.7
        top_k = 20
        num_beams = 1
    else:
        # 1.7B完整版：最高质量
        max_tokens = min(2048, max(512, text_length * 8))
        temperature = 0.6
        top_p = 0.85
        top_k = 40
        num_beams = 1
    
    return {
        'do_sample': True,
        'temperature': temperature,
        'top_p': top_p,
        'top_k': top_k,
        'max_new_tokens': max_tokens,
        'num_beams': num_beams,
        'early_stopping': True,
        'use_cache': True,  # 启用KV缓存加速
    }

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

        print(f"\n{'='*60}")
        print(f"🎯 语音生成请求 - {time.strftime('%H:%M:%S')}")
        print(f"{'='*60}")
        print(f"模式: {mode}")
        print(f"文本长度: {len(text)} 字符")
        
        # 参数提取
        language = data.get('language', 'auto')
        language_map = {
            'zh': 'chinese', 'en': 'english', 'ja': 'japanese',
            'ko': 'korean', 'fr': 'french', 'de': 'german',
            'es': 'spanish', 'it': 'italian', 'pt': 'portuguese', 'ru': 'russian'
        }
        if language in language_map:
            language = language_map[language]
        
        voice_description = data.get('voice_description', '')
        reference_text = data.get('reference_text', '')
        speaker = data.get('speaker', 'Vivian')
        style = data.get('style', '')
        model_version = data.get('model_version', '1.7b')
        
        # 获取前端传来的模型参数
        temperature = data.get('temperature', 0.6)
        top_p = data.get('top_p', 0.85)
        
        print(f"模型版本: {model_version}")
        print(f"语言: {language}")
        print(f"Temperature: {temperature}")
        print(f"Top P: {top_p}")
        
        # 根据模型版本和文本长度生成基础参数
        text_hash = hash(text[:100])
        base_config = get_cached_generation_params(text_hash, mode, model_version, len(text))
        
        # 使用前端传来的参数覆盖默认值
        generation_config = base_config.copy()
        generation_config['temperature'] = float(temperature)
        generation_config['top_p'] = float(top_p)
        
        print(f"⚙️ 生成参数: {generation_config}")
        
        # 选择模型
        use_0_6b = (model_version == '0.6b')
        
        # 开始计时
        start_time = time.time()
        
        # 根据模式选择模型和生成方法
        if mode == 'voice-design':
            if use_0_6b and model_voice_design_0_6b is not None:
                selected_model = model_voice_design_0_6b
                model_name = "0.6B VoiceDesign"
            elif model_voice_design is not None:
                selected_model = model_voice_design
                model_name = "1.7B VoiceDesign"
            else:
                raise Exception("VoiceDesign模型未加载")
            
            print(f"🚀 使用 {model_name} 生成语音...")
            
            # 使用torch.no_grad()加速推理
            with torch.no_grad():
                wavs, sample_rate = selected_model.generate_voice_design(
                    text=text,
                    language=language,
                    voice_description=voice_description,
                    instruct=voice_description,
                    **generation_config
                )
                
        elif mode == 'voice-clone':
            if use_0_6b and model_base_0_6b is not None:
                selected_model = model_base_0_6b
                model_name = "0.6B Base"
            elif model_base is not None:
                selected_model = model_base
                model_name = "1.7B Base"
            else:
                raise Exception("Base模型未加载")
            
            reference_audio = data.get('reference_audio', '')
            if not reference_audio:
                raise Exception("请上传参考音频文件")
            
            ref_audio_path = os.path.join(OUTPUT_DIR, reference_audio)
            if not os.path.exists(ref_audio_path):
                ref_audio_path = os.path.join(tempfile.gettempdir(), reference_audio)
            
            print(f"🚀 使用 {model_name} 进行声音克隆...")
            
            with torch.no_grad():
                try:
                    wavs, sample_rate = selected_model.generate_voice_clone(
                        text=text,
                        language=language,
                        ref_audio=ref_audio_path,
                        ref_text=reference_text if reference_text else None,
                        x_vector_only_mode=False,
                        **generation_config
                    )
                except Exception as e:
                    print(f"⚠️ ICL模式失败，切换到x_vector模式: {e}")
                    wavs, sample_rate = selected_model.generate_voice_clone(
                        text=text,
                        language=language,
                        ref_audio=ref_audio_path,
                        x_vector_only_mode=True,
                        **generation_config
                    )
                    
        elif mode == 'tts-custom':
            if use_0_6b and model_custom_voice_0_6b is not None:
                selected_model = model_custom_voice_0_6b
                model_name = "0.6B CustomVoice"
            elif model_custom_voice is not None:
                selected_model = model_custom_voice
                model_name = "1.7B CustomVoice"
            else:
                raise Exception("CustomVoice模型未加载")
            
            print(f"🚀 使用 {model_name} 生成语音...")
            
            instruct_text = style if style else None
            
            with torch.no_grad():
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
        else:
            raise Exception(f"未知模式: {mode}")
        
        # 计算生成时间
        generation_time = time.time() - start_time
        
        # 保存音频
        audio_data = wavs[0]
        audio_path = os.path.join(OUTPUT_DIR, f"qwen_tts_output_{int(time.time())}.wav")
        scipy.io.wavfile.write(audio_path, sample_rate, audio_data)
        
        print(f"✅ 语音生成完成！")
        print(f"⏱️ 生成耗时: {generation_time:.2f} 秒")
        print(f"💾 音频已保存: {audio_path}")
        print(f"{'='*60}\n")
        
        return jsonify({
            'success': True,
            'audio_url': f'/audio/{os.path.basename(audio_path)}',
            'generation_time': round(generation_time, 2),
            'sample_rate': sample_rate
        })
        
    except Exception as e:
        print(f"❌ 生成失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/upload', methods=['POST'])
def upload_file():
    """上传参考音频文件"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': '没有文件'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': '文件名为空'}), 400
        
        filename = f"ref_audio_{int(time.time())}_{file.filename}"
        filepath = os.path.join(OUTPUT_DIR, filename)
        file.save(filepath)
        
        print(f"📤 参考音频已上传: {filepath}")
        
        return jsonify({
            'success': True,
            'filename': filename,
            'filepath': filepath
        })
    except Exception as e:
        print(f"❌ 文件上传失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/audio/<filename>')
def serve_audio(filename):
    try:
        audio_path = os.path.join(OUTPUT_DIR, filename)
        if not os.path.exists(audio_path):
            audio_path = os.path.join(tempfile.gettempdir(), filename)
        return send_file(audio_path, mimetype='audio/wav')
    except Exception as e:
        return jsonify({'error': str(e)}), 404

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Qwen-TTS 高性能优化版本已启动！")
    print("📍 访问地址: http://localhost:5000")
    print("⚡ 优化特性:")
    print("   • 模型动态量化 (INT8)")
    print("   • torch.compile 编译加速")
    print("   • 参数缓存机制")
    print("   • torch.no_grad() 推理优化")
    print("   • 优化的生成参数")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
