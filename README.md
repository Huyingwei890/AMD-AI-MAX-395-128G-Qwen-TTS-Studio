# AIMAX395TTS - AMD AI MAX 395 本地语音合成平台

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue.svg" alt="Python 3.12">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-orange.svg" alt="PyTorch 2.0+">
  <img src="https://img.shields.io/badge/Flask-3.0+-green.svg" alt="Flask 3.0+">
  <img src="https://img.shields.io/badge/Qwen3--TTS-12Hz-purple.svg" alt="Qwen3-TTS">
  <img src="https://img.shields.io/badge/AMD-AI_MAX_395-red.svg" alt="AMD AI MAX 395">
</p>

<p align="center">
  <b>专为 AMD AI MAX 395 + 128GB 大内存平台优化的本地语音合成解决方案</b>
</p>

---

## 🎯 项目简介

AIMAX395TTS 是一个专为 **AMD AI MAX 395** 处理器 + **128GB 大内存** 平台深度优化的本地语音合成系统。基于阿里巴巴开源的 **Qwen3-TTS** 大模型，实现了高质量的文本转语音(TTS)、声音克隆和语音设计功能。

### ✨ 核心特性

- 🚀 **本地部署** - 完全离线运行，保护隐私，无需联网
- 🎭 **声音克隆** - 仅需3-10秒音频样本，即可克隆任意声音
- 🎨 **语音设计** - 通过文字描述创造独特的虚拟声音
- 🔊 **多语言支持** - 支持中文、英文、日文、韩文等10+种语言
- ⚡ **高性能优化** - 专为AMD AI MAX 395平台深度优化
- 💻 **大内存利用** - 充分利用128GB内存，支持大模型流畅运行
- 🖥️ **Web界面** - 现代化Web UI，操作简便直观

---
main page/声音设计
<img width="2068" height="1412" alt="main" src="https://github.com/user-attachments/assets/697cc5cf-5d60-4739-a30b-0c0061d8db85" />
声音克隆
<img width="2065" height="1469" alt="2" src="https://github.com/user-attachments/assets/f7082e4c-2f31-4028-a285-5f9a5812ac27" />
预设声音
<img width="2057" height="1356" alt="3" src="https://github.com/user-attachments/assets/027a7961-cb96-4dd7-8313-429476598cde" />



## 🖥️ 硬件平台

### 推荐配置

| 组件 | 规格 | 说明 |
|------|------|------|
| **处理器** | AMD AI MAX 395 | 专为AI优化的APU，集成NPU |
| **内存** | 128GB DDR5 | 大内存支持大模型加载 |
| **存储** | 500GB+ SSD | 模型文件较大，建议SSD |
| **操作系统** | Windows 11 / Linux | 64位系统 |

### 为什么选择 AMD AI MAX 395？

- **集成NPU** - 内置AI加速单元，AI推理性能强劲
- **大内存支持** - 支持128GB+内存，可加载更大模型
- **高能效比** - 低功耗高性能，适合长时间运行
- **本地部署** - 无需云端，数据安全有保障

---

## 📦 模型支持

本项目支持以下Qwen3-TTS模型：

### 1.7B 完整版模型（高质量）

| 模型名称 | 大小 | 用途 | 显存需求 |
|---------|------|------|---------|
| Qwen3-TTS-1.7B-Base | ~3.5GB | 语音克隆基础模型 | ~6GB |
| Qwen3-TTS-1.7B-VoiceDesign | ~3.8GB | 语音设计 | ~7GB |
| Qwen3-TTS-1.7B-CustomVoice | ~3.8GB | 自定义声音 | ~7GB |

### 0.6B 轻量版模型（快速）

| 模型名称 | 大小 | 用途 | 显存需求 |
|---------|------|------|---------|
| Qwen3-TTS-0.6B-Base | ~1.2GB | 语音克隆基础模型 | ~3GB |
| Qwen3-TTS-0.6B-VoiceDesign | ~1.3GB | 语音设计 | ~3GB |
| Qwen3-TTS-0.6B-CustomVoice | ~1.3GB | 自定义声音 | ~3GB |

> **注意**：模型文件需要从HuggingFace或ModelScope下载，不包含在本项目中。

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/yourusername/AIMAX395TTS.git
cd AIMAX395TTS

# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 安装依赖
pip install torch transformers flask scipy numpy
pip install qwen-tts  # Qwen3-TTS官方库
```

### 2. 下载模型

```bash
# 使用提供的下载脚本
python download_models.py

# 或者手动从HuggingFace下载
# https://huggingface.co/Qwen
```

### 3. 启动服务

**方法一：一键启动（推荐）**

双击运行 `一键启动.bat`，选择启动模式：
- **高性能优化版** - 更快的生成速度
- **标准版本** - 原始稳定版本

**方法二：命令行启动**

```bash
# 高性能优化版
python app_optimized.py

# 标准版本
python app.py
```

### 4. 访问Web界面

打开浏览器访问：http://localhost:5000

---

## 🎨 功能模块

### 1. 声音设计 (Voice Design)

通过文字描述创造独特的虚拟声音。

**使用示例**：
```
描述：温柔的女声，语速适中，带有关怀的语气
文本：你好，很高兴为你服务。
```

### 2. 声音克隆 (Voice Clone)

上传3-10秒的音频样本，克隆任意声音。

**支持格式**：WAV, MP3, OGG
**推荐时长**：5-10秒
**推荐质量**：44.1kHz, 16bit

### 3. TTS预设 (TTS Custom)

使用预设的说话人声音进行文本转语音。

**内置说话人**：
- Vivian (女声)
- Ava (女声)
- Emma (女声)
- Brian (男声)
- 等20+种声音

---

## ⚡ 性能优化

本项目针对AMD AI MAX 395 + 128GB内存平台进行了深度优化：

### 优化技术

| 优化项 | 说明 | 效果 |
|--------|------|------|
| **参数缓存** | LRU缓存生成参数 | 减少重复计算 |
| **模型量化** | INT8动态量化 | 内存占用降低50% |
| **torch.compile** | PyTorch 2.0编译加速 | 推理速度提升20-30% |
| **torch.no_grad** | 禁用梯度计算 | 减少内存开销 |
| **批处理** | 批量推理 | 提高吞吐量 |

### 性能对比

| 模型版本 | 生成速度 | 内存占用 | 质量 |
|---------|---------|---------|------|
| 1.7B-Full | 较慢 | ~20GB | ⭐⭐⭐⭐⭐ |
| 0.6B | 快3倍 | ~8GB | ⭐⭐⭐⭐ |
| 极速模式 | 最快 | ~6GB | ⭐⭐⭐ |

### 在AMD AI MAX 395上的表现

- **1.7B模型**：生成10秒语音约需 8-12秒
- **0.6B模型**：生成10秒语音约需 3-5秒
- **极速模式**：生成10秒语音约需 2-3秒
- **并发支持**：可同时加载6个模型，支持多用户

---

## 📁 项目结构

```
AIMAX395TTS/
├── app.py                    # 标准版本主程序
├── app_optimized.py          # 高性能优化版本
├── 一键启动.bat              # Windows一键启动脚本
├── templates/
│   └── index.html           # Web界面模板
├── README.md                # 项目说明文档
├── requirements.txt         # Python依赖列表
├── download_models.py       # 模型下载脚本
└── output/                  # 生成的音频输出目录
```

---

## 🛠️ 高级配置

### 模型参数调整

在Web界面的右侧设置面板中，可以调整以下参数：

| 参数 | 范围 | 默认值 | 说明 |
|------|------|--------|------|
| Temperature | 0.0-1.0 | 0.6 | 控制创造性，越高越随机 |
| Top P | 0.0-1.0 | 0.85 | 控制多样性，越高选择越多 |
| 采样率 | 24kHz/48kHz | 24kHz | 音频质量 |
| 输出格式 | WAV/MP3/OGG | WAV | 音频格式 |

### 环境变量

```bash
# 设置模型路径
export QWEN_TTS_MODEL_PATH="/path/to/models"

# 设置输出目录
export QWEN_TTS_OUTPUT_DIR="/path/to/output"

# 设置调试模式
export FLASK_DEBUG=0
```

---

## 🔧 故障排除

### 常见问题

**1. 模型加载失败**
- 检查模型文件是否完整下载
- 确认模型路径正确
- 检查内存是否充足

**2. 生成速度慢**
- 尝试使用0.6B轻量版模型
- 开启极速模式
- 关闭其他占用内存的程序

**3. 音频质量不佳**
- 使用1.7B完整版模型
- 调整Temperature参数
- 提供更清晰的参考音频（声音克隆）

**4. 内存不足**
- 使用0.6B轻量版模型
- 减少同时加载的模型数量
- 增加虚拟内存

---

## 📊 系统要求

### 最低配置
- CPU: 8核以上
- 内存: 32GB
- 存储: 100GB SSD
- Python: 3.10+

### 推荐配置（本项目优化目标）
- **CPU**: AMD AI MAX 395
- **内存**: 128GB DDR5
- **存储**: 500GB NVMe SSD
- **Python**: 3.12

---

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

### 开发流程

1. Fork本项目
2. 创建特性分支：`git checkout -b feature/AmazingFeature`
3. 提交更改：`git commit -m 'Add some AmazingFeature'`
4. 推送分支：`git push origin feature/AmazingFeature`
5. 创建Pull Request

---

## 📄 许可证

本项目采用 [MIT License](LICENSE) 开源许可证。

Qwen3-TTS模型遵循其原始许可证，详情请参考：
- https://huggingface.co/Qwen
- https://github.com/QwenLM/Qwen

---

## 🙏 致谢

- [阿里巴巴通义千问团队](https://github.com/QwenLM) - 提供Qwen3-TTS模型
- [HuggingFace](https://huggingface.co/) - 模型托管平台
- [PyTorch团队](https://pytorch.org/) - 深度学习框架
- [AMD](https://www.amd.com/) - AI MAX 395硬件平台

---

## 📞 联系方式

- 项目主页：ttps://github.com/Huyingwei890/AMD-AI-MAX-395-128G-Qwen-TTS-Studio
- Issue反馈：https://github.com/Huyingwei890/AMD-AI-MAX-395-128G-Qwen-TTS-Studio/issue
- 邮箱：huyingwei@live.cn

---

<p align="center">
  <b>Made with ❤️ for AMD AI MAX 395 Platform</b>
</p>

