@echo off
chcp 65001 >nul
title Qwen-TTS 一键启动器
color 0A

echo.
echo ============================================
echo    🚀 Qwen-TTS 语音合成系统 - 一键启动
echo ============================================
echo.

:: 检查是否以管理员身份运行
net session >nul 2>&1
if %errorLevel% == 0 (
    echo [✓] 已以管理员权限运行
) else (
    echo [!] 提示：建议以管理员身份运行以获得最佳性能
    echo.
)

:: 设置工作目录
cd /d "%~dp0"
echo [✓] 工作目录: %CD%
echo.

:: 检查虚拟环境
if exist ".venv\Scripts\activate.bat" (
    echo [✓] 发现虚拟环境，正在激活...
    call .venv\Scripts\activate.bat
    echo [✓] 虚拟环境已激活
) else (
    echo [!] 警告：未找到虚拟环境，将使用系统Python
)
echo.

:: 检查必要的目录
echo [*] 检查项目目录结构...
if not exist "output" (
    mkdir output
    echo [✓] 创建输出目录: output\
)
if not exist "templates" (
    echo [✗] 错误：缺少 templates 目录！
    pause
    exit /b 1
)
echo [✓] 目录结构检查完成
echo.

:: 检查模型文件
echo [*] 检查模型文件...
set MODEL_COUNT=0
if exist "Qwen3-TTS-12Hz-1.7B-Base" set /a MODEL_COUNT+=1
if exist "Qwen3-TTS-12Hz-1.7B-VoiceDesign-Full" set /a MODEL_COUNT+=1
if exist "Qwen3-TTS-12Hz-1.7B-CustomVoice-Full" set /a MODEL_COUNT+=1
if exist "Qwen3-TTS-12Hz-0.6B-Base" set /a MODEL_COUNT+=1
if exist "Qwen3-TTS-12Hz-0.6B-VoiceDesign" set /a MODEL_COUNT+=1
if exist "Qwen3-TTS-12Hz-0.6B-CustomVoice" set /a MODEL_COUNT+=1

echo [✓] 发现 %MODEL_COUNT% 个模型文件夹
echo.

:: 选择启动模式
echo ============================================
echo    请选择启动模式：
echo ============================================
echo.
echo   [1] 🚀 高性能优化版（推荐）
echo       - 更快的生成速度
echo       - 参数缓存优化
echo       - 极速模式支持
echo.
echo   [2] 📦 标准版本
echo       - 原始稳定版本
echo       - 兼容性更好
echo.
echo ============================================
echo.

set /p choice="请输入选项 (1 或 2): "

if "%choice%"=="1" (
    echo.
    echo [*] 正在启动高性能优化版...
    echo [*] 启动时间约 30-60 秒，请耐心等待...
    echo.
    python app_optimized.py
) else if "%choice%"=="2" (
    echo.
    echo [*] 正在启动标准版本...
    echo [*] 启动时间约 30-60 秒，请耐心等待...
    echo.
    python app.py
) else (
    echo.
    echo [!] 无效选项，默认启动高性能优化版...
    echo.
    python app_optimized.py
)

:: 如果程序异常退出
echo.
echo ============================================
echo    服务已停止
echo ============================================
echo.
echo 按任意键退出...
pause >nul
