#!/usr/bin/env python3
"""
AIMAX395TTS - Qwen3-TTS模型下载脚本
专为AMD AI MAX 395 + 128GB内存平台优化

使用方法:
    python download_models.py [--all] [--1.7b] [--0.6b]

选项:
    --all   下载所有模型（1.7B + 0.6B）
    --1.7b  仅下载1.7B完整版模型
    --0.6b  仅下载0.6B轻量版模型（默认）
"""

import os
import sys
import argparse
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
    from tqdm import tqdm
except ImportError:
    print("❌ 请先安装依赖: pip install huggingface-hub tqdm")
    sys.exit(1)

# 模型配置
MODELS = {
    "1.7b": {
        "base": {
            "repo_id": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            "local_dir": "./Qwen3-TTS-12Hz-1.7B-Base",
            "size": "~3.5GB"
        },
        "voice_design": {
            "repo_id": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign-Full",
            "local_dir": "./Qwen3-TTS-12Hz-1.7B-VoiceDesign-Full",
            "size": "~3.8GB"
        },
        "custom_voice": {
            "repo_id": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice-Full",
            "local_dir": "./Qwen3-TTS-12Hz-1.7B-CustomVoice-Full",
            "size": "~3.8GB"
        }
    },
    "0.6b": {
        "base": {
            "repo_id": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            "local_dir": "./Qwen3-TTS-12Hz-0.6B-Base",
            "size": "~1.2GB"
        },
        "voice_design": {
            "repo_id": "Qwen/Qwen3-TTS-12Hz-0.6B-VoiceDesign",
            "local_dir": "./Qwen3-TTS-12Hz-0.6B-VoiceDesign",
            "size": "~1.3GB"
        },
        "custom_voice": {
            "repo_id": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            "local_dir": "./Qwen3-TTS-12Hz-0.6B-CustomVoice",
            "size": "~1.3GB"
        }
    }
}


def download_model(repo_id: str, local_dir: str, model_name: str) -> bool:
    """
    下载单个模型
    
    Args:
        repo_id: HuggingFace模型仓库ID
        local_dir: 本地保存路径
        model_name: 模型显示名称
    
    Returns:
        bool: 下载是否成功
    """
    try:
        print(f"\n📥 正在下载: {model_name}")
        print(f"   仓库: {repo_id}")
        print(f"   保存到: {local_dir}")
        
        # 检查是否已存在
        if os.path.exists(local_dir) and os.listdir(local_dir):
            print(f"   ⚠️  模型已存在，跳过下载")
            return True
        
        # 创建目录
        os.makedirs(local_dir, exist_ok=True)
        
        # 下载模型
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        
        print(f"   ✅ 下载完成: {model_name}")
        return True
        
    except Exception as e:
        print(f"   ❌ 下载失败: {model_name}")
        print(f"   错误: {str(e)}")
        return False


def download_models(version: str = "0.6b") -> dict:
    """
    下载指定版本的模型
    
    Args:
        version: 模型版本 ("0.6b" 或 "1.7b")
    
    Returns:
        dict: 下载结果统计
    """
    if version not in MODELS:
        print(f"❌ 未知模型版本: {version}")
        return {"success": 0, "failed": 0}
    
    models = MODELS[version]
    results = {"success": 0, "failed": 0}
    
    print(f"\n{'='*60}")
    print(f"🚀 开始下载 Qwen3-TTS {version.upper()} 模型")
    print(f"{'='*60}")
    
    for model_type, config in models.items():
        model_name = f"Qwen3-TTS-{version.upper()}-{model_type.replace('_', '-').title()}"
        if download_model(config["repo_id"], config["local_dir"], model_name):
            results["success"] += 1
        else:
            results["failed"] += 1
    
    return results


def print_summary(results_1_7b: dict, results_0_6b: dict):
    """打印下载摘要"""
    print(f"\n{'='*60}")
    print("📊 下载摘要")
    print(f"{'='*60}")
    
    total_success = results_1_7b["success"] + results_0_6b["success"]
    total_failed = results_1_7b["failed"] + results_0_6b["failed"]
    
    if results_1_7b["success"] > 0 or results_1_7b["failed"] > 0:
        print(f"\n1.7B 完整版模型:")
        print(f"   ✅ 成功: {results_1_7b['success']}")
        print(f"   ❌ 失败: {results_1_7b['failed']}")
    
    if results_0_6b["success"] > 0 or results_0_6b["failed"] > 0:
        print(f"\n0.6B 轻量版模型:")
        print(f"   ✅ 成功: {results_0_6b['success']}")
        print(f"   ❌ 失败: {results_0_6b['failed']}")
    
    print(f"\n总计:")
    print(f"   ✅ 成功: {total_success}")
    print(f"   ❌ 失败: {total_failed}")
    
    if total_failed == 0:
        print(f"\n🎉 所有模型下载成功！")
    else:
        print(f"\n⚠️  部分模型下载失败，请检查网络连接或手动下载")
    
    print(f"{'='*60}\n")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="AIMAX395TTS - Qwen3-TTS模型下载脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python download_models.py              # 下载0.6B轻量版模型（默认）
  python download_models.py --0.6b       # 下载0.6B轻量版模型
  python download_models.py --1.7b       # 下载1.7B完整版模型
  python download_models.py --all        # 下载所有模型
        """
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="下载所有模型（1.7B + 0.6B）"
    )
    parser.add_argument(
        "--1.7b",
        dest="download_1_7b",
        action="store_true",
        help="仅下载1.7B完整版模型"
    )
    parser.add_argument(
        "--0.6b",
        dest="download_0_6b",
        action="store_true",
        help="仅下载0.6B轻量版模型（默认）"
    )
    
    args = parser.parse_args()
    
    # 如果没有指定参数，默认下载0.6B
    if not (args.all or args.download_1_7b or args.download_0_6b):
        args.download_0_6b = True
    
    results_1_7b = {"success": 0, "failed": 0}
    results_0_6b = {"success": 0, "failed": 0}
    
    # 下载1.7B模型
    if args.all or args.download_1_7b:
        results_1_7b = download_models("1.7b")
    
    # 下载0.6B模型
    if args.all or args.download_0_6b:
        results_0_6b = download_models("0.6b")
    
    # 打印摘要
    print_summary(results_1_7b, results_0_6b)
    
    # 返回退出码
    total_failed = results_1_7b["failed"] + results_0_6b["failed"]
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
