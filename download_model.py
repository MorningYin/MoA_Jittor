#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path
from modelscope.hub.snapshot_download import snapshot_download

def find_moa_jittor_dir():
    """
    自动查找MoA_Jittor目录
    从当前脚本位置开始向上查找，直到找到MoA_Jittor目录
    """
    current_path = Path(__file__).resolve().parent
    
    # 从当前目录开始向上查找MoA_Jittor
    while current_path != current_path.parent:
        if current_path.name == "MoA_Jittor":
            return current_path
        current_path = current_path.parent
    
    # 如果没找到，返回当前目录
    print("警告: 未找到MoA_Jittor目录，使用当前目录")
    return Path.cwd()

def download_llama_model():
    """
    下载Meta-Llama-3-8B-Instruct模型的original子目录
    """
    # 自动查找MoA_Jittor目录
    moa_dir = find_moa_jittor_dir()
    print(f"找到MoA_Jittor目录: {moa_dir}")
    
    # 设置下载目录
    local_dir = moa_dir / "Pre_trained_Models" / "LLaMA"
    local_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"开始下载模型到: {local_dir}")
    
    try:
        # 下载original子目录下的所有文件
        model_dir = snapshot_download(
            model_id='LLM-Research/Meta-Llama-3-8B-Instruct',
            allow_patterns='original/*',  # 只下载 original 子目录下的所有文件
            local_dir=str(local_dir)
        )
        
        print(f"'original' 文件已下载到: {model_dir}")
        
        # 列出下载的文件
        print("\n下载的文件列表:")
        for file_path in local_dir.rglob("*"):
            if file_path.is_file():
                file_size = file_path.stat().st_size
                print(f"  {file_path} ({file_size} bytes)")
        
        return True
        
    except Exception as e:
        print(f"下载失败: {e}")
        return False

if __name__ == "__main__":
    success = download_llama_model()
    if success:
        print("\n✅ 模型下载完成！")
    else:
        print("\n❌ 模型下载失败！")
        sys.exit(1)