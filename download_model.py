#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path
from modelscope.hub.snapshot_download import snapshot_download


def find_moa_jittor_dir():
    """自动查找MoA_Jittor目录"""
    current_path = Path(__file__).resolve().parent
    
    # 向上查找MoA_Jittor目录
    while current_path != current_path.parent:
        if current_path.name == "MoA_Jittor":
            return current_path
        current_path = current_path.parent
    
    print("警告: 未找到MoA_Jittor目录，使用当前目录")
    return Path.cwd()


def download_llama_model():
    """下载Meta-Llama-3-8B-Instruct模型"""
    # 查找项目目录
    moa_dir = find_moa_jittor_dir()
    print(f"找到MoA_Jittor目录: {moa_dir}")
    
    # 设置下载目录
    data_dir = Path("/HOME/thzskj_wfeng34/thzskj_wfeng34_1/MoA_Jittor/Pre_trained_Models")
    local_dir = data_dir / "LLaMA"
    local_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"开始下载模型到数据盘: {local_dir}")
    
    try:
        # 下载original子目录
        model_dir = snapshot_download(
            model_id='LLM-Research/Meta-Llama-3-8B-Instruct',
            allow_patterns='original/*',
            local_dir=str(local_dir)
        )
        
        print(f"'original' 文件已下载到: {model_dir}")
        
        # 列出下载的文件
        print("\n下载的文件列表:")
        for file_path in local_dir.rglob("*"):
            if file_path.is_file():
                file_size = file_path.stat().st_size
                print(f"  {file_path} ({file_size} bytes)")
        
        # 创建软链接
        project_model_dir = moa_dir / "Pre_trained_Models" / "LLaMA"
        project_model_dir.mkdir(parents=True, exist_ok=True)
        
        link_path = project_model_dir / "original"
        if not link_path.exists():
            link_path.symlink_to(local_dir / "original")
            print(f"已创建软链接: {link_path} -> {local_dir / 'original'}")
        
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