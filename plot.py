import jittor as jt
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from Models.Tokenizer import Tokenizer

# 全局配置
tokenizer = Tokenizer(model_path='/hy-tmp/LLaMA/original/tokenizer.model')
adapter_types = ['LoRAQ', 'LoRAK', 'LoRAV', 'Prompt', 'LoRAO', 'LoRAUP', 'ParallelAdapter']

# 自定义颜色点
custom_color_points1 = [
    (0.0, (0, 0, 1)),      # 深紫色
    (0.4, (0.5, 0, 0.5)),  # 紫色
    (0.8, (1, 0.5, 0)),    # 橙色
    (1.0, (1, 1, 0))       # 黄色
]


def create_custom_cmap(color_points, n_bins=256):
    """创建自定义颜色映射"""
    colors = [
        (0.0, (0, 0, 1)),      # 深紫色
        (0.4, (0.5, 0, 0.5)),  # 紫色
        (1.0, (1, 1, 0))       # 黄色
    ]
    if color_points:
        colors = color_points
    cmap = LinearSegmentedColormap.from_list('custom_cmap', [c[1] for c in sorted(colors, key=lambda x: x[0])], N=n_bins)
    return cmap


def load_weights(save_path):
    """加载权重文件"""
    out_file = os.path.join(save_path, 'weights.pkl')
    data = jt.load(out_file)
    return data['tokens_weight'], data['type_weight']


def plot_type_weight_heatmap(type_weight, save_path, cmap):
    """绘制类型权重热力图"""
    type_weight_np = type_weight.numpy()
    num_layers = type_weight_np.shape[0]
    
    # 层标签，从高到低
    layer_labels = [str(i) for i in range(num_layers - 1, -1, -1)]
    
    plt.figure(figsize=(10, 12))
    sns.heatmap(type_weight_np, annot=False, cmap=cmap, vmin=0, vmax=1, 
                xticklabels=adapter_types, yticklabels=layer_labels)
    plt.title('Type Weight Heatmap')
    plt.xlabel('Adapter Types')
    plt.ylabel('Layers')
    plt.savefig(os.path.join(save_path, 'type_weight_heatmap.png'))
    plt.close()


def plot_token_weight_heatmap(token_weights, save_path, cmap):
    """绘制token权重热力图"""
    # 取前100个token
    tokens = list(token_weights.keys())[:100]
    decoded_tokens = [tokenizer.decode([token]) for token in tokens]
    
    # 构建权重矩阵
    weights_np = np.stack([token_weights[token].numpy() * 2 for token in tokens])
    
    plt.figure(figsize=(10, 20))
    sns.heatmap(weights_np, annot=False, cmap=cmap, vmin=0, vmax=1, 
                xticklabels=adapter_types, yticklabels=decoded_tokens)
    plt.title('Token Weight Heatmap')
    plt.xlabel('Adapter Types')
    plt.ylabel('Tokens')
    plt.savefig(os.path.join(save_path, 'token_weight_heatmap.png'))
    plt.close()


if __name__ == '__main__':
    # 设置路径
    save_path = '/root/MoA_Jittor/Test_seed125/AddSub'
    
    # 加载权重
    token_weights, type_weight = load_weights(save_path)
    
    # 创建颜色映射
    cmap1 = create_custom_cmap(custom_color_points1)
    
    # 绘制热力图
    plot_type_weight_heatmap(type_weight, save_path, cmap1)
    plot_token_weight_heatmap(token_weights, save_path, cmap1)
    print('Heatmaps saved to:', save_path)