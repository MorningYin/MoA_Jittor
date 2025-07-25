import jittor as jt
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from Models.Tokenizer import Tokenizer

# 定义自定义颜色映射函数，指定渐变点
def create_custom_cmap(color_points, n_bins=256):
    # color_points 是一个列表，格式 [(value, (r, g, b)), ...]
    # 自定义渐变：0.0 (深紫色) -> 0.6 (橙色) -> 0.8 (红色) -> 1.0 (黄色)
    colors = [
        (0.0, (0, 0, 1)),  # 深紫色
        (0.4, (0.5, 0, 0.5)),    # 深紫色
        (1.0, (1, 1, 0))       # 黄色
    ]
    if color_points:
        colors = color_points
    cmap = LinearSegmentedColormap.from_list('custom_cmap', [c[1] for c in sorted(colors, key=lambda x: x[0])], N=n_bins)
    return cmap

# 加载 weights.pkl 文件
def load_weights(save_path):
    out_file = os.path.join(save_path, 'weights.pkl')
    data = jt.load(out_file)
    return data['tokens_weight'], data['type_weight']

tokenizer = Tokenizer(model_path='/hy-tmp/LLaMA/original/tokenizer.model')  # 替换为你的 tokenizer 路径
adapter_types = ['LoRAQ', 'LoRAK', 'LoRAV', 'Prompt', 'LoRAO', 'LoRAUP', 'ParallelAdapter']  # 根据图调整

# 自定义颜色点（可选修改）
custom_color_points1 = [
    (0.0, (0, 0, 1)),  # 深紫色
    (0.4, (0.5, 0, 0.5)),
    (0.8, (1, 0.5, 0)),
    (1.0, (1, 1, 0))       # 黄色
]

def plot_type_weight_heatmap(type_weight, save_path, cmap):
    # type_weight 是 [num_layers, num_adapter_types] 的 jt.Var
    type_weight_np = type_weight.numpy()  # 转换为 numpy
    num_layers = type_weight_np.shape[0]
    
    # 层标签，从高到低：31 到 0（假设 32 层）
    layer_labels = [str(i) for i in range(num_layers - 1, -1, -1)]
    
    # 使用 seaborn 绘制热力图，使用自定义颜色映射
    plt.figure(figsize=(10, 12))
    sns.heatmap(type_weight_np, annot=False, cmap=cmap, vmin=0, vmax=1, xticklabels=adapter_types, yticklabels=layer_labels)
    plt.title('Type Weight Heatmap')
    plt.xlabel('Adapter Types')
    plt.ylabel('Layers')
    plt.savefig(os.path.join(save_path, 'type_weight_heatmap.png'))
    plt.close()

def plot_token_weight_heatmap(token_weights, save_path, cmap):
    # token_weights 是 dict: token_id -> weight_vector [num_adapter_types]
    tokens = list(token_weights.keys())[:100]  # 取前 100 个 token
    decoded_tokens = [tokenizer.decode([token]) for token in tokens]  # decode 每个 token
    
    # stack 成 matrix [num_tokens, num_adapter_types]
    weights_np = np.stack([1 - token_weights[token].numpy() * 2 for token in tokens])

    vmax = np.max(weights_np)
    vmin = np.min(weights_np)

    # l = vmax - vmin
    # vmax = vmax + l * 0.3
    # vmin = vmin - l * 0.3
    
    # 使用 seaborn 绘制热力图，使用自定义颜色映射
    plt.figure(figsize=(10, 20))  # 调整大小根据 token 数量
    sns.heatmap(weights_np, annot=False, cmap=cmap, vmin=0, vmax=1, xticklabels=adapter_types, yticklabels=decoded_tokens)
    plt.title('Token Weight Heatmap')
    plt.xlabel('Adapter Types')
    plt.ylabel('Tokens')
    plt.savefig(os.path.join(save_path, 'token_weight_heatmap.png'))
    plt.close()
    
# 主函数
if __name__ == '__main__':
    save_path = '/root/MoA_Jittor/Test_seed125/AddSub'  # 替换为你的路径
    token_weights, type_weight = load_weights(save_path)
    
    # 创建自定义颜色映射
    cmap1 = create_custom_cmap(custom_color_points1)
    
    plot_type_weight_heatmap(type_weight, save_path, cmap1)
    plot_token_weight_heatmap(token_weights, save_path, cmap1)
    print('Heatmaps saved to:', save_path)