import jittor as jt
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 
from Models.Tokenizer import Tokenizer    

# 加载 weights.pkl 文件
def load_weights(save_path):
    out_file = os.path.join(save_path, 'weights.pkl')
    data = jt.load(out_file)
    return data['tokens_weight'], data['type_weight']

tokenizer = Tokenizer(model_path='/hy-tmp/LLaMA/original/tokenizer.model')  # 替换为你的 tokenizer 路径

adapter_types = ['LoRAQ', 'LoRAK', 'LoRAV', 'Prompt', 'LoRAO', 'LoRAUP', 'Adapter', 'ParallelAdapter']  # 根据图调整

def plot_type_weight_heatmap(type_weight, save_path):
    # type_weight 是 [num_layers, num_adapter_types] 的 jt.Var
    type_weight_np = type_weight.numpy()  # 转换为 numpy
    num_layers = type_weight_np.shape[0]
    
    # 层标签，从高到低：31 到 0（假设 32 层）
    layer_labels = [str(i) for i in range(num_layers - 1, -1, -1)]
    
    # 使用 seaborn 绘制热力图
    plt.figure(figsize=(10, 12))
    sns.heatmap(type_weight_np, annot=False, cmap='viridis', xticklabels=adapter_types, yticklabels=layer_labels)
    plt.title('Type Weight Heatmap')
    plt.xlabel('Adapter Types')
    plt.ylabel('Layers')
    plt.savefig(os.path.join(save_path, 'type_weight_heatmap.png'))
    plt.close()

def plot_token_weight_heatmap(token_weights, save_path):
    # token_weights 是 dict: token_id -> weight_vector [num_adapter_types]
    tokens = list(token_weights.keys())
    decoded_tokens = [tokenizer.decode([token]) for token in tokens]  # decode 每个 token
    
    # stack 成 matrix [num_tokens, num_adapter_types]
    weights_np = np.stack([token_weights[token].numpy() for token in tokens])
    
    # 使用 seaborn 绘制热力图
    plt.figure(figsize=(10, 20))  # 调整大小根据 token 数量
    sns.heatmap(weights_np, annot=False, cmap='viridis', xticklabels=adapter_types, yticklabels=decoded_tokens)
    plt.title('Token Weight Heatmap')
    plt.xlabel('Adapter Types')
    plt.ylabel('Tokens')
    plt.savefig(os.path.join(save_path, 'token_weight_heatmap.png'))
    plt.close()

# 主函数
if __name__ == '__main__':
    save_path = '/root/MoA_Jittor/Test/AddSub'  # 替换为你的路径
    token_weights, type_weight = load_weights(save_path)
    plot_type_weight_heatmap(type_weight, save_path)
    plot_token_weight_heatmap(token_weights, save_path)
    print('Heatmaps saved to:', save_path)