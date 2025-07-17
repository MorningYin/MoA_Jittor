import jittor as jt

# Jittor DataLoader
from jittor.dataset import DataLoader as JTDataLoader
from Utils.misc import init_distributed_mode, get_rank, get_world_size, add_weight_decay, save_model
from Models.LLaMA_Adapter import LLaMA_adapter

from Data.Dataset import FinetuneDataset

import argparse
import sys
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

from Train.engine_finetune import train_one_epoch

# TensorBoard 写事件文件
from tensorboardX import SummaryWriter

def str2bool(v):
    """
    将字符串转换为布尔值的工具函数
    
    该函数用于处理命令行参数中的布尔值，支持多种字符串格式：
    - 如果输入已经是布尔类型，直接返回
    - 支持 'yes', 'true', 't', 'y', '1' 等表示 True 的字符串
    - 支持 'no', 'false', 'f', 'n', '0' 等表示 False 的字符串
    - 其他情况抛出 ArgumentTypeError 异常
    
    Args:
        v: 输入值，可以是布尔类型或字符串
        
    Returns:
        bool: 转换后的布尔值
        
    Raises:
        argparse.ArgumentTypeError: 当输入无法转换为布尔值时抛出
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args_parser():
    """
    定义命令行参数解析器
    
    该函数创建了一个 ArgumentParser 对象，用于解析训练脚本的命令行参数。
    参数分为以下几个主要类别：
    1. 基础训练参数（批次大小、训练轮数等）
    2. 模型参数（LLaMA路径、序列长度等）
    3. LoRA相关参数（LoRA配置、专家数量等）
    4. Prompt Tuning参数
    5. Parallel Adapter参数
    6. 优化器参数（学习率、权重衰减等）
    7. 数据集参数
    8. 分布式训练参数
    
    Returns:
        argparse.ArgumentParser: 配置好的参数解析器
    """
    parser = argparse.ArgumentParser('llama_adapterV2 finetuning', add_help=False)
    
    # ==================== 基础训练参数 ====================
    parser.add_argument('--batch_size', default=64, type=int,
                        help='每个GPU的批次大小 (有效批次大小 = batch_size * accum_iter * GPU数量)')
    parser.add_argument('--epochs', default=400, type=int,
                        help='训练轮数')
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='梯度累积迭代次数 (用于在内存限制下增加有效批次大小)')

    # ==================== 模型参数 ====================
    parser.add_argument('--llama_path', default='/path/to/llama', type=str,
                        help='LLaMA预训练模型路径')
    parser.add_argument('--max_seq_len', default=512, type=int, 
                        help='输入序列的最大长度')
    parser.add_argument('--max_batch_size', default=32, type=int, 
                        help='推理时的最大批次大小')

    # ==================== LoRA参数 ====================
    parser.add_argument('--w_bias', default=False, type=bool, 
                        help='是否微调偏置项')
    parser.add_argument('--lora_layers', default='0-0', type=str, 
                        help='应用LoRA的层范围，格式如"0-32"表示第0到32层')
    parser.add_argument('--lora_rank', default=16, type=int, 
                        help='LoRA的秩 (rank)，控制低秩分解的维度')
    parser.add_argument('--lora_targets', default='Q,V', type=str, 
                        help='LoRA应用的目标模块，可选：Q,K,V,O,FFN_UP,FFN_DOWN')
    parser.add_argument('--lora_alpha', default=8, type=int, 
                        help='LoRA的缩放参数alpha')
    parser.add_argument('--expert_num', default=1, type=int, 
                        help='专家数量 (MoE中的专家个数)')
    parser.add_argument('--hydra_moe', type=str2bool, nargs='?', const=True, default=False, 
                        help='是否启用Hydra MoE (多头专家混合)')
    parser.add_argument('--expert_weight', type=str2bool, nargs='?', const=True, default=False, 
                        help='是否根据专家参数数量设置专家权重')

    # ==================== Prompt Tuning参数 ====================
    parser.add_argument('--prompt_layers', default='0-0', type=str, 
                        help='应用Prompt Tuning的层范围')
    parser.add_argument('--prompt_len', default=10, type=int, 
                        help='Prompt的长度')

    # ==================== Parallel Adapter参数 ====================
    parser.add_argument('--p_adapter_layers', default='0-0', type=str, 
                        help='应用Parallel Adapter的层范围')
    parser.add_argument('--p_adapter_size', default=16, type=int, 
                        help='Parallel Adapter的隐藏层大小')
    parser.add_argument('--p_adapter_hydra', type=str2bool, nargs='?', const=True, default=False, 
                        help='Parallel Adapter是否使用Hydra模式')

    # ==================== Adapter类型路由参数 ====================
    parser.add_argument('--swi_x', default=0, type=int, 
                        help='适配器类型路由参数：0表示普通Linear，否则swi_x * adapter_type作为SwiGLU路由器的隐藏层大小')

    # ==================== 优化器参数 ====================
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='权重衰减系数 (默认: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='学习率 (绝对学习率)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='基础学习率：绝对学习率 = 基础学习率 * 总批次大小 / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='循环调度器的最小学习率下界')
    parser.add_argument('--warmup_epochs', type=float, default=1, metavar='N',
                        help='学习率预热轮数')

    # ==================== 数据集参数 ====================
    parser.add_argument("--data_path", default="", type=str, 
                        help="训练数据集路径")
    parser.add_argument("--val_data_path", default="", type=str, 
                        help="验证数据集路径")
    parser.add_argument('--num_workers', default=10, type=int,
                        help='数据加载器的工作进程数')

    # ==================== 输出和设备参数 ====================
    parser.add_argument('--output_dir', default='./output',
                        help='模型和日志保存路径')
    parser.add_argument('--device', default='cuda',
                        help='训练/测试使用的设备')
    parser.add_argument('--seed', default=0, type=int,
                        help='随机种子，用于结果复现')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='开始训练的轮数 (用于断点续训)')


    return parser


def main(args):
    if args.device == 'cuda':
        jt.flags.use_cuda = 1
        
    init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    # -------------------- 固定随机种子 --------------------
    seed = args.seed + get_rank()
    jt.set_global_seed(seed)  # Jittor 全局随机种子
    np.random.seed(seed)      # NumPy 随机种子
    # Jittor 无需 cudnn.benchmark 设置

    # 定义并初始化 LLaMA 模型
    # 注意：这里 llama_type 被硬编码为空字符串，原本可能从 args.llama_type 获取
    llama_type = ''
    
    # 构建模型检查点目录路径
    # 格式：args.llama_path + llama_type (当前为空字符串)
    llama_ckpt_dir = os.path.join(args.llama_path, llama_type)
    
    # 构建分词器模型文件路径
    # 分词器文件通常位于预训练模型根目录下的 'tokenizer.model'
    llama_tokenzier_path = os.path.join(args.llama_path, 'tokenizer.model')
    
    # 创建 LLaMA_adapter 模型实例
    # 传入参数：args(配置参数), llama_ckpt_dir(模型检查点目录), llama_tokenzier_path(分词器路径)
    # LLaMA_adapter 是自定义的适配器模型类，集成了 LoRA、Prompt Tuning、Parallel Adapter 等功能
    model = LLaMA_adapter(args, llama_ckpt_dir, llama_tokenzier_path)
    
    # ==================== 模型信息打印 ====================
    # 打印完整的模型结构信息
    # 包括：模型类型、层结构、参数数量等详细信息
    print("Model = %s" % str(model))

    # ==================== 可训练参数统计 ====================
    # 统计并打印模型中的可训练参数信息
    print("Trainable Params:")
    trainable_params_sum = 0
    trainable_params_kv = []
    
    # 遍历模型的所有命名参数
    for key, val in model.named_parameters():
        if not getattr(val, "stop_grad", False):  # Jittor: stop_grad=False 表示可训练
            trainable_params_kv.append((key, val.shape))  # 记录参数名称和形状
            trainable_params_sum += int(val.numel())  # 累加参数数量
    
    # 注释掉详细参数列表打印，避免输出过多信息
    # print(trainable_params_kv)  # 会打印每个可训练参数的名称和形状
    
    # 打印总的可训练参数数量
    # 这个数字反映了实际需要更新的参数数量，对于参数高效微调很重要
    print(f'total {trainable_params_sum} trainable params')
    
    # 注释掉的替代打印方式，功能相同但格式不同
    # print([(key, val.shape) for key, val in model.named_parameters() if val.requires_grad])

    # ==================== 训练配置计算 ====================
    # 计算有效批次大小：单GPU批次大小 × 梯度累积次数 × GPU数量
    # 这个值决定了实际用于梯度更新的样本数量
    eff_batch_size = args.batch_size * args.accum_iter * get_world_size()

    # ==================== 学习率计算 ====================
    # 如果只指定了基础学习率(blr)，则根据有效批次大小计算实际学习率
    # 公式：实际学习率 = 基础学习率 × 有效批次大小 / 256
    # 这是深度学习中的标准做法，确保不同批次大小下的训练稳定性
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    # 打印学习率相关信息，用于调试和验证
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))  # 反推基础学习率
    print("actual lr: %.2e" % args.lr)  # 实际使用的学习率

    # 打印训练配置信息
    print("accumulate grad iterations: %d" % args.accum_iter)  # 梯度累积次数
    print("effective batch size: %d" % eff_batch_size)  # 有效批次大小

    # ==================== 优化器配置 ====================
    # 参考 timm 库的做法：为偏置项和归一化层设置权重衰减为0
    # 这样可以避免对这些层进行不必要的正则化
    param_groups = add_weight_decay(model, args.weight_decay)
    
    # 创建 AdamW 优化器
    # lr: 学习率
    # betas=(0.9, 0.95): Adam优化器的动量参数，0.95是LLaMA论文中使用的值
    optimizer = jt.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)  # 打印优化器配置信息
    
    # 不再使用自定义损失缩放器，直接依赖 Jittor 内部 AMP
    loss_scaler = None

    # 注释掉的模型加载代码，用于从预训练检查点恢复训练
    # misc.load_model(model_without_ddp, args.pretrained_path)

    # ==================== 数据集创建 ====================
    # 创建训练数据集实例
    # FinetuneDataset: 自定义的数据集类，专门用于微调任务
    # args.data_path: 训练数据文件路径
    # llama_tokenzier_path: 分词器路径，用于文本tokenization
    # max_tokens=args.max_seq_len: 最大序列长度，控制输入文本的长度
    # partition="train": 指定为训练分区
    dataset_train = FinetuneDataset(
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        data_path=args.data_path, 
        tokenizer_path=llama_tokenzier_path, 
        max_tokens=args.max_seq_len, 
        partition="train"
    )
    print(dataset_train)  # 打印数据集信息，包括样本数量等
    
    data_loader_train = JTDataLoader(dataset_train)

    # ==================== 日志记录器配置 ====================
    # 设置TensorBoard日志目录
    args.log_dir = os.path.join(args.output_dir, 'log')
    
    # 只在主进程中创建日志记录器，避免多进程冲突
    if get_rank() == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)  # 创建日志目录
        log_writer = SummaryWriter(log_dir=args.log_dir)  # 创建TensorBoard写入器
    else:
        log_writer = None  # 非主进程不创建日志记录器

    # ==================== 训练循环开始 ====================
    print(f"Start training for {args.epochs} epochs")  # 打印训练开始信息
    start_time = time.time()  # 记录训练开始时间
    
    # 主训练循环：遍历每个训练轮次
    for epoch in range(args.start_epoch, args.epochs):
        # ==================== 单轮训练 ====================
        # 调用训练函数，执行一个完整的训练轮次
        # 返回该轮次的训练统计信息（损失、准确率等）
        train_stats = train_one_epoch(
            model=model,
            data_loader=data_loader_train,
            optimizer=optimizer,
            epoch=epoch,
            args=args, 
            log_writer=log_writer)

        # ==================== 模型保存 ====================
        # 定期保存模型检查点（每5个epoch或最后一个epoch）
        # 保存内容包括：模型状态、优化器状态、训练轮次、损失缩放器状态
        if args.output_dir and (epoch % 5 == 0 or epoch + 1 == args.epochs):
            save_model(args=args, model=model, optimizer=optimizer, epoch=epoch)

        # ==================== 日志记录 ====================
        # 构建日志统计信息字典
        # 包含训练指标、验证指标（这里验证指标使用训练指标，因为没有单独的验证集）
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},  # 训练指标
                     'epoch': epoch,  # 当前轮次
                     **{f'val_{k}': v for k, v in train_stats.items()}}  # 验证指标（使用训练指标）

        # ==================== 日志写入 ====================
        # 只在主进程中写入日志，避免多进程冲突
        if args.output_dir and get_rank() == 0:
            # 刷新TensorBoard日志
            if log_writer is not None:
                log_writer.flush()
            
            # 将日志信息写入文本文件
            # 使用JSON格式保存，便于后续分析
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    # ==================== 训练完成 ====================
    # 计算并打印总训练时间
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = get_args_parser()

    # 如果未提供命令行参数，则使用默认调试参数
    if len(sys.argv) == 1:
        default_cli = [
            '--llama_path', '/root/MoA_Jittor/Pretrained_Model/Meta-Llama-3.1-8B',
            '--data_path',  '/root/MoA_Jittor/Data/Dataset/commonsense_15k/train.json',
            '--output_dir', './output_jt',
            '--device', 'cuda',
            '--batch_size', '2',
            '--epochs', '10',
            '--max_seq_len', '512',
            '--lr', '1e-4',
            '--accum_iter', '1',
            '--lora_layers', '0-32',
            '--lora_rank', '8',
            '--lora_targets', 'Q,K,V,O',
            '--prompt_layers', '2-32',
            '--p_adapter_layers', '2-32',
            '--swi_x', '1'
        ]
        print('[DEBUG] 使用默认参数:', ' '.join(default_cli))
        args = parser.parse_args(default_cli)
    else:
        args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
