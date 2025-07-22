import jittor as jt
from Utils.misc import setup_for_distributed, get_rank, get_world_size, add_weight_decay, save_model
from Models.LLaMA_Adapter import LLaMA_adapter
from Data.MathDataset import MathDataset
from Utils.EarlyStopper import EarlyStopper

import argparse
import datetime
import json
import numpy as np
import os
import time
from typing import List

from pathlib import Path
from Utils.misc import MetricLogger, SmoothedValue
from Train.engine_finetune import train_one_epoch
from tensorboardX import SummaryWriter

jt.flags.log_silent = 1
jt.flags.auto_mixed_precision_level = 1

def str2bool(v):
    """字符串转布尔值"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args_parser():
    """定义命令行参数"""
    parser = argparse.ArgumentParser('llama_adapterV2 finetuning', add_help=False)
    
    # 基础训练参数
    parser.add_argument('--batch_size', default=64, type=int, help='批次大小')
    parser.add_argument('--epochs', default=400, type=int, help='训练轮数')
    parser.add_argument('--accum_iter', default=1, type=int, help='梯度累积次数')
    parser.add_argument('--early_stop_patience', default=5, type=int, help='早停轮数')

    # 模型参数
    parser.add_argument('--llama_path', default='/path/to/llama', type=str, help='LLaMA模型路径')
    parser.add_argument('--max_seq_len', default=512, type=int, help='最大序列长度')
    parser.add_argument('--max_batch_size', default=32, type=int, help='推理批次大小')

    # LoRA参数
    parser.add_argument('--w_bias', default=False, type=bool, help='是否微调偏置项')
    parser.add_argument('--lora_layers', default='0-0', type=str, help='LoRA层范围')
    parser.add_argument('--lora_rank', default=16, type=int, help='LoRA秩')
    parser.add_argument('--lora_targets', default='Q,V', type=str, help='LoRA目标模块')
    parser.add_argument('--lora_alpha', default=8, type=int, help='LoRA缩放参数')
    parser.add_argument('--expert_num', default=1, type=int, help='专家数量')
    parser.add_argument('--hydra_moe', type=str2bool, nargs='?', const=True, default=False, help='启用Hydra MoE')
    parser.add_argument('--expert_weight', type=str2bool, nargs='?', const=True, default=False, help='专家权重设置')

    # Prompt Tuning参数
    parser.add_argument('--prompt_layers', default='0-0', type=str, help='Prompt层范围')
    parser.add_argument('--prompt_len', default=10, type=int, help='Prompt长度')

    # Parallel Adapter参数
    parser.add_argument('--p_adapter_layers', default='0-0', type=str, help='Parallel Adapter层范围')
    parser.add_argument('--p_adapter_size', default=16, type=int, help='Parallel Adapter隐藏层大小')
    parser.add_argument('--p_adapter_hydra', type=str2bool, nargs='?', const=True, default=False, help='Parallel Adapter Hydra模式')

    # Adapter路由参数
    parser.add_argument('--swi_x', default=0, type=int, help='适配器路由参数')
    parser.add_argument('--sparse', default=True, type=bool, help='是否稀疏路由')
    parser.add_argument('--if_trainable_gamma', default=True, type=bool, help='阈值是否可训练')
    parser.add_argument('--gamma', default=0.5, type=float, help='阈值')

    # 数据集参数
    parser.add_argument("--data_path", default="", type=str, help="训练数据路径")
    parser.add_argument("--val_data_path", default="", type=str, help="验证数据路径")

    # 优化器参数
    parser.add_argument('--weight_decay', type=float, default=0.05, help='权重衰减')
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='学习率')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR', help='基础学习率')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR', help='最小学习率')
    parser.add_argument('--warmup_epochs', type=float, default=1, metavar='N', help='预热轮数')

    # 输出和设备参数
    parser.add_argument('--output_dir', default='./output', help='输出目录')
    parser.add_argument('--device', default='cuda', help='训练设备')
    parser.add_argument('--seed', default=0, type=int, help='随机种子')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='开始轮数')

    # 断点续训参数
    parser.add_argument('--trainable_params', type=str2bool, nargs='?', const=True, default=False, help='是否断点续训')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='断点续训路径')

    return parser


def prepare_args(data_path):
    default_cli = [
        '--llama_path', '/hy-tmp/LLaMA/original',
        '--data_path',  data_path,
        '--device', 'cuda',
        '--batch_size', '16',
        '--epochs', '5',
        '--max_seq_len', '300',
        '--lr', '5e-5',
        '--accum_iter', '1',
        '--lora_layers', '0-32',
        '--lora_rank', '8',
        '--lora_targets', 'Q,K,V,O',
        '--prompt_layers', '0-32',
        '--p_adapter_layers', '0-32',
        '--swi_x', '1',
        '--seed', '0',
        '--output_dir', '/root/MoA_Jittor/Output',
        '--early_stop_patience', '5',
        '--sparse', 'True',
        '--if_trainable_gamma', 'True',
        '--gamma', '0.5',
        '--trainable_params', 'False',
        '--checkpoint_path', '/root/MoA_Jittor/Output/LoRA_0-32_r8_a8_Q,K,V,O_Prompt_0-32_len10_PAdapter_0-32_size16_swi_x1_lr5e-5_bs2_Debug_seed0/checkpoint-0.pth'
    ]
    args = get_args_parser().parse_args(default_cli)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    return args

def finetune(args_list: List[argparse.Namespace], model : LLaMA_adapter):
    args = args_list[0]
    llama_tokenzier_path = os.path.join(args.llama_path, 'tokenizer.model')
    # 创建数据集
    dataset_train = MathDataset(
        data_paths=[args.data_path for args in args_list], 
        tokenizer_path=llama_tokenzier_path, 
        max_tokens=args.max_seq_len, 
        partition="train",
        val_ratio=0.05,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    dataset_train.datainit()

    dataset_val = MathDataset(
        data_paths=[args.data_path for args in args_list], 
        tokenizer_path=llama_tokenzier_path, 
        max_tokens=args.max_seq_len, 
        partition="val",
        val_ratio=0.05,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    dataset_val.datainit()

    print('====================================================== 数据集大小 =======================================================')
    print(f'训练集大小: {len(dataset_train)}')
    print(f'验证集大小: {len(dataset_val)}')

    # 配置优化器
    param_groups = add_weight_decay(model, args.weight_decay)
    optimizer = jt.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    early_stopper = EarlyStopper(patience=args.early_stop_patience, min_delta=0.01, mode='min')

    if args.trainable_params:
        checkpoint = jt.load(args.checkpoint_path)
        args.start_epoch = checkpoint['epoch']
        data_iter_step = int(checkpoint['trainable_params'][0].split('_')[-1])
        model.load_state_dict(checkpoint['trainable_params'][1])
        optimizer.load_state_dict(checkpoint['optimizer'])
        early_stopper.load_state(checkpoint['early_stopper'])
        metric_logger = MetricLogger()
        metric_logger.load_state_dict(checkpoint['metric_logger'])
        val_metric_logger = MetricLogger()
        val_metric_logger.load_state_dict(checkpoint['val_metric_logger'])
        print(f'================================================== 断点续训 =======================================================')
        print(f'断点续训轮数: {args.start_epoch}')
        print(f'断点续训步数: {data_iter_step}')
    else:       
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('closs', SmoothedValue(window_size=5, fmt='{median:.6f}'))
        metric_logger.add_meter('mloss', SmoothedValue(window_size=5, fmt='{median:.6f}'))

        val_metric_logger = MetricLogger(delimiter="  ")
        val_metric_logger.add_meter('loss', SmoothedValue(window_size=1, fmt='{value:.6f}'))

        data_iter_step = -1

    # 配置日志
    args.log_dir = os.path.join(args.output_dir, 'log')
    
    if get_rank() == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    # 开始训练
    print(f"============================================ 开始训练  ==================================================")
    start_time = time.time()
    
    for epoch in range(args.start_epoch, args.epochs):
        # 训练一个epoch
        train_stats, val_stats, early_stopped = train_one_epoch(
            model=model,
            data_loader=dataset_train,
            val_loader=dataset_val,
            optimizer=optimizer,
            epoch=epoch,
            data_iter_step=data_iter_step,
            args=args, 
            log_writer=log_writer,
            early_stopper=early_stopper,
            val_interval=50,
            metric_logger=metric_logger,
            val_metric_logger=val_metric_logger
        )

        data_iter_step = -1

        # 保存模型
        if args.output_dir:
            save_model(args=args, epoch=epoch, model=model, optimizer=optimizer, early_stopper=early_stopper, log_writer=log_writer, metric_logger=metric_logger, val_metric_logger=val_metric_logger)

        # 记录日志
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch,
                     **{f'val_{k}': v for k, v in val_stats.items()}}

        if args.output_dir and get_rank() == 0:
            if log_writer is not None:
                log_writer.flush()
            
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        if early_stopped:
            print(f'================================================== 早停 =======================================================')
            break

        print(f'================================================== 第 {epoch} 轮训练完成 =======================================================')

    # 训练完成
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('训练时间: {}'.format(total_time_str))
    print(f'================================================  训练完成 =====================================================')


def main(args_list):
    # 设置设备
    args = args_list[0]
    if args.device == 'cuda':
        jt.flags.use_cuda = 1
    
    if jt.in_mpi:
        setup_for_distributed(jt.rank == 0)

    print(f'==================================================== Job Start ====================================================')
    print('工作文件夹: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    
    
    print("================================================== 训练参数配置 =====================================================")
    
    # 按类别分组打印参数
    param_groups = {
        "🔧 基础训练参数": [
            'batch_size', 'epochs', 'accum_iter', 'seed', 'start_epoch'
        ],
        "🤖 模型参数": [
            'llama_path', 'max_seq_len', 'max_batch_size', 'device'
        ],
        "🎯 LoRA参数": [
            'w_bias', 'lora_layers', 'lora_rank', 'lora_targets', 
            'lora_alpha', 'expert_num', 'hydra_moe', 'expert_weight'
        ],
        "💬 Prompt Tuning参数": [
            'prompt_layers', 'prompt_len'
        ],
        "🔗 Parallel Adapter参数": [
            'p_adapter_layers', 'p_adapter_size', 'p_adapter_hydra'
        ],
        "🎛️ 适配器路由参数": [
            'swi_x'
        ],
        "⚙️ 优化器参数": [
            'weight_decay', 'lr', 'blr', 'min_lr', 'warmup_epochs'
        ],
        "📊 数据集参数": [
            'data_path', 'val_data_path', 'num_workers'
        ],
        "📁 输出参数": [
            'output_dir'
        ]
    }
    
    for group_name, param_names in param_groups.items():
        print(f"\n{group_name}:")
        print("-" * 30)
        for param_name in param_names:
            if hasattr(args, param_name):
                value = getattr(args, param_name)
                # 格式化显示
                if isinstance(value, str) and len(value) > 50:
                    # 长路径截断显示
                    display_value = value[:30] + "..." + value[-20:]
                else:
                    display_value = value
                print(f"  {param_name:20} = {display_value}")

    # 设置随机种子
    seed = args.seed + get_rank()
    jt.set_global_seed(seed)
    np.random.seed(seed)

    # 初始化模型
    llama_type = ''
    llama_ckpt_dir = os.path.join(args.llama_path, llama_type)
    llama_tokenzier_path = os.path.join(args.llama_path, 'tokenizer.model')
    model = LLaMA_adapter(args, llama_ckpt_dir, llama_tokenzier_path)
    model.float_auto()
    model.train()
    model.init()
    
    # 统计可训练参数
    print("================================================== 可训练参数 =====================================================")
    trainable_params_sum = 0
    trainable_params_kv = []
    
    for key, val in model.named_parameters():
        if val.requires_grad:
            trainable_params_kv.append((key, val.shape))
            trainable_params_sum += int(val.numel())
    
    print("可训练参数总量: {}".format(trainable_params_sum))

    # 计算训练配置
    eff_batch_size = args.batch_size * args.accum_iter * get_world_size()

    # 计算学习率
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print("================================================== 训练配置 =====================================================")
    print("基础学习率: %.2e" % (args.lr * 256 / eff_batch_size))
    print("实际学习率: %.2e" % args.lr)
    print("梯度累积次数: %d" % args.accum_iter)
    print("有效批次大小: %d" % eff_batch_size)

    finetune(args_list, model)

    if jt.in_mpi:
        jt.sync_all(True)

    print(f'==================================================== Job End ====================================================')

if __name__ == '__main__':
    datas = ['AddSub/addsub_1.json', 'AQuA/aqua_1.json', 'gsm8k/gsm8k_1.json', 'MultiArith/multiarith_1.json', 'SingleEq/singleeq_1.json', 'SVAMP/svamp_1.json']
    # datas = ['Debug/2.json']

    args_list = []
    for data in datas:
        data_path = '/root/MoA_Jittor/Data/Dataset/math_commonsense/' + data
        args = prepare_args(data_path)
        args_list.append(args)

    main(args_list)