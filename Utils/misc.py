# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import builtins
import datetime
import os
import time
from collections import defaultdict, deque
from pathlib import Path
import urllib
from tqdm import tqdm
import pynvml

import jittor as jt

# Jittor配置
jt.flags.log_silent = 1


class SmoothedValue(object):
    """平滑值跟踪器，用于监控训练指标"""

    def __init__(self, window_size=20, fmt=None):
        # 初始化平滑值跟踪器
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.window_size = window_size
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        """更新跟踪器的值"""
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """在多进程间同步统计数据"""
        if jt.in_mpi == None:
            return

        # MPI同步
        t = jt.array([self.count, self.total], dtype=jt.float64)
        t.mpi_all_reduce(op="sum")
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    def get_state(self):
        """获取状态字典"""
        return {
            'count': self.count,
            'total': self.total,
            'deque': list(self.deque),
            'fmt': self.fmt,
            'window_size': self.window_size
        }

    def load_state(self, state_dict):
        """从状态字典加载数据"""
        self.deque = deque(state_dict['deque'], maxlen=state_dict['window_size'])
        self.count = state_dict['count']
        self.total = state_dict['total']
        self.fmt = state_dict['fmt']

    @property
    def median(self):
        """计算滑动窗口中值的中位数"""
        d = jt.array(list(self.deque))
        return float(jt.median(d))

    @property
    def avg(self):
        """计算滑动窗口中值的平均值"""
        d = jt.array(list(self.deque), dtype=jt.float16)
        return float(jt.mean(d))
    
    @property
    def global_avg(self):
        """计算全局平均值"""
        return self.total / self.count

    @property
    def max(self):
        """获取滑动窗口中的最大值"""
        return max(self.deque)

    @property
    def value(self):
        """获取最新的值"""
        return self.deque[-1]

    def __str__(self):
        """返回格式化的字符串表示"""
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    """指标记录器，管理训练过程中的多个指标"""

    def __init__(self, delimiter="\t"):
        # 初始化指标记录器
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        """更新多个指标的值"""
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, jt.Var):
                v = float(v)
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def get_state_dict(self):
        """获取状态字典"""
        return {
            'meters': {k: v.get_state() for k, v in self.meters.items()},
            'delimiter': self.delimiter
        }

    def load_state_dict(self, state_dict):
        """从状态字典加载数据"""
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = state_dict['delimiter']
        for k, v in state_dict['meters'].items():
            self.meters[k].load_state(v)

    def __getattr__(self, attr):
        """动态属性访问，允许直接访问指标"""
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        """返回所有指标的字符串表示"""
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        """同步所有指标在多进程间的数据"""
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        """添加新的指标"""
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, if_print=True, header=None):
        """迭代器包装器，自动记录训练进度"""
        i = 0
        if not header:
            header = ''
        
        # 初始化GPU监控
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(jt.rank)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        
        # 时间跟踪
        start_time = time.time()
        end = time.time()
        
        # 创建时间跟踪器
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        
        # 构建日志消息格式
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'Eta: {eta}',
            '{meters}',
            'Time: {time}',
            'Data: {data}'
        ]
        
        if jt.has_cuda:
            log_msg.append('Mem: {used_mb:.0f} / {total_mb:.0f} MB')
        
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        
        # 迭代处理
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            
            # 定期打印进度
            if (i % print_freq == 0 or i == len(iterable) - 1) and if_print == True:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                
                if jt.has_cuda:
                    try:
                        total_mb = info.total  / 1024**2
                        used_mb  = info.used   / 1024**2
                    except Exception:
                        total_mb = 0
                        used_mb = 0
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        used_mb=used_mb, total_mb=total_mb))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            
            i += 1
            end = time.time()
        
        # 打印总时间统计
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def setup_for_distributed(is_master):
    """设置分布式训练中的打印控制"""
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    """检测是否处于MPI分布式环境"""
    return jt.in_mpi


def get_world_size():
    """返回总进程数"""
    if not jt.in_mpi:
        return 1
    return jt.world_size


def get_rank():
    """获取当前进程的排名"""
    if not jt.in_mpi:
        return 0
    return jt.rank


def is_main_process():
    """判断当前进程是否为主进程"""
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    """只在主进程中保存文件"""
    if is_main_process():
        jt.save(*args, **kwargs)


def init_distributed_mode(args):
    """分布式初始化"""
    # 初始化分布式参数
    args.distributed = False
    args.rank = 0
    args.world_size = 1

    # 检测MPI环境
    if jt.in_mpi:
        args.rank = jt.rank
        args.world_size = jt.world_size
        args.distributed = args.world_size > 1

    # GPU设备设置
    if jt.has_cuda:
        if args.distributed:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
            print(f"[Jittor] Rank {args.rank} using GPU {args.gpu}")

    # 设置分布式打印
    if args.distributed:
        setup_for_distributed(args.rank == 0)

    # 打印初始化信息
    if args.distributed and args.rank == 0:
        print(f"[Jittor] Distributed init done | rank {args.rank}/{args.world_size} | nproc {jt.get_nproc()}")
    elif not args.distributed:
        print("[Jittor] Running in single-process mode.")


def save_model(args, epoch, model, optimizer, early_stopper, log_writer, metric_logger, val_metric_logger):
    """保存模型检查点"""
    # 生成模型文件夹名称
    folder_name = generate_model_folder_name(args)
    output_dir = Path(args.output_dir) / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 构建检查点路径
    epoch_name = str(epoch)
    checkpoint_path = output_dir / (f"checkpoint-{epoch_name}.pth")

    print('保存模型: ' + model.best_model_state_dict[0])

    # 准备保存数据
    to_save = {
        'trainable_params': model.best_model_state_dict,
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'args': args,
        'early_stopper': early_stopper.get_state(),
        'metric_logger': metric_logger.get_state_dict(),
        'val_metric_logger': val_metric_logger.get_state_dict(),
    }

    save_on_master(to_save, str(checkpoint_path))
    
    print(f"已保存可训练参数")


def generate_model_folder_name(args):
    """根据微调配置生成文件夹名称"""
    components = []
    
    # LoRA配置
    if args.lora_layers != '0-0':
        lora_info = f"LoRA_{args.lora_layers}_r{args.lora_rank}_a{args.lora_alpha}"
        if args.lora_targets != 'Q,V':
            lora_info += f"_{args.lora_targets}"
        components.append(lora_info)
    
    # Prompt Tuning配置
    if args.prompt_layers != '0-0':
        prompt_info = f"Prompt_{args.prompt_layers}_len{args.prompt_len}"
        components.append(prompt_info)
    
    # Parallel Adapter配置
    if args.p_adapter_layers != '0-0':
        adapter_info = f"PAdapter_{args.p_adapter_layers}_size{args.p_adapter_size}"
        if args.p_adapter_hydra:
            adapter_info += "_hydra"
        components.append(adapter_info)
    
    # 专家配置
    if args.expert_num > 1:
        expert_info = f"MoE_expert{args.expert_num}"
        if args.hydra_moe:
            expert_info += "_hydra"
        if args.expert_weight:
            expert_info += "_weighted"
        components.append(expert_info)
    
    # 偏置微调配置
    if args.w_bias:
        components.append("bias_tune")
    
    # SwiGLU路由配置
    if args.swi_x > 0:
        components.append(f"swi_x{args.swi_x}")
    
    # 学习率配置
    if hasattr(args, 'lr') and args.lr is not None:
        lr_str = f"lr{args.lr:.0e}".replace('e-0', 'e-').replace('e+0', 'e')
        components.append(lr_str)
    
    # 批次大小配置
    if hasattr(args, 'batch_size'):
        components.append(f"bs{args.batch_size}")
    
    # 数据集名称（从路径中提取倒数第二层目录名）
    if hasattr(args, 'data_path') and args.data_path:
        # 获取倒数第二层目录名
        dataset_dir = os.path.dirname(args.data_path)
        dataset_name = os.path.basename(dataset_dir)
        if dataset_name:
            components.append(dataset_name)
    
    # 随机种子配置
    if hasattr(args, 'seed') and args.seed is not None:
        components.append(f"seed{args.seed}")
    
    # 组合文件夹名称
    if components:
        folder_name = "_".join(components)
    else:
        folder_name = "full_finetune"
    
    return folder_name
    

def all_reduce_mean(x):
    """计算所有进程的平均值"""
    world_size = get_world_size()
    
    if world_size > 1:
        x_reduce = jt.array(x)
        jt.mpi.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    """为模型参数添加权重衰减分组"""
    decay = []
    no_decay = []
    
    # 遍历模型参数
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # 偏置项和一维参数不进行权重衰减
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}
    ]


def find_latest_checkpoint(output_dir, folder_name=None):
    """查找最新的检查点文件"""
    output_path = Path(output_dir)
    
    # 根据是否指定文件夹名查找检查点
    if folder_name:
        folder_path = output_path / folder_name
        if not folder_path.exists():
            return None
        checkpoints = list(folder_path.glob("checkpoint-*.pth"))
    else:
        checkpoints = []
        for folder in output_path.iterdir():
            if folder.is_dir():
                checkpoints.extend(folder.glob("checkpoint-*.pth"))
    
    if not checkpoints:
        return None
    
    # 返回最新的检查点
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.stem.split('-')[1]))
    return str(latest_checkpoint)

def get_first_notpid(tokens: jt.Var):
    """获取序列中最后一个非-1的位置索引"""
    assert len(tokens.shape) == 2, f'tokens.shape: {tokens.shape}'

    # 假设 tokens.shape = [B, L]
    B, L = tokens.shape

    # 构造mask，True表示该位置不是-1
    mask = tokens != -1               # shape [B, L]

    # 把序列在第1维（长度维）上反转
    rev_mask = mask[:, ::-1]          # shape [B, L]，第0维不动，第1维反转

    # 找出反转后第一个True的位置（就是原序列从右向左的第一个非-1）
    # argmax会返回第一个最大值（True=1, False=0）的位置索引
    pos_from_right, _ = jt.argmax(rev_mask, dim=1)  # 取元组的第二个元素，shape [B]

    # 把"从右边数来"的索引，映射回原来的坐标
    last_valid_idx = (L - 1) - pos_from_right    # shape [B]

    return last_valid_idx