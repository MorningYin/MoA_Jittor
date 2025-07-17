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

import jittor as jt


class SmoothedValue(object):
    """
    平滑值跟踪器
    
    该类用于跟踪一系列数值，并提供对滑动窗口或全局序列平均值的平滑访问。
    主要用于训练过程中监控损失、准确率等指标的平滑统计。
    
    功能特点：
    1. 维护一个固定大小的滑动窗口，存储最近的值
    2. 提供多种统计信息：中位数、平均值、全局平均值、最大值、当前值
    3. 支持多进程间的数据同步
    4. 支持自定义格式化输出
    """

    def __init__(self, window_size=20, fmt=None):
        """
        初始化平滑值跟踪器
        
        Args:
            window_size (int): 滑动窗口大小，默认为20
            fmt (str): 自定义格式化字符串，默认为显示中位数和全局平均值
        """
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"  # 默认格式：中位数 (全局平均值)
        self.deque = deque(maxlen=window_size)  # 使用双端队列实现滑动窗口
        self.total = 0.0  # 累计总和，用于计算全局平均值
        self.count = 0     # 累计计数，用于计算全局平均值
        self.fmt = fmt     # 格式化字符串

    def update(self, value, n=1):
        """
        更新跟踪器的值
        
        Args:
            value (float): 要添加的新值
            n (int): 该值对应的样本数量，默认为1
        """
        self.deque.append(value)  # 将新值添加到滑动窗口
        self.count += n           # 更新累计计数
        self.total += value * n   # 更新累计总和

    def synchronize_between_processes(self):
        """
        在多进程间同步统计数据
        
        注意：此方法只同步全局统计信息（count和total），不同步滑动窗口（deque）！
        这是因为滑动窗口在不同进程间可能不同步，但全局平均值需要保持一致。
        """
        if jt.in_mpi == None:
            return  # 没有初始化分布式

        t = jt.array([self.count, self.total], dtype=jt.float64)
        jt.mpi.mpi_reduce(t, op="sum")
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        """
        计算滑动窗口中值的中位数
        
        Returns:
            float: 滑动窗口中值的中位数
        """
        d = jt.array(list(self.deque))
        return float(jt.median(d))

    @property
    def avg(self):
        """
        计算滑动窗口中值的平均值
        
        Returns:
            float: 滑动窗口中值的平均值
        """
        d = jt.array(list(self.deque), dtype=jt.float16)
        return float(jt.mean(d))
    
    @property
    def global_avg(self):
        """
        计算全局平均值（基于所有历史数据）
        
        Returns:
            float: 全局平均值
        """
        return self.total / self.count

    @property
    def max(self):
        """
        获取滑动窗口中的最大值
        
        Returns:
            float: 滑动窗口中的最大值
        """
        return max(self.deque)

    @property
    def value(self):
        """
        获取最新的值（滑动窗口中的最后一个值）
        
        Returns:
            float: 最新的值
        """
        return self.deque[-1]

    def __str__(self):
        """
        返回格式化的字符串表示
        
        使用初始化时指定的格式字符串，显示各种统计信息
        
        Returns:
            str: 格式化的统计信息字符串
        """
        return self.fmt.format(
            median=self.median,      # 中位数
            avg=self.avg,            # 滑动窗口平均值
            global_avg=self.global_avg,  # 全局平均值
            max=self.max,            # 最大值
            value=self.value)        # 当前值


class MetricLogger(object):
    """
    指标记录器
    
    该类用于管理训练过程中的多个指标，提供统一的更新、显示和同步功能。
    主要特点：
    1. 管理多个 SmoothedValue 实例，每个对应一个指标
    2. 支持自动类型转换（Tensor转float）
    3. 提供迭代器包装，自动记录训练进度
    4. 支持多进程间的指标同步
    5. 提供ETA估算和内存监控
    """

    def __init__(self, delimiter="\t"):
        """
        初始化指标记录器
        
        Args:
            delimiter (str): 指标之间的分隔符，默认为制表符
        """
        self.meters = defaultdict(SmoothedValue)  # 使用默认字典存储多个指标
        self.delimiter = delimiter  # 分隔符

    def update(self, **kwargs):
        """
        更新多个指标的值
        
        Args:
            **kwargs: 键值对，键为指标名称，值为指标值
        """
        for k, v in kwargs.items():
            if v is None:
                continue  # 跳过None值
            if isinstance(v, jt.Var):
                v = float(v)
            assert isinstance(v, (float, int))  # 确保值为数值类型
            self.meters[k].update(v)  # 更新对应的指标

    def __getattr__(self, attr):
        """
        动态属性访问，允许直接访问指标
        
        例如：logger.loss 等价于 logger.meters['loss']
        
        Args:
            attr (str): 属性名称
            
        Returns:
            SmoothedValue: 对应的指标对象
            
        Raises:
            AttributeError: 如果属性不存在
        """
        if attr in self.meters:
            return self.meters[attr]  # 返回指标对象
        if attr in self.__dict__:
            return self.__dict__[attr]  # 返回实例属性
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        """
        返回所有指标的字符串表示
        
        Returns:
            str: 格式化的指标字符串
        """
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))  # 指标名: 指标值
            )
        return self.delimiter.join(loss_str)  # 用分隔符连接

    def synchronize_between_processes(self):
        """
        同步所有指标在多进程间的数据
        
        调用每个 SmoothedValue 的同步方法
        """
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        """
        添加新的指标
        
        Args:
            name (str): 指标名称
            meter (SmoothedValue): 指标对象
        """
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        """
        迭代器包装器，自动记录训练进度
        
        这是一个生成器函数，包装任何可迭代对象，在迭代过程中自动记录：
        - 进度信息（当前步数/总步数）
        - 指标值（损失、准确率等）
        - 时间信息（迭代时间、数据加载时间）
        - ETA估算（预计剩余时间）
        - 内存使用情况（如果使用GPU）
        
        Args:
            iterable: 要包装的可迭代对象（通常是数据加载器）
            print_freq (int): 打印频率，每隔多少步打印一次
            header (str): 日志头部信息
            
        Yields:
            原始迭代器的元素
        """
        i = 0
        if not header:
            header = ''
        
        # 记录开始时间和结束时间
        start_time = time.time()
        end = time.time()
        
        # 创建时间跟踪器
        iter_time = SmoothedValue(fmt='{avg:.4f}')  # 迭代时间
        data_time = SmoothedValue(fmt='{avg:.4f}')  # 数据加载时间
        
        # 构建日志消息格式
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'  # 进度数字的格式
        log_msg = [
            header,  # 头部信息
            '[{0' + space_fmt + '}/{1}]',  # 进度格式 [当前/总数]
            'eta: {eta}',  # 预计剩余时间
            '{meters}',  # 指标信息
            'time: {time}',  # 迭代时间
            'data: {data}'  # 数据加载时间
        ]
        
        # 如果使用GPU，添加内存监控（Jittor cuda）
        if jt.has_cuda:
            log_msg.append('max mem: {memory:.0f}')  # 最大内存使用
        
        log_msg = self.delimiter.join(log_msg)  # 用分隔符连接
        MB = 1024.0 * 1024.0  # 字节到MB的转换因子
        
        # 遍历迭代器
        for obj in iterable:
            data_time.update(time.time() - end)  # 更新数据加载时间
            yield obj  # 返回原始元素
            iter_time.update(time.time() - end)  # 更新迭代时间
            
            # 按频率打印日志
            if i % print_freq == 0 or i == len(iterable) - 1:
                # 计算ETA（预计剩余时间）
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                
                # 根据是否使用GPU选择不同的打印格式
                if jt.has_cuda:
                    try:
                        mem_mb = jt.cuda.memory_info()[1] / MB  # used bytes → MB
                    except Exception:
                        mem_mb = 0
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=mem_mb))
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
    """
    设置分布式训练中的打印控制
    
    该函数重写内置的print函数，只在主进程中打印信息，避免多进程输出混乱。
    同时添加时间戳，便于调试和日志分析。
    
    Args:
        is_master (bool): 是否为主进程
    """
    builtin_print = builtins.print  # 保存原始的print函数

    def print(*args, **kwargs):
        """
        重写的print函数，只在主进程中打印
        
        Args:
            *args: 要打印的参数
            **kwargs: 关键字参数，支持force参数强制打印
        """
        force = kwargs.pop('force', False)  # 获取force参数，默认为False
        force = force or (get_world_size() > 8)  # 如果进程数大于8，强制打印
        if is_master or force:  # 只在主进程或强制模式下打印
            now = datetime.datetime.now().time()  # 获取当前时间
            builtin_print('[{}] '.format(now), end='')  # 打印时间戳
            builtin_print(*args, **kwargs)  # 打印原始内容

    builtins.print = print  # 替换内置print函数


def is_dist_avail_and_initialized():
    """Jittor：检测是否处于 MPI 分布式环境并已初始化通信器。"""
    return jt.in_mpi  # Jittor 会在首次调用分布式 API 时自动初始化


def get_world_size():
    """返回总进程数；单进程时为 1。"""
    if not jt.in_mpi:
        return 1
    return jt.world_size


def get_rank():
    """
    获取当前进程的排名
    
    Returns:
        int: 当前进程排名，如果未使用分布式训练则返回0
    """
    # Jittor: 若未使用 MPI，视为单进程
    if not jt.in_mpi:
        return 0
    return jt.rank


def is_main_process():
    """
    判断当前进程是否为主进程
    
    Returns:
        bool: 如果当前进程排名为0则返回True，否则返回False
    """
    return get_rank() == 0  # 排名为0的进程为主进程


def save_on_master(*args, **kwargs):
    """
    只在主进程中保存文件
    
    该函数包装了torch.save，确保只在主进程中执行保存操作，
    避免多进程同时写入文件导致的冲突。
    
    Args:
        *args: 传递给torch.save的位置参数
        **kwargs: 传递给torch.save的关键字参数
    """
    if is_main_process():  # 只在主进程中执行
        jt.save(*args, **kwargs)  # Jittor 保存模型/状态


def init_distributed_mode(args):
    """Jittor 版分布式初始化，兼容两种启动方式（MPI/单进程）。"""

    # ---------- 0. 单进程默认 ----------
    args.distributed = False
    args.rank = 0
    args.world_size = 1
    args.gpu = 0

    # ========== MPI/Distributed 初始化 ==========
    if jt.in_mpi:
        args.rank       = jt.rank
        args.world_size = jt.world_size
        # 用 rank 去做 round‐robin 分配 GPU
        ngpus = jt.has_cuda and jt.get_device_count() or 1
        gpu_id = args.rank % ngpus
        args.gpu        = gpu_id
        args.distributed = args.world_size > 1

    # ========== GPU 设置 ==========
    if jt.has_cuda:
        # 限制这个进程只能“看到”它该用的那块卡
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        # 打开全局 GPU 开关
        jt.flags.use_cuda = 1

    # ---------- 打印 / 同步 ----------
    if args.distributed:
        setup_for_distributed(args.rank == 0)

    if args.distributed and args.rank == 0:
        print(f"[Jittor] Distributed init done | rank {args.rank}/{args.world_size} | gpu {args.gpu} | nproc {jt.get_nproc()}")
    elif not args.distributed:
        print("[Jittor] Running in single-process mode.")


def save_model(args, epoch, model, optimizer):
    """
    保存模型检查点
    
    该函数用于保存训练过程中的模型检查点，包含完整的训练状态。
    支持两种保存模式：
    1. 使用loss_scaler的混合精度训练模式
    2. 不使用loss_scaler的普通训练模式
    
    Args:
        args: 训练参数对象
        epoch (int): 当前训练轮次
        model: 模型对象（可能包含DDP包装）
        model_without_ddp: 不带DDP包装的模型对象
        optimizer: 优化器对象
        loss_scaler: 损失缩放器对象，如果为None则使用普通保存模式
    """
    # ==================== 路径设置 ====================
    output_dir = Path(args.output_dir)  # 输出目录
    epoch_name = str(epoch)  # 轮次名称
    
    # ==================== 统一保存（不区分 loss_scaler） ====================
    # 构建检查点文件路径
    checkpoint_path = output_dir / (f"checkpoint-{epoch_name}.pth")

    # 统一构建要保存的状态字典
    to_save = {
        'model': model.state_dict(),          # 模型权重
        'optimizer': optimizer.state_dict(),  # 优化器状态
        'epoch': epoch,                       # 轮次编号
        'args': args,                         # 训练参数
    }

    # 保存检查点（仅主进程执行）
    save_on_master(to_save, str(checkpoint_path))


def all_reduce_mean(x):
    """
    计算所有进程的平均值
    
    该函数用于在分布式训练中计算所有进程的平均值。
    如果只有一个进程，直接返回原值；如果有多个进程，则计算所有进程的平均值。
    
    Args:
        x: 要计算平均值的数值
        
    Returns:
        所有进程的平均值
    """
    # ==================== 获取进程数量 ====================
    world_size = get_world_size()  # 获取总进程数
    
    # ==================== 分布式计算平均值 ====================
    if world_size > 1:
        # 多进程模式：计算所有进程的平均值
        x_reduce = jt.array(x)             # 将数值转换为CUDA张量
        jt.mpi.all_reduce(x_reduce)        # 对所有进程的值进行求和
        x_reduce /= world_size             # 除以进程数得到平均值
        return x_reduce.item()             # 返回标量值
    else:
        # 单进程模式：直接返回原值
        return x


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    """
    为模型参数添加权重衰减分组
    
    该函数将模型参数分为两组：
    1. 需要权重衰减的参数（如权重矩阵）
    2. 不需要权重衰减的参数（如偏置项、一维参数等）
    
    这种分组策略是深度学习中常用的优化技巧，可以提高训练稳定性和模型性能。
    
    Args:
        model: 要分组的模型
        weight_decay (float): 权重衰减系数，默认为1e-5
        skip_list (tuple): 要跳过权重衰减的参数名列表
        
    Returns:
        list: 包含两个参数组的列表，每个组包含参数和对应的权重衰减设置
    """
    # ==================== 参数分组初始化 ====================
    decay = []      # 需要权重衰减的参数组
    no_decay = []   # 不需要权重衰减的参数组
    
    # ==================== 遍历模型参数 ====================
    for name, param in model.named_parameters():
        # 跳过不需要梯度的参数（冻结权重）
        if not param.requires_grad:
            continue  # frozen weights
        
        # ==================== 判断是否需要权重衰减 ====================
        # 以下参数不需要权重衰减：
        # 1. 一维参数（如LayerNorm的scale和bias）
        # 2. 偏置项（以.bias结尾的参数）
        # 3. 在skip_list中的参数
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)  # 添加到无衰减组
        else:
            decay.append(param)     # 添加到衰减组
    
    # ==================== 返回参数组配置 ====================
    return [
        {'params': no_decay, 'weight_decay': 0.},           # 无衰减组
        {'params': decay, 'weight_decay': weight_decay}      # 衰减组
    ]


def download(url: str, root: str):
    """
    下载文件到指定目录
    
    该函数用于从URL下载文件到本地目录，支持进度条显示和文件存在性检查。
    如果文件已存在，则直接返回文件路径；如果不存在，则下载文件。
    
    Args:
        url (str): 要下载的文件URL
        root (str): 本地保存目录
        
    Returns:
        str: 下载文件的完整路径
        
    Raises:
        RuntimeError: 当目标路径存在但不是普通文件时抛出异常
    """
    # ==================== 目录创建 ====================
    os.makedirs(root, exist_ok=True)  # 创建目录（如果不存在）
    
    # ==================== 文件路径设置 ====================
    filename = os.path.basename(url)  # 从URL提取文件名
    download_target = os.path.join(root, filename)  # 构建完整文件路径

    # ==================== 文件存在性检查 ====================
    # 检查目标路径是否存在但不是普通文件（如目录）
    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    # 如果文件已存在，直接返回路径
    if os.path.isfile(download_target):
        return download_target

    # ==================== 文件下载 ====================
    # 使用上下文管理器同时处理网络连接和文件写入
    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        # 创建进度条显示下载进度
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            # 分块读取和写入文件
            while True:
                buffer = source.read(8192)  # 每次读取8KB数据
                if not buffer:
                    break  # 读取完毕，退出循环
                output.write(buffer)        # 写入文件
                loop.update(len(buffer))    # 更新进度条

    # ==================== 返回文件路径 ====================
    return download_target