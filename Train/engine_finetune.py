import math
import sys
import jittor as jt
from jittor.dataset import Dataset

from Utils.misc import MetricLogger, SmoothedValue
from Utils.lr_sched import adjust_learning_rate

from Models.LLaMA_Adapter import LLaMA_adapter
# args 是通过 argparse 解析得到的训练参数集合
from argparse import Namespace

def train_one_epoch(model: LLaMA_adapter,
                    data_loader: Dataset, optimizer: jt.optim.Optimizer,
                    epoch: int,
                    args: Namespace,
                    log_writer=None):
    """
    训练一个完整的轮次
    
    该函数执行一个完整的训练轮次，包括：
    1. 数据加载和预处理
    2. 前向传播和损失计算
    3. 反向传播和梯度更新
    4. 学习率调度
    5. 指标记录和日志记录
    6. 分布式训练同步
    
    Args:
        model (LLaMA_adapter): 要训练的模型
        data_loader (Iterable): 数据加载器
        optimizer (jt.optim.Optimizer): 优化器
        device (jt.device): 计算设备
        epoch (int): 当前训练轮次
        loss_scaler: 损失缩放器（用于混合精度训练）
        log_writer: TensorBoard日志记录器，默认为None
        args: 训练参数对象，默认为None
        
    Returns:
        dict: 包含所有指标全局平均值的字典
    """
    # ==================== 模型训练模式设置 ====================
    # model.train(True)  # 设置为训练模式
    # model.module.set_default_trainability()  # 注释掉的默认可训练性设置

    # ==================== 指标记录器初始化 ====================
    metric_logger = MetricLogger(delimiter="  ")  # 创建指标记录器，使用双空格分隔
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))  # 添加学习率指标
    header = 'Epoch: [{}]'.format(epoch)  # 设置日志头部信息
    print_freq = 10  # 打印频率：每10步打印一次

    # ==================== 梯度累积设置 ====================
    accum_iter = args.accum_iter  # 获取梯度累积步数

    # ==================== 优化器初始化 ====================
    optimizer.zero_grad()  # 清空梯度

    # ==================== 日志记录器设置 ====================
    if log_writer is not None:
        print('log_dir: {}'.format(args.log_dir))  # 打印日志目录
    
    # ==================== 训练循环 ====================
    # 遍历数据加载器，使用metric_logger包装以自动记录进度
    # for data_iter_step, (examples, labels, example_mask, imgs) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
    for data_iter_step, (examples, labels, prompt_mask) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # ==================== 学习率调度 ====================
        # 使用每步（而不是每轮次）的学习率调度器
        if data_iter_step % accum_iter == 0:
            # 计算当前进度：data_iter_step / len(data_loader) + epoch
            adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        
        
        # ==================== 前向传播和损失计算 ====================
        # Jittor 使用 flag_scope 控制 AMP
        with jt.flag_scope(amp_level=1):
            # 模型前向传播，返回分类损失和掩码损失
            c_loss, m_loss = model(examples, labels, prompt_mask)
        
        # ==================== 损失组合 ====================
        loss = c_loss + m_loss * 0  # 当前只使用分类损失，掩码损失权重为0
        loss_value = loss.item()     # 获取损失值
        c_loss_value = c_loss.item() # 分类损失值
        m_loss_value = m_loss        # 掩码损失值
        
        # ==================== 损失有效性检查 ====================
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)  # 如果损失为无穷大或NaN，停止训练

        # ==================== 梯度累积处理 ====================
        loss /= accum_iter  # 将损失除以累积步数

        # ==================== 反向传播 ====================
        optimizer.backward(loss)  # 累积梯度

        # ==================== 更新权重 ====================
        if (data_iter_step + 1) % accum_iter == 0:
            # 可选：梯度裁剪，如有需要可启用
            # optimizer.clip_grad_norm(max_norm=1.0)
            optimizer.step()      # 参数更新
            optimizer.zero_grad() # 清零梯度，为下次累积做准备

        # ==================== GPU同步 ====================
        jt.sync_all(True)  # type: ignore[attr-defined]  # 确保GPU操作完成

        # ==================== 指标更新 ====================
        metric_logger.update(closs=c_loss_value)  # 更新分类损失指标
        metric_logger.update(mloss=m_loss_value)  # 更新掩码损失指标

        # ==================== 学习率记录 ====================
        lr = optimizer.param_groups[0]["lr"]  # 获取当前学习率
        metric_logger.update(lr=lr)  # 更新学习率指标

        # ==================== 分布式训练指标同步 ====================
        # 计算所有进程的平均损失值（若处于 MPI 环境）
        if jt.in_mpi:  # type: ignore[attr-defined]
            loss_value_reduce = loss_value.mpi_all_reduce("mean")  # type: ignore[attr-defined]
            c_loss_value_reduce = c_loss_value.mpi_all_reduce("mean")  # type: ignore[attr-defined]
            m_loss_value_reduce = m_loss_value.mpi_all_reduce("mean")  # type: ignore[attr-defined]
        else:
            loss_value_reduce = loss_value
            c_loss_value_reduce = c_loss_value
            m_loss_value_reduce = m_loss_value
        
        # ==================== TensorBoard日志记录 ====================
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ 
            我们使用epoch_1000x作为tensorboard的x轴。
            这在校准不同批次大小变化的曲线时很有用。
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('c_train_loss', float(c_loss_value_reduce), epoch_1000x)  # 记录分类训练损失
            log_writer.add_scalar('m_train_loss', float(m_loss_value_reduce), epoch_1000x)  # 记录掩码训练损失
            log_writer.add_scalar('lr', float(lr), epoch_1000x)  # 记录学习率

    # ==================== 指标同步和统计 ====================
    # 收集所有进程的统计信息
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)  # 打印平均统计信息
    
    # ==================== 返回结果 ====================
    # 返回所有指标的全局平均值
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
