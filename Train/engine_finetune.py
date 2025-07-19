import math
import sys
import jittor as jt
from jittor.dataset import Dataset
from argparse import Namespace

from Utils.misc import MetricLogger, SmoothedValue
from Utils.lr_sched import adjust_learning_rate
from Utils.EarlyStopper import EarlyStopper
from Models.LLaMA_Adapter import LLaMA_adapter

import os
import json


jt.flags.log_silent = 1


# def train_one_epoch(model: LLaMA_adapter,
#                     data_loader: Dataset, optimizer: jt.optim.Optimizer,
#                     epoch: int,
#                     args: Namespace,
#                     log_writer=None):
#     """训练一个完整的轮次"""
    
#     # 初始化指标记录器
#     metric_logger = MetricLogger(delimiter="  ")
#     metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
#     header = 'Epoch: [{}]'.format(epoch)
#     print_freq = 10

#     # 梯度累积设置
#     accum_iter = args.accum_iter
#     optimizer.zero_grad()

#     # 设置日志记录器
#     # if log_writer is not None:
#     #     print('log_dir: {}'.format(args.log_dir))
    
#     # 训练循环
#     for data_iter_step, (examples, labels, prompt_mask) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
#         # 学习率调度
#         if data_iter_step % accum_iter == 0:
#             adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        
#         # 前向传播和损失计算
#         with jt.flag_scope(amp_level=1):
#             c_loss, m_loss = model(examples, labels, prompt_mask)
        
#         # 损失组合
#         loss = c_loss + m_loss * 0   # 只使用分类损失
#         loss_value = loss
#         c_loss_value = c_loss
#         m_loss_value = m_loss
        
#         # 梯度累积处理
#         loss /= accum_iter

#         # 反向传播
#         optimizer.backward(loss)

#         # 更新权重
#         if (data_iter_step + 1) % accum_iter == 0:
#             optimizer.step()
#             optimizer.zero_grad()
#             model.float16()

#         # GPU同步
#         jt.sync_all(True)

#         # 更新指标
#         metric_logger.update(closs=c_loss_value)
#         metric_logger.update(mloss=m_loss_value)

#         # 记录学习率
#         lr = optimizer.param_groups[0]["lr"]
#         metric_logger.update(lr=lr)

#         # 分布式训练指标同步
#         if jt.in_mpi:
#             loss_value_reduce = loss_value.mpi_all_reduce("mean")
#             c_loss_value_reduce = c_loss_value.mpi_all_reduce("mean")
#             m_loss_value_reduce = m_loss_value.mpi_all_reduce("mean")
#         else:
#             loss_value_reduce = loss_value
#             c_loss_value_reduce = c_loss_value
#             m_loss_value_reduce = m_loss_value
        
#         # TensorBoard日志记录
#         if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
#             epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
#             log_writer.add_scalar('c_train_loss', float(c_loss_value_reduce), epoch_1000x)
#             log_writer.add_scalar('m_train_loss', float(m_loss_value_reduce), epoch_1000x)
#             log_writer.add_scalar('lr', float(lr), epoch_1000x)

#     # 指标同步和统计
#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger)
    
#     # 返回结果
#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch(model: LLaMA_adapter,
                    data_loader: Dataset,  # 训练数据加载器
                    val_loader: Dataset,  # 新增: 验证数据加载器
                    optimizer: jt.optim.Optimizer,
                    epoch: int,
                    args: Namespace,
                    log_writer=None,
                    early_stopper: EarlyStopper = None,  # 新增: 早停对象
                    val_interval: int = 100  # 新增: 验证间隔（每隔多少batch验证一次，默认100）
                    ):
    """训练一个完整的轮次，并每隔固定batch进行验证和早停检查"""
    
    # 初始化指标记录器
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('closs', SmoothedValue(window_size=5, fmt='{median:.6f}'))
    metric_logger.add_meter('mloss', SmoothedValue(window_size=5, fmt='{median:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    # 梯度累积设置
    accum_iter = args.accum_iter
    optimizer.zero_grad()

    # 初始化早停标志和验证统计
    early_stopped = False
    val_stats = {}  # 如果没有验证，则返回空dict

    # 设置日志记录器
    # if log_writer is not None:
    #     print('log_dir: {}'.format(args.log_dir))
    
    # 训练循环
    for data_iter_step, (examples, labels, prompt_mask) in enumerate(metric_logger.log_every(data_loader, print_freq, if_print=True, header=header)):
        # 学习率调度
        if data_iter_step % accum_iter == 0:
            adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        
        # 前向传播和损失计算
        with jt.flag_scope(amp_level=1):
            c_loss, m_loss = model(examples, labels, prompt_mask)
        
        # 损失组合
        loss = c_loss + m_loss * 0   # 只使用分类损失
        
        # 梯度累积处理
        loss /= accum_iter

        # 反向传播
        optimizer.backward(loss)

        # 更新权重
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.step()
            optimizer.zero_grad()
            model.float16()

        # GPU同步
        jt.sync_all(True)

        # 更新指标
        metric_logger.update(closs=c_loss)
        metric_logger.update(mloss=m_loss)

        # 记录学习率
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        # 分布式训练指标同步
        if jt.in_mpi:
            loss_value_reduce = loss.mpi_all_reduce("mean")
            c_loss_value_reduce = c_loss.mpi_all_reduce("mean")
            m_loss_value_reduce = m_loss.mpi_all_reduce("mean")
        else:
            loss_value_reduce = loss
            c_loss_value_reduce = c_loss
            m_loss_value_reduce = m_loss
        
        # TensorBoard日志记录
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('c_train_loss', float(c_loss_value_reduce), epoch_1000x)
            log_writer.add_scalar('m_train_loss', float(m_loss_value_reduce), epoch_1000x)
            log_writer.add_scalar('lr', float(lr), epoch_1000x)

        # 新增: 每隔val_interval个batch进行验证
        if val_loader is not None and (data_iter_step + 1) % val_interval == 0:
            val_loss = 0

            with jt.no_grad():
                for val_examples, val_labels, val_prompt_mask in val_loader:
                    with jt.flag_scope(amp_level=1):
                        val_c_loss, val_m_loss = model(val_examples, val_labels, val_prompt_mask)
                    val_loss += val_c_loss + val_m_loss * 0  # 与训练一致，只使用分类损失
            
            val_loss = val_loss / len(val_loader)

            # 日志记录验证损失
            if log_writer is not None:
                step_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
                log_writer.add_scalar('val_loss', float(val_loss), step_1000x)

            val_stats = {'epoch': epoch, 'data_iter_step': data_iter_step, 'val_loss': val_loss.item()}

            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(val_stats) + "\n")
            
            # 新增: 早停检查（如果提供了 early_stopper）
            if early_stopper is not None:
                if early_stopper(model, val_loss, epoch, data_iter_step):
                    print(f"Early stopping triggered at step {data_iter_step + 1} in epoch {epoch}")
                    early_stopped = True
                    # 由于在epoch内触发早停，直接break训练循环
                    break

    # 指标同步和统计
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    if val_loader is None:
        return train_stats
    else:
        return train_stats, val_stats, early_stopped
