import os
import glob
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import seaborn as sns


def smooth_data(values, smooth_factor=0.1):
    """使用指数移动平均对数据进行平滑处理"""
    if smooth_factor <= 0:
        return values
    
    # 指数移动平均平滑
    smoothed = [values[0]]
    for i in range(1, len(values)):
        smoothed.append(smooth_factor * values[i] + (1 - smooth_factor) * smoothed[i-1])
    
    return smoothed


def plot_event_scalars(event_file: str, out_base_dir: str, smooth_factor: float = 0.3, 
                      line_width: float = 1.5, marker_size: float = 3.0, 
                      show_markers: bool = True, marker_interval: int = 10):
    """从TensorBoard事件文件提取标量数据并生成可视化图表"""
    
    # 创建输出目录
    base = os.path.basename(event_file)
    out_dir = os.path.join(out_base_dir, base)
    os.makedirs(out_dir, exist_ok=True)

    # 初始化TensorBoard事件累加器
    ea = event_accumulator.EventAccumulator(
        event_file,
        size_guidance={
            event_accumulator.SCALARS: 0,
            # 禁用其他数据类型以节省内存
            event_accumulator.HISTOGRAMS: 0,
            event_accumulator.IMAGES: 0,
            event_accumulator.COMPRESSED_HISTOGRAMS: 0,
            event_accumulator.AUDIO: 0,
            event_accumulator.GRAPH: 0,
            event_accumulator.META_GRAPH: 0,
        }
    )
    ea.Reload()

    # 设置matplotlib全局样式
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "lines.linewidth": line_width,
        "lines.markersize": marker_size,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
    })

    # 处理每个标量标签
    for tag in ea.Tags().get('scalars', []):
        # 提取并排序事件数据
        events = ea.Scalars(tag)
        events = sorted(events, key=lambda e: e.step)
        steps = [e.step for e in events]
        values = [e.value for e in events]

        # 应用数据平滑
        if smooth_factor > 0:
            values = smooth_data(values, smooth_factor)

        # 创建图表
        fig, ax = plt.subplots(figsize=(8, 5))

        # 绘制主曲线
        line_plot = ax.plot(steps, values, color='#2E86AB', alpha=0.8, 
                           linewidth=line_width, label='Training Curve')

        # 选择性添加数据点标记
        if show_markers and len(steps) > marker_interval:
            marker_steps = steps[::marker_interval]
            marker_values = values[::marker_interval]
            ax.scatter(marker_steps, marker_values, color='#A23B72', 
                      s=marker_size*2, alpha=0.7, zorder=5, label='Data Points')

        # 设置标题和轴标签
        ax.set_title(tag.replace('_', ' ').title(), pad=15, fontweight='bold')
        ax.set_xlabel("Training Steps", fontweight='bold')
        ax.set_ylabel(tag.replace('_', ' ').title(), fontweight='bold')

        # 美化网格线
        ax.grid(True, linestyle='--', linewidth=0.3, alpha=0.6, color='#CCCCCC')

        # 移除顶部和右侧边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#666666')
        ax.spines['bottom'].set_color('#666666')

        # 添加图例（当显示数据点时）
        if show_markers and len(steps) > marker_interval:
            ax.legend(loc='upper right', framealpha=0.9)

        # 优化布局并保存
        fig.tight_layout(pad=2.0)
        safe_tag = tag.replace('/', '_')
        out_path = os.path.join(out_dir, f"{safe_tag}.png")
        fig.savefig(out_path, bbox_inches='tight', dpi=150)
        plt.close(fig)

    print(f"[+] 已保存 {len(ea.Tags().get('scalars', []))} 个图表到 {out_dir}")


def main(logdir: str, out_base_dir: str, smooth_factor: float = 0.3, 
         line_width: float = 1.5, marker_size: float = 3.0, 
         show_markers: bool = True, marker_interval: int = 10):
    """主函数：处理TensorBoard日志目录中的所有事件文件"""
    
    # 查找所有事件文件
    files = glob.glob(os.path.join(logdir, '**', 'events.*'), recursive=True)
    if not files:
        print("未找到事件文件:", logdir)
        return
    
    # 显示处理信息
    print(f"找到 {len(files)} 个事件文件")
    print(f"绘图设置: smooth_factor={smooth_factor}, line_width={line_width}, "
          f"marker_size={marker_size}, show_markers={show_markers}")
    
    # 处理每个事件文件
    for f in files:
        print(f"正在处理: {f}")
        plot_event_scalars(f, out_base_dir, smooth_factor, line_width, 
                          marker_size, show_markers, marker_interval)


if __name__ == "__main__":
    import argparse
    
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="TensorBoard日志可视化工具")
    parser.add_argument('--logdir', type=str, required=True,
                        help="TensorBoard 日志根目录")
    parser.add_argument('--out_base', type=str, default="tb_plots",
                        help="所有输出子目录的父目录")
    parser.add_argument('--smooth_factor', type=float, default=0.3,
                        help="平滑因子 (0-1)，0为不平滑，1为完全平滑")
    parser.add_argument('--line_width', type=float, default=1.5,
                        help="折线宽度")
    parser.add_argument('--marker_size', type=float, default=3.0,
                        help="数据点大小")
    parser.add_argument('--show_markers', action='store_true', default=True,
                        help="是否显示数据点")
    parser.add_argument('--marker_interval', type=int, default=10,
                        help="数据点显示间隔（每隔多少个点显示一个）")
    parser.add_argument('--no_markers', action='store_true',
                        help="不显示数据点")
    
    args = parser.parse_args()
    
    # 处理参数冲突
    if args.no_markers:
        args.show_markers = False

    # 执行主函数
    main(args.logdir, args.out_base, args.smooth_factor, args.line_width,
         args.marker_size, args.show_markers, args.marker_interval)
