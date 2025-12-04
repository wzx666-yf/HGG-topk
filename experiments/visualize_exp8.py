# -*- coding: utf-8 -*-
"""
实验8可视化: 不同梯度量下的稀疏化时间对比
在同一图中展示所有压缩算法随梯度量增加的时间变化
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# 设置科研风格
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['lines.markersize'] = 8
sns.set_style("whitegrid")

# 压缩算法配色和线型（科研论文常用）
STYLE_CONFIG = {
    'TopK': {
        'color': '#E74C3C',
        'marker': 'o',
        'linestyle': '-',
        'label': 'TopK'
    },
    'Gaussian': {
        'color': '#3498DB',
        'marker': 's',
        'linestyle': '--',
        'label': 'Gaussian'
    },
    'RedSync': {
        'color': '#F39C12',
        'marker': '^',
        'linestyle': '-.',
        'label': 'RedSync'
    },
    'HGG-TopK': {
        'color': '#2ECC71',
        'marker': 'D',
        'linestyle': '-',
        'label': 'HGG-TopK'
    }
}


def load_results(log_dir):
    """加载实验结果"""
    result_file = os.path.join(log_dir, 'benchmark_results.json')

    if not os.path.exists(result_file):
        raise FileNotFoundError(f"Result file not found: {result_file}")

    with open(result_file, 'r') as f:
        data = json.load(f)

    return data


def format_size_label(size):
    """格式化张量大小标签"""
    if size >= 10**6:
        return f"{size/10**6:.0f}M"
    elif size >= 10**3:
        return f"{size/10**3:.0f}K"
    else:
        return str(size)


def plot_scaling_performance(results, output_dir):
    """
    绘制主图：所有算法的时间-梯度量曲线
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    data = results['data']
    sizes = sorted([int(s) for s in data.keys()])

    # 为每个压缩算法绘制曲线
    for comp_name, style in STYLE_CONFIG.items():
        times = []
        sizes_valid = []
        errors = []

        for size in sizes:
            size_data = data[str(size)]
            if comp_name in size_data and 'avg_time_ms' in size_data[comp_name]:
                times.append(size_data[comp_name]['avg_time_ms'])
                errors.append(size_data[comp_name]['std_time_ms'])
                sizes_valid.append(size)

        if times:
            # 绘制主线
            line = ax.plot(sizes_valid, times,
                          color=style['color'],
                          marker=style['marker'],
                          linestyle=style['linestyle'],
                          label=style['label'],
                          linewidth=2.5,
                          markersize=8,
                          markeredgecolor='white',
                          markeredgewidth=1.5,
                          alpha=0.9)

            # 添加误差带
            times_arr = np.array(times)
            errors_arr = np.array(errors)
            ax.fill_between(sizes_valid,
                           times_arr - errors_arr,
                           times_arr + errors_arr,
                           color=style['color'],
                           alpha=0.15)

    # 设置对数坐标
    ax.set_xscale('log')
    ax.set_yscale('log')

    # 设置标签
    ax.set_xlabel('Gradient Size (Number of Elements)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Sparsification Time (ms)', fontsize=14, fontweight='bold')
    ax.set_title('Sparsification Time vs. Gradient Size',
                fontsize=16, fontweight='bold', pad=15)

    # 设置网格
    ax.grid(True, which='both', alpha=0.3, linestyle='--', linewidth=0.8)
    ax.grid(True, which='minor', alpha=0.15, linestyle=':', linewidth=0.5)

    # 设置图例
    ax.legend(fontsize=12, loc='upper left', framealpha=0.95,
             edgecolor='black', fancybox=True, shadow=True)

    # 自定义x轴刻度标签
    ax.set_xticks([1e4, 1e5, 1e6, 1e7])
    ax.set_xticklabels(['10K', '100K', '1M', '10M'])

    # 设置刻度参数
    ax.tick_params(axis='both', which='major', labelsize=11, width=1.5, length=6)
    ax.tick_params(axis='both', which='minor', width=1, length=3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'exp8_scaling_performance.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_speedup_comparison(results, output_dir):
    """
    绘制加速比对比图：HGG-TopK相对于其他算法的加速比
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    data = results['data']
    sizes = sorted([int(s) for s in data.keys()])

    # 以TopK为基准计算加速比
    baseline = 'TopK'
    compare_algos = ['Gaussian', 'RedSync', 'HGG-TopK']

    for comp_name in compare_algos:
        speedups = []
        sizes_valid = []

        for size in sizes:
            size_data = data[str(size)]

            if (baseline in size_data and comp_name in size_data and
                'avg_time_ms' in size_data[baseline] and
                'avg_time_ms' in size_data[comp_name]):

                baseline_time = size_data[baseline]['avg_time_ms']
                comp_time = size_data[comp_name]['avg_time_ms']

                if comp_time > 0:
                    speedup = baseline_time / comp_time
                    speedups.append(speedup)
                    sizes_valid.append(size)

        if speedups:
            style = STYLE_CONFIG[comp_name]
            ax.plot(sizes_valid, speedups,
                   color=style['color'],
                   marker=style['marker'],
                   linestyle=style['linestyle'],
                   label=f"{style['label']} vs {baseline}",
                   linewidth=2.5,
                   markersize=8,
                   markeredgecolor='white',
                   markeredgewidth=1.5,
                   alpha=0.9)

    # 添加基准线 (1x)
    ax.axhline(y=1.0, color='black', linestyle=':', linewidth=2, alpha=0.5, label='Baseline (1x)')

    # 设置对数坐标（仅x轴）
    ax.set_xscale('log')

    # 设置标签
    ax.set_xlabel('Gradient Size (Number of Elements)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Speedup Relative to TopK', fontsize=14, fontweight='bold')
    ax.set_title('Speedup Comparison Against TopK',
                fontsize=16, fontweight='bold', pad=15)

    # 设置网格
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

    # 设置图例
    ax.legend(fontsize=11, loc='best', framealpha=0.95,
             edgecolor='black', fancybox=True, shadow=True)

    # 自定义x轴刻度标签
    ax.set_xticks([1e4, 1e5, 1e6, 1e7])
    ax.set_xticklabels(['10K', '100K', '1M', '10M'])

    # 设置刻度参数
    ax.tick_params(axis='both', which='major', labelsize=11, width=1.5, length=6)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'exp8_speedup_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_complexity_analysis(results, output_dir):
    """
    绘制复杂度分析图：展示时间增长率
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    data = results['data']
    sizes = sorted([int(s) for s in data.keys()])

    for comp_name, style in STYLE_CONFIG.items():
        times = []
        sizes_valid = []

        for size in sizes:
            size_data = data[str(size)]
            if comp_name in size_data and 'avg_time_ms' in size_data[comp_name]:
                times.append(size_data[comp_name]['avg_time_ms'])
                sizes_valid.append(size)

        if len(times) > 1:
            # 计算时间增长率 (相对于第一个数据点)
            times_normalized = np.array(times) / times[0]
            sizes_normalized = np.array(sizes_valid) / sizes_valid[0]

            ax.plot(sizes_normalized, times_normalized,
                   color=style['color'],
                   marker=style['marker'],
                   linestyle=style['linestyle'],
                   label=style['label'],
                   linewidth=2.5,
                   markersize=8,
                   markeredgecolor='white',
                   markeredgewidth=1.5,
                   alpha=0.9)

    # 添加理论复杂度参考线
    x_ref = np.array([1, sizes[-1] / sizes[0]])

    # O(n)
    ax.plot(x_ref, x_ref, 'k--', linewidth=1.5, alpha=0.5, label='O(n) reference')

    # O(n log n)
    y_nlogn = x_ref * np.log(x_ref * sizes[0]) / np.log(sizes[0])
    ax.plot(x_ref, y_nlogn, 'k:', linewidth=1.5, alpha=0.5, label='O(n log n) reference')

    # 设置对数坐标
    ax.set_xscale('log')
    ax.set_yscale('log')

    # 设置标签
    ax.set_xlabel('Relative Gradient Size (normalized)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Relative Time (normalized)', fontsize=14, fontweight='bold')
    ax.set_title('Time Complexity Analysis',
                fontsize=16, fontweight='bold', pad=15)

    # 设置网格
    ax.grid(True, which='both', alpha=0.3, linestyle='--', linewidth=0.8)
    ax.grid(True, which='minor', alpha=0.15, linestyle=':', linewidth=0.5)

    # 设置图例
    ax.legend(fontsize=11, loc='upper left', framealpha=0.95,
             edgecolor='black', fancybox=True, shadow=True)

    # 设置刻度参数
    ax.tick_params(axis='both', which='major', labelsize=11, width=1.5, length=6)
    ax.tick_params(axis='both', which='minor', width=1, length=3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'exp8_complexity_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path}")
    plt.close()


def generate_summary_report(results, output_dir):
    """生成文本摘要报告"""
    data = results['data']
    sizes = sorted([int(s) for s in data.keys()])

    report = []
    report.append("="*80)
    report.append("实验8: 不同梯度量下的稀疏化时间对比 - 结果摘要")
    report.append("="*80)
    report.append("")

    report.append("实验配置:")
    report.append(f"  Device: {results['config']['device']}")
    report.append(f"  Density: {results['config']['density']*100}%")
    report.append(f"  Warmup iterations: {results['config']['warmup_iterations']}")
    report.append(f"  Test iterations: {results['config']['test_iterations']}")
    report.append("")

    report.append("="*80)
    report.append("详细结果:")
    report.append("="*80)

    # 表头
    header = f"{'Size':<12}"
    for comp_name in ['TopK', 'Gaussian', 'RedSync', 'HGG-TopK']:
        header += f"{comp_name:<18}"
    report.append(header)
    report.append("-" * 80)

    # 数据行
    for size in sizes:
        size_str = format_size_label(size)
        row = f"{size_str:<12}"

        for comp_name in ['TopK', 'Gaussian', 'RedSync', 'HGG-TopK']:
            if comp_name in data[str(size)] and 'avg_time_ms' in data[str(size)][comp_name]:
                time_val = data[str(size)][comp_name]['avg_time_ms']
                std_val = data[str(size)][comp_name]['std_time_ms']
                row += f"{time_val:>6.2f}±{std_val:<4.2f} ms   "
            else:
                row += f"{'N/A':<18}"

        report.append(row)

    report.append("="*80)
    report.append("")

    # 加速比分析
    report.append("HGG-TopK相对于TopK的加速比:")
    report.append("-" * 80)

    for size in sizes:
        if ('TopK' in data[str(size)] and 'HGG-TopK' in data[str(size)] and
            'avg_time_ms' in data[str(size)]['TopK'] and
            'avg_time_ms' in data[str(size)]['HGG-TopK']):

            topk_time = data[str(size)]['TopK']['avg_time_ms']
            hgg_time = data[str(size)]['HGG-TopK']['avg_time_ms']

            if hgg_time > 0:
                speedup = topk_time / hgg_time
                comparison = "faster" if speedup > 1 else "slower"
                size_str = format_size_label(size)
                report.append(f"{size_str:<12}: {speedup:.2f}x ({comparison})")

    report.append("="*80)
    report.append("")

    # 关键发现
    report.append("关键发现:")
    report.append("-" * 80)

    # 找到最大梯度量下的性能
    max_size = sizes[-1]
    if str(max_size) in data:
        report.append(f"\n在最大梯度量 ({format_size_label(max_size)}) 下:")

        for comp_name in ['TopK', 'Gaussian', 'RedSync', 'HGG-TopK']:
            if comp_name in data[str(max_size)] and 'avg_time_ms' in data[str(max_size)][comp_name]:
                time_val = data[str(max_size)][comp_name]['avg_time_ms']
                report.append(f"  {comp_name}: {time_val:.2f} ms")

    report.append("")
    report.append("="*80)

    # 保存报告
    report_file = os.path.join(output_dir, 'exp8_summary_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    print(f"✓ Saved: {report_file}")

    # 打印到控制台
    print("\n" + '\n'.join(report))


def main():
    import argparse

    parser = argparse.ArgumentParser(description='实验8可视化')
    parser.add_argument('--log-dir', type=str, default='./logs/exp8_gradient_scaling',
                       help='日志目录')
    parser.add_argument('--output-dir', type=str, default='./figures/exp8',
                       help='输出目录')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "="*80)
    print("实验8可视化: 不同梯度量下的稀疏化时间对比")
    print("="*80 + "\n")

    # 加载结果
    print("Loading results...")
    try:
        results = load_results(args.log_dir)
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        return

    print(f"Found results for {len(results['data'])} gradient sizes")

    # 生成图表
    print("\n绘制主图：时间-梯度量曲线...")
    plot_scaling_performance(results, args.output_dir)

    print("\n绘制加速比对比图...")
    plot_speedup_comparison(results, args.output_dir)

    print("\n绘制复杂度分析图...")
    plot_complexity_analysis(results, args.output_dir)

    print("\n生成摘要报告...")
    generate_summary_report(results, args.output_dir)

    print("\n" + "="*80)
    print(f"✓ All figures and reports saved to: {args.output_dir}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
