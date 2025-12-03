# -*- coding: utf-8 -*-
"""
实验1可视化: 算法对比 - 堆叠柱形图
展示计算时间、通信时间、稀疏开销
"""

import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置科研风格
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

# 颜色方案
COLORS = {
    'forward': '#3498db',      # 蓝色 - Forward
    'backward': '#e74c3c',     # 红色 - Backward
    'sparsification': '#f39c12', # 橙色 - 稀疏化
    'communication': '#2ecc71',  # 绿色 - 通信
    'update': '#9b59b6'        # 紫色 - 更新
}

def load_experiment_results(log_dir):
    """加载实验结果"""
    results = {}

    json_files = glob.glob(os.path.join(log_dir, '*.json'))

    for json_file in json_files:
        filename = os.path.basename(json_file)
        if filename == 'experiment_summary.json':
            continue

        with open(json_file, 'r') as f:
            data = json.load(f)

        # 提取模型名和压缩器名
        model = data.get('model', 'unknown')
        compressor = data.get('compressor') or 'baseline'

        if model not in results:
            results[model] = {}

        results[model][compressor] = data

    return results


def plot_stacked_bar(results, output_dir):
    """绘制堆叠柱形图"""
    models = list(results.keys())

    # 为每个模型创建一个子图
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 6))

    if n_models == 1:
        axes = [axes]

    for idx, model in enumerate(models):
        ax = axes[idx]
        model_data = results[model]

        # 提取数据
        compressors = list(model_data.keys())
        n_comp = len(compressors)

        forward_times = []
        backward_times = []
        sparse_times = []
        comm_times = []
        update_times = []

        for comp in compressors:
            data = model_data[comp]
            forward_times.append(np.mean(data.get('forward_times', [0])))
            backward_times.append(np.mean(data.get('backward_times', [0])))
            sparse_times.append(np.mean(data.get('sparsification_times', [0])))
            comm_times.append(np.mean(data.get('communication_times', [0])))
            update_times.append(np.mean(data.get('update_times', [0])))

        # 绘制堆叠柱形图
        x = np.arange(n_comp)
        width = 0.6

        p1 = ax.bar(x, forward_times, width, label='Forward',
                   color=COLORS['forward'], edgecolor='black', linewidth=0.5)
        p2 = ax.bar(x, backward_times, width, bottom=forward_times,
                   label='Backward', color=COLORS['backward'], edgecolor='black', linewidth=0.5)

        bottom1 = np.array(forward_times) + np.array(backward_times)
        p3 = ax.bar(x, sparse_times, width, bottom=bottom1,
                   label='Sparsification', color=COLORS['sparsification'], edgecolor='black', linewidth=0.5)

        bottom2 = bottom1 + np.array(sparse_times)
        p4 = ax.bar(x, comm_times, width, bottom=bottom2,
                   label='Communication', color=COLORS['communication'], edgecolor='black', linewidth=0.5)

        bottom3 = bottom2 + np.array(comm_times)
        p5 = ax.bar(x, update_times, width, bottom=bottom3,
                   label='Update', color=COLORS['update'], edgecolor='black', linewidth=0.5)

        # 计算总时间并标注
        total_times = np.array(forward_times) + np.array(backward_times) + \
                     np.array(sparse_times) + np.array(comm_times) + np.array(update_times)

        for i, total in enumerate(total_times):
            ax.text(i, total + 0.5, f'{total:.1f}s',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

        # 设置标签
        ax.set_ylabel('Time per Epoch (s)', fontsize=11)
        ax.set_title(f'{model.upper()}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace('baseline', 'Base').replace('hggtopk', 'HGG-TopK')
                           for c in compressors], rotation=45, ha='right')
        ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'exp1_time_breakdown.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_overhead_comparison(results, output_dir):
    """绘制稀疏化和通信开销对比"""
    models = list(results.keys())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左图：稀疏化开销
    ax1 = axes[0]
    # 右图：通信开销
    ax2 = axes[1]

    for model in models:
        model_data = results[model]
        compressors = []
        sparse_overheads = []
        comm_overheads = []

        for comp, data in model_data.items():
            if comp == 'baseline':
                continue

            compressors.append(comp)

            sparse_time = np.mean(data.get('sparsification_times', [0]))
            comm_time = np.mean(data.get('communication_times', [0]))
            total_time = data.get('avg_epoch_time', 1.0)

            sparse_overhead = (sparse_time / total_time * 100) if total_time > 0 else 0
            comm_overhead = (comm_time / total_time * 100) if total_time > 0 else 0

            sparse_overheads.append(sparse_overhead)
            comm_overheads.append(comm_overhead)

        if compressors:
            x = np.arange(len(compressors))
            width = 0.2
            offset = models.index(model) * width

            ax1.bar(x + offset, sparse_overheads, width, label=model.upper())
            ax2.bar(x + offset, comm_overheads, width, label=model.upper())

    # 设置左图
    ax1.set_ylabel('Sparsification Overhead (%)', fontsize=11)
    ax1.set_title('(a) Sparsification Overhead', fontsize=12, fontweight='bold')
    ax1.set_xticks(x + width * (len(models) - 1) / 2)
    ax1.set_xticklabels([c.upper() for c in compressors], rotation=45, ha='right')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')

    # 设置右图
    ax2.set_ylabel('Communication Overhead (%)', fontsize=11)
    ax2.set_title('(b) Communication Overhead', fontsize=12, fontweight='bold')
    ax2.set_xticks(x + width * (len(models) - 1) / 2)
    ax2.set_xticklabels([c.upper() for c in compressors], rotation=45, ha='right')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'exp1_overhead_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path}")
    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='实验1可视化')
    parser.add_argument('--log-dir', type=str, default='./logs/exp1_algorithm_comparison',
                       help='日志目录')
    parser.add_argument('--output-dir', type=str, default='./figures/exp1',
                       help='输出目录')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "="*80)
    print("实验1可视化: 算法时间对比")
    print("="*80 + "\n")

    # 加载结果
    print("Loading results...")
    results = load_experiment_results(args.log_dir)

    if not results:
        print("⚠ No results found!")
        return

    print(f"Found results for {len(results)} models")

    # 生成图表
    print("\n绘制堆叠柱形图...")
    plot_stacked_bar(results, args.output_dir)

    print("\n绘制开销对比图...")
    plot_overhead_comparison(results, args.output_dir)

    print("\n" + "="*80)
    print(f"✓ All figures saved to: {args.output_dir}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
