# -*- coding: utf-8 -*-
"""
实验2可视化: HGG-TopK历史阈值 vs 全局二分
展示稀疏化时间和总训练时间对比
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

        model = data.get('model', 'unknown')
        compressor = data.get('compressor', 'unknown')

        if model not in results:
            results[model] = {}

        # 区分galloping和global
        if 'no_history_threshold' in data and data['no_history_threshold']:
            key = 'hggtopk_global'
        else:
            key = 'hggtopk_galloping'

        results[model][key] = data

    return results


def plot_time_comparison(results, output_dir):
    """对比稀疏化时间和总时间"""
    models = list(results.keys())
    n_models = len(models)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左图：稀疏化时间对比
    ax1 = axes[0]
    # 右图：总训练时间对比
    ax2 = axes[1]

    x = np.arange(n_models)
    width = 0.35

    galloping_sparse = []
    global_sparse = []
    galloping_total = []
    global_total = []

    for model in models:
        model_data = results[model]

        if 'hggtopk_galloping' in model_data:
            galloping_data = model_data['hggtopk_galloping']
            galloping_sparse.append(np.mean(galloping_data.get('sparsification_times', [0])))
            galloping_total.append(galloping_data.get('avg_epoch_time', 0))
        else:
            galloping_sparse.append(0)
            galloping_total.append(0)

        if 'hggtopk_global' in model_data:
            global_data = model_data['hggtopk_global']
            global_sparse.append(np.mean(global_data.get('sparsification_times', [0])))
            global_total.append(global_data.get('avg_epoch_time', 0))
        else:
            global_sparse.append(0)
            global_total.append(0)

    # 绘制稀疏化时间
    bars1 = ax1.bar(x - width/2, galloping_sparse, width, label='Galloping Search (w/ History)',
                    color='#3498db', edgecolor='black', linewidth=0.5)
    bars2 = ax1.bar(x + width/2, global_sparse, width, label='Global Binary Search',
                    color='#e74c3c', edgecolor='black', linewidth=0.5)

    # 添加数值标签和加速比
    for i in range(n_models):
        if galloping_sparse[i] > 0 and global_sparse[i] > 0:
            speedup = global_sparse[i] / galloping_sparse[i]
            ax1.text(i, max(galloping_sparse[i], global_sparse[i]) + 0.5,
                    f'{speedup:.1f}x', ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color='green')

    ax1.set_ylabel('Sparsification Time per Epoch (s)', fontsize=11)
    ax1.set_title('(a) Sparsification Time Comparison', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.upper() for m in models])
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')

    # 绘制总训练时间
    bars3 = ax2.bar(x - width/2, galloping_total, width, label='Galloping Search (w/ History)',
                    color='#3498db', edgecolor='black', linewidth=0.5)
    bars4 = ax2.bar(x + width/2, global_total, width, label='Global Binary Search',
                    color='#e74c3c', edgecolor='black', linewidth=0.5)

    # 添加数值标签和加速比
    for i in range(n_models):
        if galloping_total[i] > 0 and global_total[i] > 0:
            speedup = global_total[i] / galloping_total[i]
            ax2.text(i, max(galloping_total[i], global_total[i]) + 2,
                    f'{speedup:.2f}x', ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color='green')

    ax2.set_ylabel('Total Time per Epoch (s)', fontsize=11)
    ax2.set_title('(b) Total Training Time Comparison', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.upper() for m in models])
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'exp2_galloping_vs_binary.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_overhead_comparison(results, output_dir):
    """对比稀疏化开销百分比"""
    models = list(results.keys())
    n_models = len(models)

    fig, ax = plt.subplots(figsize=(8, 6))

    x = np.arange(n_models)
    width = 0.35

    galloping_overhead = []
    global_overhead = []

    for model in models:
        model_data = results[model]

        if 'hggtopk_galloping' in model_data:
            data = model_data['hggtopk_galloping']
            sparse_time = np.mean(data.get('sparsification_times', [0]))
            total_time = data.get('avg_epoch_time', 1.0)
            overhead = (sparse_time / total_time * 100) if total_time > 0 else 0
            galloping_overhead.append(overhead)
        else:
            galloping_overhead.append(0)

        if 'hggtopk_global' in model_data:
            data = model_data['hggtopk_global']
            sparse_time = np.mean(data.get('sparsification_times', [0]))
            total_time = data.get('avg_epoch_time', 1.0)
            overhead = (sparse_time / total_time * 100) if total_time > 0 else 0
            global_overhead.append(overhead)
        else:
            global_overhead.append(0)

    # 绘制开销百分比
    bars1 = ax.bar(x - width/2, galloping_overhead, width, label='Galloping Search (w/ History)',
                   color='#2ecc71', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, global_overhead, width, label='Global Binary Search',
                   color='#f39c12', edgecolor='black', linewidth=0.5)

    # 添加数值标签
    for i in range(n_models):
        ax.text(i - width/2, galloping_overhead[i] + 0.5,
                f'{galloping_overhead[i]:.1f}%', ha='center', va='bottom', fontsize=9)
        ax.text(i + width/2, global_overhead[i] + 0.5,
                f'{global_overhead[i]:.1f}%', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Sparsification Overhead (%)', fontsize=11)
    ax.set_title('Sparsification Overhead Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in models])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'exp2_overhead.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path}")
    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='实验2可视化')
    parser.add_argument('--log-dir', type=str, default='./logs/exp2_galloping_vs_binary',
                       help='日志目录')
    parser.add_argument('--output-dir', type=str, default='./figures/exp2',
                       help='输出目录')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "="*80)
    print("实验2可视化: HGG-TopK历史阈值 vs 全局二分")
    print("="*80 + "\n")

    # 加载结果
    print("Loading results...")
    results = load_experiment_results(args.log_dir)

    if not results:
        print("⚠ No results found!")
        return

    print(f"Found results for {len(results)} models")

    # 生成图表
    print("\n绘制时间对比图...")
    plot_time_comparison(results, args.output_dir)

    print("\n绘制开销对比图...")
    plot_overhead_comparison(results, args.output_dir)

    print("\n" + "="*80)
    print(f"✓ All figures saved to: {args.output_dir}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
