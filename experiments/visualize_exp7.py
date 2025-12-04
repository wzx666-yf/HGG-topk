# -*- coding: utf-8 -*-
"""
实验7可视化: 不同稀疏率下的压缩算法对比
为每个稀疏率生成单独的图表，展示不同模型和压缩算法的性能
"""

import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

# 设置科研风格
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'
sns.set_style("whitegrid")

# 压缩算法颜色方案（科研论文常用配色）
COLORS = {
    'topk': '#E74C3C',       # 红色
    'gaussian': '#3498DB',   # 蓝色
    'redsync': '#F39C12',    # 橙色
    'hggtopk': '#2ECC71',    # 绿色
}

# 模型显示名称
MODEL_NAMES = {
    'resnet18': 'ResNet-18',
    'resnet50': 'ResNet-50',
    'vgg11': 'VGG-11',
    'vgg16': 'VGG-16',
    'lstm': 'LSTM',
    'gpt2-small': 'GPT-2 Small'
}


def load_experiment_results(log_dir):
    """加载实验结果，按模型-稀疏率-压缩器组织"""
    results = {}

    json_files = glob.glob(os.path.join(log_dir, '*.json'))

    for json_file in json_files:
        filename = os.path.basename(json_file)
        if filename == 'experiment_summary.json':
            continue

        with open(json_file, 'r') as f:
            data = json.load(f)

        # 提取模型名、压缩器名和稀疏率
        model = data.get('model', 'unknown')
        compressor = data.get('compressor', 'unknown')
        density = data.get('density', 0.05)

        if model not in results:
            results[model] = {}
        if density not in results[model]:
            results[model][density] = {}

        results[model][density][compressor] = data

    return results


def plot_sparsity_comparison(results, density, output_dir):
    """
    为单个稀疏率绘制对比图
    包含：训练时间、稀疏化时间、通信时间、测试精度
    """
    models = sorted(results.keys())
    compressors = ['topk', 'gaussian', 'redsync', 'hggtopk']

    # 创建2x2子图布局
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Performance Comparison at Sparsity Rate = {density*100:.0f}%',
                 fontsize=16, fontweight='bold', y=0.995)

    # 准备数据
    total_times = {comp: [] for comp in compressors}
    sparse_times = {comp: [] for comp in compressors}
    comm_times = {comp: [] for comp in compressors}
    accuracies = {comp: [] for comp in compressors}
    model_labels = []

    for model in models:
        if density not in results[model]:
            continue

        model_labels.append(MODEL_NAMES.get(model, model))

        for comp in compressors:
            if comp in results[model][density]:
                data = results[model][density][comp]
                total_times[comp].append(data.get('avg_epoch_time', 0))
                sparse_times[comp].append(np.mean(data.get('sparsification_times', [0])))
                comm_times[comp].append(np.mean(data.get('communication_times', [0])))
                # 获取最佳测试精度
                test_accs = data.get('test_accuracies', [0])
                accuracies[comp].append(max(test_accs) if test_accs else 0)
            else:
                total_times[comp].append(0)
                sparse_times[comp].append(0)
                comm_times[comp].append(0)
                accuracies[comp].append(0)

    x = np.arange(len(model_labels))
    width = 0.2

    # 子图1: 总训练时间
    ax1 = axes[0, 0]
    for i, comp in enumerate(compressors):
        offset = (i - 1.5) * width
        bars = ax1.bar(x + offset, total_times[comp], width,
                      label=comp.upper(), color=COLORS[comp],
                      edgecolor='black', linewidth=0.8, alpha=0.9)

    ax1.set_ylabel('Training Time per Epoch (s)', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Training Time', fontsize=13, fontweight='bold', pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_labels, rotation=45, ha='right')
    ax1.legend(fontsize=10, loc='upper left', framealpha=0.95)
    ax1.grid(True, alpha=0.3, axis='y')

    # 子图2: 稀疏化时间
    ax2 = axes[0, 1]
    for i, comp in enumerate(compressors):
        offset = (i - 1.5) * width
        bars = ax2.bar(x + offset, sparse_times[comp], width,
                      label=comp.upper(), color=COLORS[comp],
                      edgecolor='black', linewidth=0.8, alpha=0.9)

    ax2.set_ylabel('Sparsification Time (s)', fontsize=12, fontweight='bold')
    ax2.set_title('(b) Sparsification Overhead', fontsize=13, fontweight='bold', pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_labels, rotation=45, ha='right')
    ax2.legend(fontsize=10, loc='upper left', framealpha=0.95)
    ax2.grid(True, alpha=0.3, axis='y')

    # 子图3: 通信时间
    ax3 = axes[1, 0]
    for i, comp in enumerate(compressors):
        offset = (i - 1.5) * width
        bars = ax3.bar(x + offset, comm_times[comp], width,
                      label=comp.upper(), color=COLORS[comp],
                      edgecolor='black', linewidth=0.8, alpha=0.9)

    ax3.set_ylabel('Communication Time (s)', fontsize=12, fontweight='bold')
    ax3.set_title('(c) Communication Overhead', fontsize=13, fontweight='bold', pad=10)
    ax3.set_xticks(x)
    ax3.set_xticklabels(model_labels, rotation=45, ha='right')
    ax3.legend(fontsize=10, loc='upper left', framealpha=0.95)
    ax3.grid(True, alpha=0.3, axis='y')

    # 子图4: 测试精度
    ax4 = axes[1, 1]
    for i, comp in enumerate(compressors):
        offset = (i - 1.5) * width
        bars = ax4.bar(x + offset, accuracies[comp], width,
                      label=comp.upper(), color=COLORS[comp],
                      edgecolor='black', linewidth=0.8, alpha=0.9)
        # 标注精度值
        for j, (bar, val) in enumerate(zip(bars, accuracies[comp])):
            if val > 0:
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax4.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax4.set_title('(d) Best Test Accuracy', fontsize=13, fontweight='bold', pad=10)
    ax4.set_xticks(x)
    ax4.set_xticklabels(model_labels, rotation=45, ha='right')
    ax4.legend(fontsize=10, loc='lower left', framealpha=0.95)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim([0, 105])

    plt.tight_layout()
    save_path = os.path.join(output_dir, f'exp7_sparsity_{int(density*100)}percent.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_summary_heatmap(results, output_dir):
    """
    生成汇总热力图：展示所有稀疏率、模型和算法的性能
    """
    densities = sorted(set(d for model_data in results.values() for d in model_data.keys()))
    models = sorted(results.keys())
    compressors = ['topk', 'gaussian', 'redsync', 'hggtopk']

    # 创建图表
    fig, axes = plt.subplots(1, len(compressors), figsize=(18, 6))
    fig.suptitle('Performance Heatmap: Test Accuracy (%) Across All Sparsity Rates',
                 fontsize=16, fontweight='bold', y=1.02)

    for idx, comp in enumerate(compressors):
        ax = axes[idx]

        # 准备数据矩阵
        data_matrix = np.zeros((len(models), len(densities)))

        for i, model in enumerate(models):
            for j, density in enumerate(densities):
                if density in results[model] and comp in results[model][density]:
                    data = results[model][density][comp]
                    test_accs = data.get('test_accuracies', [0])
                    data_matrix[i, j] = max(test_accs) if test_accs else 0

        # 绘制热力图
        im = ax.imshow(data_matrix, cmap='YlGnBu', aspect='auto', vmin=0, vmax=100)

        # 设置刻度标签
        ax.set_xticks(np.arange(len(densities)))
        ax.set_yticks(np.arange(len(models)))
        ax.set_xticklabels([f'{d*100:.0f}%' for d in densities])
        ax.set_yticklabels([MODEL_NAMES.get(m, m) for m in models])

        # 标注数值
        for i in range(len(models)):
            for j in range(len(densities)):
                val = data_matrix[i, j]
                if val > 0:
                    text = ax.text(j, i, f'{val:.1f}',
                                 ha="center", va="center", color="black" if val > 50 else "white",
                                 fontsize=9, fontweight='bold')

        ax.set_title(f'{comp.upper()}', fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel('Sparsity Rate', fontsize=11, fontweight='bold')
        if idx == 0:
            ax.set_ylabel('Model', fontsize=11, fontweight='bold')

        # 添加颜色条
        if idx == len(compressors) - 1:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Accuracy (%)', fontsize=11, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'exp7_summary_heatmap.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_efficiency_comparison(results, output_dir):
    """
    生成效率对比图：稀疏化时间占比 vs 精度损失
    """
    densities = sorted(set(d for model_data in results.values() for d in model_data.keys()))
    compressors = ['topk', 'gaussian', 'redsync', 'hggtopk']

    fig, axes = plt.subplots(1, len(densities), figsize=(5*len(densities), 5))
    if len(densities) == 1:
        axes = [axes]

    fig.suptitle('Efficiency-Accuracy Trade-off Analysis',
                 fontsize=16, fontweight='bold', y=1.02)

    for idx, density in enumerate(densities):
        ax = axes[idx]

        for comp in compressors:
            sparse_overheads = []
            accuracies = []

            for model in results.keys():
                if density in results[model] and comp in results[model][density]:
                    data = results[model][density][comp]

                    # 计算稀疏化开销百分比
                    sparse_time = np.mean(data.get('sparsification_times', [0]))
                    total_time = data.get('avg_epoch_time', 1.0)
                    overhead = (sparse_time / total_time * 100) if total_time > 0 else 0

                    # 获取精度
                    test_accs = data.get('test_accuracies', [0])
                    acc = max(test_accs) if test_accs else 0

                    sparse_overheads.append(overhead)
                    accuracies.append(acc)

            if sparse_overheads and accuracies:
                ax.scatter(sparse_overheads, accuracies,
                          label=comp.upper(), color=COLORS[comp],
                          s=120, alpha=0.7, edgecolor='black', linewidth=1.5)

        ax.set_xlabel('Sparsification Overhead (%)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Test Accuracy (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'Sparsity Rate = {density*100:.0f}%', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10, loc='best', framealpha=0.95)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'exp7_efficiency_tradeoff.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path}")
    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='实验7可视化')
    parser.add_argument('--log-dir', type=str, default='./logs/exp7_sparsity_comparison',
                       help='日志目录')
    parser.add_argument('--output-dir', type=str, default='./figures/exp7',
                       help='输出目录')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "="*80)
    print("实验7可视化: 不同稀疏率下的压缩算法对比")
    print("="*80 + "\n")

    # 加载结果
    print("Loading results...")
    results = load_experiment_results(args.log_dir)

    if not results:
        print("⚠ No results found!")
        return

    print(f"Found results for {len(results)} models")

    # 获取所有稀疏率
    all_densities = set()
    for model_data in results.values():
        all_densities.update(model_data.keys())
    densities = sorted(all_densities)

    print(f"Found {len(densities)} sparsity rates: {[f'{d*100:.0f}%' for d in densities]}")

    # 为每个稀疏率生成对比图
    print("\n生成各稀疏率对比图...")
    for density in densities:
        print(f"  绘制稀疏率 {density*100:.0f}% ...")
        plot_sparsity_comparison(results, density, args.output_dir)

    # 生成汇总热力图
    print("\n生成汇总热力图...")
    plot_summary_heatmap(results, args.output_dir)

    # 生成效率对比图
    print("\n生成效率权衡分析图...")
    plot_efficiency_comparison(results, args.output_dir)

    print("\n" + "="*80)
    print(f"✓ All figures saved to: {args.output_dir}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
