# -*- coding: utf-8 -*-
"""
实验5可视化: HGG-TopK最优分桶数分析
展示不同NUM_BINS对稀疏化时间、精度和压缩质量的影响
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


def extract_num_bins(filename):
    """从文件名提取NUM_BINS值"""
    import re
    match = re.search(r'bins_(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0


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
        num_bins = extract_num_bins(filename)

        if model not in results:
            results[model] = {}

        if num_bins > 0:
            results[model][num_bins] = data

    return results


def plot_sparsification_time_vs_bins(results, output_dir):
    """绘制稀疏化时间 vs 分桶数"""
    fig, ax = plt.subplots(figsize=(10, 6))

    for model, model_data in results.items():
        bins_list = sorted(model_data.keys())
        sparse_times = []

        for num_bins in bins_list:
            data = model_data[num_bins]
            sparse_time = np.mean(data.get('sparsification_times', [0]))
            sparse_times.append(sparse_time)

        ax.plot(bins_list, sparse_times, marker='o', linewidth=2, markersize=8,
               label=model.upper())

    ax.set_xlabel('Number of Bins (NUM_BINS)', fontsize=12)
    ax.set_ylabel('Sparsification Time per Epoch (s)', fontsize=12)
    ax.set_title('Sparsification Time vs Number of Bins', fontsize=14, fontweight='bold')
    ax.set_xscale('log', base=2)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'exp5_sparse_time_vs_bins.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_accuracy_vs_bins(results, output_dir):
    """绘制精度 vs 分桶数"""
    fig, ax = plt.subplots(figsize=(10, 6))

    for model, model_data in results.items():
        bins_list = sorted(model_data.keys())
        accuracies = []

        for num_bins in bins_list:
            data = model_data[num_bins]
            if 'best_test_accuracy' in data:
                accuracies.append(data['best_test_accuracy'])
            elif 'test_accuracies' in data and data['test_accuracies']:
                accuracies.append(max(data['test_accuracies']))
            else:
                accuracies.append(0)

        ax.plot(bins_list, accuracies, marker='s', linewidth=2, markersize=8,
               label=model.upper())

    ax.set_xlabel('Number of Bins (NUM_BINS)', fontsize=12)
    ax.set_ylabel('Best Test Accuracy (%)', fontsize=12)
    ax.set_title('Test Accuracy vs Number of Bins', fontsize=14, fontweight='bold')
    ax.set_xscale('log', base=2)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'exp5_accuracy_vs_bins.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_compression_quality_vs_bins(results, output_dir):
    """绘制压缩质量 vs 分桶数"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：压缩率
    ax1 = axes[0]
    # 右图：阈值精度
    ax2 = axes[1]

    for model, model_data in results.items():
        bins_list = sorted(model_data.keys())
        compression_ratios = []
        threshold_accuracies = []

        for num_bins in bins_list:
            data = model_data[num_bins]

            if 'compression_ratios' in data and data['compression_ratios']:
                compression_ratios.append(np.mean(data['compression_ratios']))
            else:
                compression_ratios.append(0)

            if 'threshold_accuracies' in data and data['threshold_accuracies']:
                threshold_accuracies.append(np.mean(data['threshold_accuracies']))
            else:
                threshold_accuracies.append(0)

        ax1.plot(bins_list, compression_ratios, marker='o', linewidth=2, markersize=8,
                label=model.upper())
        ax2.plot(bins_list, threshold_accuracies, marker='s', linewidth=2, markersize=8,
                label=model.upper())

    # 设置左图
    ax1.set_xlabel('Number of Bins (NUM_BINS)', fontsize=11)
    ax1.set_ylabel('Compression Ratio', fontsize=11)
    ax1.set_title('(a) Compression Ratio vs Bins', fontsize=12, fontweight='bold')
    ax1.set_xscale('log', base=2)
    ax1.axhline(y=0.05, color='red', linestyle='--', linewidth=1, label='Target (5%)')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 设置右图
    ax2.set_xlabel('Number of Bins (NUM_BINS)', fontsize=11)
    ax2.set_ylabel('Threshold Accuracy (Error)', fontsize=11)
    ax2.set_title('(b) Threshold Accuracy vs Bins', fontsize=12, fontweight='bold')
    ax2.set_xscale('log', base=2)
    ax2.set_yscale('log')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'exp5_compression_quality_vs_bins.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_overhead_vs_bins(results, output_dir):
    """绘制稀疏化开销 vs 分桶数"""
    fig, ax = plt.subplots(figsize=(10, 6))

    for model, model_data in results.items():
        bins_list = sorted(model_data.keys())
        overheads = []

        for num_bins in bins_list:
            data = model_data[num_bins]
            sparse_time = np.mean(data.get('sparsification_times', [0]))
            total_time = data.get('avg_epoch_time', 1.0)
            overhead = (sparse_time / total_time * 100) if total_time > 0 else 0
            overheads.append(overhead)

        ax.plot(bins_list, overheads, marker='o', linewidth=2, markersize=8,
               label=model.upper())

    ax.set_xlabel('Number of Bins (NUM_BINS)', fontsize=12)
    ax.set_ylabel('Sparsification Overhead (%)', fontsize=12)
    ax.set_title('Sparsification Overhead vs Number of Bins', fontsize=14, fontweight='bold')
    ax.set_xscale('log', base=2)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'exp5_overhead_vs_bins.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path}")
    plt.close()


def generate_recommendation(results, output_dir):
    """生成最优分桶数推荐"""
    recommendations = {}

    for model, model_data in results.items():
        bins_list = sorted(model_data.keys())

        # 计算综合得分：权衡速度、精度和开销
        scores = {}
        for num_bins in bins_list:
            data = model_data[num_bins]

            # 获取指标
            sparse_time = np.mean(data.get('sparsification_times', [0]))
            total_time = data.get('avg_epoch_time', 1.0)
            overhead = (sparse_time / total_time * 100) if total_time > 0 else 100

            if 'best_test_accuracy' in data:
                accuracy = data['best_test_accuracy']
            elif 'test_accuracies' in data and data['test_accuracies']:
                accuracy = max(data['test_accuracies'])
            else:
                accuracy = 0

            # 综合得分：精度高、开销低
            # 归一化到[0, 1]范围
            score = accuracy / 100.0 - overhead / 100.0
            scores[num_bins] = (score, sparse_time, overhead, accuracy)

        # 找到最佳分桶数
        best_bins = max(scores.keys(), key=lambda x: scores[x][0])
        recommendations[model] = {
            'best_bins': best_bins,
            'sparse_time': scores[best_bins][1],
            'overhead': scores[best_bins][2],
            'accuracy': scores[best_bins][3]
        }

    # 保存推荐
    report_file = os.path.join(output_dir, 'bucket_recommendation.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("HGG-TopK最优分桶数推荐\n")
        f.write("="*80 + "\n\n")

        for model, rec in recommendations.items():
            f.write(f"{model.upper()}:\n")
            f.write(f"  推荐NUM_BINS = {rec['best_bins']}\n")
            f.write(f"  稀疏化时间: {rec['sparse_time']:.3f}s\n")
            f.write(f"  稀疏化开销: {rec['overhead']:.2f}%\n")
            f.write(f"  最佳精度: {rec['accuracy']:.2f}%\n\n")

        f.write("="*80 + "\n")
        f.write("总体建议:\n")
        f.write("- 对于大多数场景，NUM_BINS=1024 提供了良好的速度和精度平衡\n")
        f.write("- 如果追求极致速度，可以使用 NUM_BINS=512\n")
        f.write("- 如果需要更高的压缩精度，可以使用 NUM_BINS=2048\n")
        f.write("="*80 + "\n")

    print(f"✓ Saved: {report_file}")

    return recommendations


def main():
    import argparse

    parser = argparse.ArgumentParser(description='实验5可视化')
    parser.add_argument('--log-dir', type=str, default='./logs/exp5_bucket_optimization',
                       help='日志目录')
    parser.add_argument('--output-dir', type=str, default='./figures/exp5',
                       help='输出目录')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "="*80)
    print("实验5可视化: HGG-TopK最优分桶数分析")
    print("="*80 + "\n")

    # 加载结果
    print("Loading results...")
    results = load_experiment_results(args.log_dir)

    if not results:
        print("⚠ No results found!")
        return

    print(f"Found results for {len(results)} models")

    # 生成图表
    print("\n绘制稀疏化时间 vs 分桶数...")
    plot_sparsification_time_vs_bins(results, args.output_dir)

    print("\n绘制精度 vs 分桶数...")
    plot_accuracy_vs_bins(results, args.output_dir)

    print("\n绘制压缩质量 vs 分桶数...")
    plot_compression_quality_vs_bins(results, args.output_dir)

    print("\n绘制开销 vs 分桶数...")
    plot_overhead_vs_bins(results, args.output_dir)

    print("\n生成最优分桶数推荐...")
    recommendations = generate_recommendation(results, args.output_dir)

    print("\n" + "="*80)
    print("推荐的NUM_BINS值:")
    print("="*80)
    for model, rec in recommendations.items():
        print(f"{model.upper()}: NUM_BINS = {rec['best_bins']}")

    print("\n" + "="*80)
    print(f"✓ All figures saved to: {args.output_dir}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
