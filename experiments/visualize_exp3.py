# -*- coding: utf-8 -*-
"""
实验3可视化: 精度和损失曲线
为每个模型生成一张精度图和一张损失图，展示所有压缩算法的对比
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
    'baseline': '#95a5a6',      # 灰色
    'topk': '#3498db',          # 蓝色
    'gaussian': '#e74c3c',      # 红色
    'redsync': '#f39c12',       # 橙色
    'hggtopk': '#2ecc71'        # 绿色
}

LABELS = {
    'baseline': 'Baseline (No Compression)',
    'topk': 'TopK',
    'gaussian': 'Gaussian',
    'redsync': 'RedSync',
    'hggtopk': 'HGG-TopK'
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

        model = data.get('model', 'unknown')
        compressor = data.get('compressor') or 'baseline'

        if model not in results:
            results[model] = {}

        results[model][compressor] = data

    return results


def plot_accuracy_curves(results, output_dir):
    """为每个模型绘制精度曲线"""
    for model, model_data in results.items():
        fig, ax = plt.subplots(figsize=(10, 6))

        for compressor in ['baseline', 'topk', 'gaussian', 'redsync', 'hggtopk']:
            if compressor not in model_data:
                continue

            data = model_data[compressor]

            # 获取精度数据
            if 'test_accuracies' in data and data['test_accuracies']:
                accuracies = data['test_accuracies']
                epochs = range(1, len(accuracies) + 1)

                ax.plot(epochs, accuracies,
                       label=LABELS[compressor],
                       color=COLORS[compressor],
                       linewidth=2,
                       marker='o',
                       markersize=4,
                       markevery=max(1, len(epochs)//10))

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Test Accuracy (%)', fontsize=12)
        ax.set_title(f'{model.upper()} - Test Accuracy', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(output_dir, f'{model}_accuracy.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {save_path}")
        plt.close()


def plot_loss_curves(results, output_dir):
    """为每个模型绘制损失曲线"""
    for model, model_data in results.items():
        fig, ax = plt.subplots(figsize=(10, 6))

        for compressor in ['baseline', 'topk', 'gaussian', 'redsync', 'hggtopk']:
            if compressor not in model_data:
                continue

            data = model_data[compressor]

            # 获取损失数据
            if 'train_losses' in data and data['train_losses']:
                losses = data['train_losses']
                epochs = range(1, len(losses) + 1)

                ax.plot(epochs, losses,
                       label=LABELS[compressor],
                       color=COLORS[compressor],
                       linewidth=2,
                       marker='s',
                       markersize=4,
                       markevery=max(1, len(epochs)//10))

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Training Loss', fontsize=12)
        ax.set_title(f'{model.upper()} - Training Loss', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(output_dir, f'{model}_loss.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {save_path}")
        plt.close()


def plot_perplexity_curves(results, output_dir):
    """为语言模型绘制困惑度曲线"""
    language_models = ['lstm', 'gpt2-small', 'gpt2-medium']

    for model in language_models:
        if model not in results:
            continue

        model_data = results[model]
        fig, ax = plt.subplots(figsize=(10, 6))

        for compressor in ['baseline', 'topk', 'gaussian', 'redsync', 'hggtopk']:
            if compressor not in model_data:
                continue

            data = model_data[compressor]

            # 获取困惑度数据
            if 'test_perplexities' in data and data['test_perplexities']:
                perplexities = data['test_perplexities']
                epochs = range(1, len(perplexities) + 1)

                ax.plot(epochs, perplexities,
                       label=LABELS[compressor],
                       color=COLORS[compressor],
                       linewidth=2,
                       marker='o',
                       markersize=4,
                       markevery=max(1, len(epochs)//10))

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Test Perplexity', fontsize=12)
        ax.set_title(f'{model.upper()} - Test Perplexity', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')  # 对数坐标更清晰

        plt.tight_layout()
        save_path = os.path.join(output_dir, f'{model}_perplexity.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {save_path}")
        plt.close()


def plot_final_metrics_comparison(results, output_dir):
    """对比所有模型的最终精度"""
    fig, ax = plt.subplots(figsize=(12, 6))

    models = list(results.keys())
    compressors = ['baseline', 'topk', 'gaussian', 'redsync', 'hggtopk']

    x = np.arange(len(models))
    width = 0.15

    for i, compressor in enumerate(compressors):
        final_accs = []
        for model in models:
            if compressor in results[model]:
                data = results[model][compressor]
                if 'test_accuracies' in data and data['test_accuracies']:
                    final_accs.append(data['test_accuracies'][-1])
                elif 'best_test_accuracy' in data:
                    final_accs.append(data['best_test_accuracy'])
                else:
                    final_accs.append(0)
            else:
                final_accs.append(0)

        offset = (i - len(compressors)//2) * width
        ax.bar(x + offset, final_accs, width, label=LABELS[compressor],
               color=COLORS[compressor], edgecolor='black', linewidth=0.5)

    ax.set_ylabel('Final Test Accuracy (%)', fontsize=11)
    ax.set_title('Final Test Accuracy Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in models], rotation=45, ha='right')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'final_accuracy_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path}")
    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='实验3可视化')
    parser.add_argument('--log-dir', type=str, default='./logs/exp3_accuracy_loss_curves',
                       help='日志目录')
    parser.add_argument('--output-dir', type=str, default='./figures/exp3',
                       help='输出目录')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "="*80)
    print("实验3可视化: 精度和损失曲线")
    print("="*80 + "\n")

    # 加载结果
    print("Loading results...")
    results = load_experiment_results(args.log_dir)

    if not results:
        print("⚠ No results found!")
        return

    print(f"Found results for {len(results)} models")

    # 生成图表
    print("\n绘制精度曲线...")
    plot_accuracy_curves(results, args.output_dir)

    print("\n绘制损失曲线...")
    plot_loss_curves(results, args.output_dir)

    print("\n绘制困惑度曲线...")
    plot_perplexity_curves(results, args.output_dir)

    print("\n绘制最终精度对比...")
    plot_final_metrics_comparison(results, args.output_dir)

    print("\n" + "="*80)
    print(f"✓ All figures saved to: {args.output_dir}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
