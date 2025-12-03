# -*- coding: utf-8 -*-
"""
实验4可视化: HGG-TopK流水线对比
展示使用流水线和不使用流水线的时间对比
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

        if model not in results:
            results[model] = {}

        # 区分流水线和非流水线
        if data.get('use_pipeline', False):
            key = 'with_pipeline'
        else:
            key = 'no_pipeline'

        results[model][key] = data

    return results


def plot_time_breakdown(results, output_dir):
    """绘制时间分解对比图"""
    models = list(results.keys())
    n_models = len(models)

    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 6))

    if n_models == 1:
        axes = [axes]

    for idx, model in enumerate(models):
        ax = axes[idx]
        model_data = results[model]

        configs = ['no_pipeline', 'with_pipeline']
        x = np.arange(len(configs))
        width = 0.6

        forward_times = []
        backward_times = []
        sparse_times = []
        comm_times = []
        update_times = []

        for config in configs:
            if config not in model_data:
                forward_times.append(0)
                backward_times.append(0)
                sparse_times.append(0)
                comm_times.append(0)
                update_times.append(0)
                continue

            data = model_data[config]
            forward_times.append(np.mean(data.get('forward_times', [0])))
            backward_times.append(np.mean(data.get('backward_times', [0])))
            sparse_times.append(np.mean(data.get('sparsification_times', [0])))
            comm_times.append(np.mean(data.get('communication_times', [0])))
            update_times.append(np.mean(data.get('update_times', [0])))

        # 绘制堆叠柱形图
        p1 = ax.bar(x, forward_times, width, label='Forward',
                   color='#3498db', edgecolor='black', linewidth=0.5)
        p2 = ax.bar(x, backward_times, width, bottom=forward_times,
                   label='Backward', color='#e74c3c', edgecolor='black', linewidth=0.5)

        bottom1 = np.array(forward_times) + np.array(backward_times)
        p3 = ax.bar(x, sparse_times, width, bottom=bottom1,
                   label='Sparsification', color='#f39c12', edgecolor='black', linewidth=0.5)

        bottom2 = bottom1 + np.array(sparse_times)
        p4 = ax.bar(x, comm_times, width, bottom=bottom2,
                   label='Communication', color='#2ecc71', edgecolor='black', linewidth=0.5)

        bottom3 = bottom2 + np.array(comm_times)
        p5 = ax.bar(x, update_times, width, bottom=bottom3,
                   label='Update', color='#9b59b6', edgecolor='black', linewidth=0.5)

        # 计算总时间并标注
        total_times = np.array(forward_times) + np.array(backward_times) + \
                     np.array(sparse_times) + np.array(comm_times) + np.array(update_times)

        for i, total in enumerate(total_times):
            ax.text(i, total + 0.5, f'{total:.1f}s',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

        # 标注加速比
        if total_times[0] > 0 and total_times[1] > 0:
            speedup = total_times[0] / total_times[1]
            ax.text(0.5, max(total_times) * 1.15, f'Speedup: {speedup:.2f}x',
                   ha='center', va='bottom', fontsize=10, fontweight='bold', color='green')

        # 设置标签
        ax.set_ylabel('Time per Epoch (s)', fontsize=11)
        ax.set_title(f'{model.upper()}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['No Pipeline', 'With Pipeline'], rotation=0)
        ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'exp4_pipeline_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_speedup_analysis(results, output_dir):
    """绘制加速比分析"""
    models = list(results.keys())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左图：总时间对比
    ax1 = axes[0]
    # 右图：稀疏化+通信时间对比

    ax2 = axes[1]

    no_pipeline_total = []
    with_pipeline_total = []
    no_pipeline_overhead = []
    with_pipeline_overhead = []

    for model in models:
        model_data = results[model]

        if 'no_pipeline' in model_data:
            data = model_data['no_pipeline']
            no_pipeline_total.append(data.get('avg_epoch_time', 0))
            sparse_time = np.mean(data.get('sparsification_times', [0]))
            comm_time = np.mean(data.get('communication_times', [0]))
            no_pipeline_overhead.append(sparse_time + comm_time)
        else:
            no_pipeline_total.append(0)
            no_pipeline_overhead.append(0)

        if 'with_pipeline' in model_data:
            data = model_data['with_pipeline']
            with_pipeline_total.append(data.get('avg_epoch_time', 0))
            sparse_time = np.mean(data.get('sparsification_times', [0]))
            comm_time = np.mean(data.get('communication_times', [0]))
            with_pipeline_overhead.append(sparse_time + comm_time)
        else:
            with_pipeline_total.append(0)
            with_pipeline_overhead.append(0)

    x = np.arange(len(models))
    width = 0.35

    # 总时间对比
    bars1 = ax1.bar(x - width/2, no_pipeline_total, width, label='No Pipeline',
                    color='#e74c3c', edgecolor='black', linewidth=0.5)
    bars2 = ax1.bar(x + width/2, with_pipeline_total, width, label='With Pipeline',
                    color='#2ecc71', edgecolor='black', linewidth=0.5)

    # 添加加速比
    for i in range(len(models)):
        if no_pipeline_total[i] > 0 and with_pipeline_total[i] > 0:
            speedup = no_pipeline_total[i] / with_pipeline_total[i]
            ax1.text(i, max(no_pipeline_total[i], with_pipeline_total[i]) + 2,
                    f'{speedup:.2f}x', ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color='green')

    ax1.set_ylabel('Total Time per Epoch (s)', fontsize=11)
    ax1.set_title('(a) Total Training Time', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.upper() for m in models])
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')

    # 稀疏化+通信开销对比
    bars3 = ax2.bar(x - width/2, no_pipeline_overhead, width, label='No Pipeline',
                    color='#e74c3c', edgecolor='black', linewidth=0.5)
    bars4 = ax2.bar(x + width/2, with_pipeline_overhead, width, label='With Pipeline',
                    color='#2ecc71', edgecolor='black', linewidth=0.5)

    # 添加降低百分比
    for i in range(len(models)):
        if no_pipeline_overhead[i] > 0:
            reduction = (1 - with_pipeline_overhead[i] / no_pipeline_overhead[i]) * 100
            ax2.text(i, max(no_pipeline_overhead[i], with_pipeline_overhead[i]) + 0.5,
                    f'-{reduction:.1f}%', ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color='green')

    ax2.set_ylabel('Sparse + Comm Time (s)', fontsize=11)
    ax2.set_title('(b) Sparsification + Communication Overhead', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.upper() for m in models])
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'exp4_speedup_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path}")
    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='实验4可视化')
    parser.add_argument('--log-dir', type=str, default='./logs/exp4_pipeline_comparison',
                       help='日志目录')
    parser.add_argument('--output-dir', type=str, default='./figures/exp4',
                       help='输出目录')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "="*80)
    print("实验4可视化: HGG-TopK流水线对比")
    print("="*80 + "\n")

    # 加载结果
    print("Loading results...")
    results = load_experiment_results(args.log_dir)

    if not results:
        print("⚠ No results found!")
        return

    print(f"Found results for {len(results)} models")

    # 生成图表
    print("\n绘制时间分解对比图...")
    plot_time_breakdown(results, args.output_dir)

    print("\n绘制加速比分析...")
    plot_speedup_analysis(results, args.output_dir)

    print("\n" + "="*80)
    print(f"✓ All figures saved to: {args.output_dir}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
