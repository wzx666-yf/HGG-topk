# -*- coding: utf-8 -*-
"""
可视化工具 - 生成PNG图表（科研风格）

特性:
- PNG格式输出，高分辨率
- 时间戳命名
- 同类数据在一张图中对比
- 科研论文风格
"""

import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from typing import Dict, List
from datetime import datetime

# 设置科研风格
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['mathtext.fontset'] = 'stix'
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)

# 科研配色方案（色盲友好）
COLORS = {
    'baseline': '#1f77b4',  # 蓝色
    'topk': '#ff7f0e',      # 橙色
    'topk2': '#ff9f4e',     # 浅橙色
    'gaussian': '#2ca02c',  # 绿色
    'gaussian2': '#5cc05c', # 浅绿色
    'dgcsampling': '#d62728', # 红色
    'redsync': '#9467bd',   # 紫色
    'randomk': '#8c564b',   # 棕色
    'randomkec': '#bc847b', # 浅棕色
    'hggtopk': '#e377c2',   # 粉色
}

NAMES = {
    'baseline': 'Baseline',
    'topk': 'TopK',
    'topk2': 'TopK-NoEC',
    'gaussian': 'Gaussian',
    'gaussian2': 'Gaussian-NoEC',
    'dgcsampling': 'DGC-Sampling',
    'redsync': 'RedSync',
    'randomk': 'RandomK',
    'randomkec': 'RandomK-EC',
    'hggtopk': 'HGG-TopK',
}


class Visualizer:
    """可视化器 - 生成PNG格式图表"""

    def __init__(self, log_dir='./logs', output_dir='./figures'):
        self.log_dir = log_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 生成时间戳前缀
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def load_results(self):
        """加载所有结果"""
        files = glob.glob(os.path.join(self.log_dir, '*.json'))
        results = []
        for file in files:
            with open(file, 'r') as f:
                results.append(json.load(f))
        print(f"✓ Loaded {len(results)} result files")
        return results

    def _get_save_path(self, name):
        """生成带时间戳的保存路径"""
        filename = f"{self.timestamp}_{name}.png"
        return os.path.join(self.output_dir, filename)

    def plot_training_curves(self, results):
        """绘制训练曲线（4子图合并）"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Training Curves Comparison', fontsize=14, fontweight='bold', y=0.995)

        for result in results:
            compressor = result.get('compressor') or 'baseline'
            use_pipeline = result.get('use_pipeline', False)
            label = NAMES.get(compressor, compressor)
            if use_pipeline:
                label += ' (Pipeline)'
            color = COLORS.get(compressor, 'gray')
            linestyle = '--' if use_pipeline else '-'

            # (a) Test Accuracy
            if 'test_accs' in result and result['test_accs']:
                epochs = range(1, len(result['test_accs']) + 1)
                axes[0, 0].plot(epochs, result['test_accs'], label=label,
                              color=color, linewidth=2, linestyle=linestyle, alpha=0.8)

            # (b) Train Loss
            if 'train_losses' in result and result['train_losses']:
                epochs = range(1, len(result['train_losses']) + 1)
                axes[0, 1].plot(epochs, result['train_losses'], label=label,
                              color=color, linewidth=2, linestyle=linestyle, alpha=0.8)

            # (c) Train Accuracy
            if 'train_accs' in result and result['train_accs']:
                epochs = range(1, len(result['train_accs']) + 1)
                axes[1, 0].plot(epochs, result['train_accs'], label=label,
                              color=color, linewidth=2, linestyle=linestyle, alpha=0.8)

            # (d) Time per Epoch
            if 'train_times' in result and result['train_times']:
                epochs = range(1, len(result['train_times']) + 1)
                axes[1, 1].plot(epochs, result['train_times'], label=label,
                              color=color, linewidth=2, linestyle=linestyle, alpha=0.8)

        # 子图样式设置
        axes[0, 0].set(xlabel='Epoch', ylabel='Test Accuracy (%)')
        axes[0, 0].set_title('(a) Test Accuracy', fontsize=11)
        axes[0, 0].legend(fontsize=8, framealpha=0.9, loc='best')
        axes[0, 0].grid(True, alpha=0.3, linestyle='--')

        axes[0, 1].set(xlabel='Epoch', ylabel='Training Loss')
        axes[0, 1].set_title('(b) Training Loss', fontsize=11)
        axes[0, 1].legend(fontsize=8, framealpha=0.9, loc='best')
        axes[0, 1].grid(True, alpha=0.3, linestyle='--')

        axes[1, 0].set(xlabel='Epoch', ylabel='Training Accuracy (%)')
        axes[1, 0].set_title('(c) Training Accuracy', fontsize=11)
        axes[1, 0].legend(fontsize=8, framealpha=0.9, loc='best')
        axes[1, 0].grid(True, alpha=0.3, linestyle='--')

        axes[1, 1].set(xlabel='Epoch', ylabel='Time per Epoch (s)')
        axes[1, 1].set_title('(d) Training Time', fontsize=11)
        axes[1, 1].legend(fontsize=8, framealpha=0.9, loc='best')
        axes[1, 1].grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()
        save_path = self._get_save_path('training_curves')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {save_path}")
        plt.close()

    def plot_performance_comparison(self, results):
        """绘制性能对比（6子图）"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle('Performance Comparison', fontsize=14, fontweight='bold', y=0.995)

        methods = []
        best_accs = []
        avg_times = []
        sparse_overheads = []
        threshold_accs = []
        colors_list = []

        for result in results:
            compressor = result.get('compressor') or 'baseline'
            use_pipeline = result.get('use_pipeline', False)
            label = NAMES.get(compressor, compressor)
            if use_pipeline:
                label += '\n(Pipe)'

            methods.append(label)
            best_accs.append(result.get('best_acc', 0))
            avg_times.append(result.get('avg_epoch_time', 0))
            colors_list.append(COLORS.get(compressor, 'gray'))

            # 稀疏化开销
            if 'avg_sparsification_time' in result and 'avg_epoch_time' in result and result['avg_epoch_time'] > 0:
                overhead = result['avg_sparsification_time'] / result['avg_epoch_time'] * 100
                sparse_overheads.append(overhead)
            else:
                sparse_overheads.append(0)

            threshold_accs.append(result.get('avg_threshold_accuracy', 0))

        x = np.arange(len(methods))

        # (a) Best Accuracy
        axes[0, 0].bar(x, best_accs, color=colors_list, edgecolor='black', linewidth=0.5, alpha=0.8)
        axes[0, 0].set_ylabel('Best Test Accuracy (%)', fontsize=10)
        axes[0, 0].set_title('(a) Best Accuracy', fontsize=11)
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
        axes[0, 0].grid(True, alpha=0.3, axis='y', linestyle='--')
        # 添加数值标签
        for i, v in enumerate(best_accs):
            axes[0, 0].text(i, v, f'{v:.1f}', ha='center', va='bottom', fontsize=7)

        # (b) Training Time
        axes[0, 1].bar(x, avg_times, color=colors_list, edgecolor='black', linewidth=0.5, alpha=0.8)
        axes[0, 1].set_ylabel('Avg Time per Epoch (s)', fontsize=10)
        axes[0, 1].set_title('(b) Training Time', fontsize=11)
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
        axes[0, 1].grid(True, alpha=0.3, axis='y', linestyle='--')
        for i, v in enumerate(avg_times):
            axes[0, 1].text(i, v, f'{v:.1f}', ha='center', va='bottom', fontsize=7)

        # (c) Sparsification Overhead
        axes[0, 2].bar(x, sparse_overheads, color=colors_list, edgecolor='black', linewidth=0.5, alpha=0.8)
        axes[0, 2].set_ylabel('Sparsification Overhead (%)', fontsize=10)
        axes[0, 2].set_title('(c) Sparsification Overhead', fontsize=11)
        axes[0, 2].set_xticks(x)
        axes[0, 2].set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
        axes[0, 2].grid(True, alpha=0.3, axis='y', linestyle='--')
        for i, v in enumerate(sparse_overheads):
            if v > 0:
                axes[0, 2].text(i, v, f'{v:.1f}', ha='center', va='bottom', fontsize=7)

        # (d) Threshold Accuracy
        axes[1, 0].bar(x, threshold_accs, color=colors_list, edgecolor='black', linewidth=0.5, alpha=0.8)
        axes[1, 0].set_ylabel('Threshold Relative Error', fontsize=10)
        axes[1, 0].set_title('(d) Threshold Accuracy', fontsize=11)
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
        axes[1, 0].grid(True, alpha=0.3, axis='y', linestyle='--')

        # (e) Time Breakdown (堆叠柱状图)
        if any('forward_times' in r for r in results):
            fwd_times = [np.mean(r.get('forward_times', [0])) for r in results]
            bwd_times = [np.mean(r.get('backward_times', [0])) for r in results]
            sparse_times = [np.mean(r.get('sparsification_times', [0])) for r in results]

            axes[1, 1].bar(x, fwd_times, label='Forward', color='#3498db', edgecolor='black', linewidth=0.5)
            axes[1, 1].bar(x, bwd_times, bottom=fwd_times, label='Backward',
                          color='#e74c3c', edgecolor='black', linewidth=0.5)
            axes[1, 1].bar(x, sparse_times, bottom=np.array(fwd_times)+np.array(bwd_times),
                          label='Sparsification', color='#f39c12', edgecolor='black', linewidth=0.5)
            axes[1, 1].set_ylabel('Time (s)', fontsize=10)
            axes[1, 1].set_title('(e) Time Breakdown', fontsize=11)
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
            axes[1, 1].legend(fontsize=8, loc='best', framealpha=0.9)
            axes[1, 1].grid(True, alpha=0.3, axis='y', linestyle='--')

        # (f) Accuracy vs Time (散点图)
        axes[1, 2].scatter(avg_times, best_accs, s=150, c=colors_list, alpha=0.7,
                          edgecolors='black', linewidths=2)
        for i, method in enumerate(methods):
            axes[1, 2].annotate(method.replace('\n', ' '), (avg_times[i], best_accs[i]),
                              fontsize=7, ha='center', va='bottom')
        axes[1, 2].set_xlabel('Avg Time per Epoch (s)', fontsize=10)
        axes[1, 2].set_ylabel('Best Accuracy (%)', fontsize=10)
        axes[1, 2].set_title('(f) Accuracy vs Time', fontsize=11)
        axes[1, 2].grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()
        save_path = self._get_save_path('performance_comparison')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {save_path}")
        plt.close()

    def plot_pipeline_comparison(self, results):
        """绘制流水线对比"""
        no_pipe = None
        with_pipe = None

        for r in results:
            if r.get('compressor') == 'hggtopk':
                if r.get('use_pipeline'):
                    with_pipe = r
                else:
                    no_pipe = r

        if not no_pipe or not with_pipe:
            print("⚠ Pipeline comparison data not available")
            return

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle('HGG-TopK Pipeline Comparison', fontsize=14, fontweight='bold')

        # (a) Time Breakdown
        categories = ['Forward', 'Backward', 'Sparsification']
        no_pipe_times = [
            np.mean(no_pipe.get('forward_times', [0])),
            np.mean(no_pipe.get('backward_times', [0])),
            np.mean(no_pipe.get('sparsification_times', [0]))
        ]
        with_pipe_times = [
            np.mean(with_pipe.get('forward_times', [0])),
            np.mean(with_pipe.get('backward_times', [0])),
            np.mean(with_pipe.get('sparsification_times', [0]))
        ]

        x = np.arange(len(categories))
        width = 0.35
        axes[0].bar(x - width/2, no_pipe_times, width, label='No Pipeline',
                   color='#3498db', edgecolor='black', linewidth=0.5)
        axes[0].bar(x + width/2, with_pipe_times, width, label='With Pipeline',
                   color='#e74c3c', edgecolor='black', linewidth=0.5)
        axes[0].set_ylabel('Time (s)', fontsize=10)
        axes[0].set_title('(a) Time Breakdown', fontsize=11)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(categories, fontsize=9)
        axes[0].legend(fontsize=9, framealpha=0.9)
        axes[0].grid(True, alpha=0.3, axis='y', linestyle='--')

        # (b) Sparsification Overhead
        total_no = sum(no_pipe_times)
        total_with = sum(with_pipe_times)
        overhead_no = no_pipe_times[2] / total_no * 100 if total_no > 0 else 0
        overhead_with = with_pipe_times[2] / total_with * 100 if total_with > 0 else 0

        axes[1].bar(['No Pipeline', 'With Pipeline'], [overhead_no, overhead_with],
                   color=['#3498db', '#e74c3c'], edgecolor='black', linewidth=0.5)
        axes[1].set_ylabel('Sparsification Overhead (%)', fontsize=10)
        axes[1].set_title('(b) Sparsification Overhead', fontsize=11)
        axes[1].grid(True, alpha=0.3, axis='y', linestyle='--')

        for i, v in enumerate([overhead_no, overhead_with]):
            axes[1].text(i, v, f'{v:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

        # (c) Accuracy
        axes[2].bar(['No Pipeline', 'With Pipeline'],
                   [no_pipe['best_acc'], with_pipe['best_acc']],
                   color=['#3498db', '#e74c3c'], edgecolor='black', linewidth=0.5)
        axes[2].set_ylabel('Best Test Accuracy (%)', fontsize=10)
        axes[2].set_title('(c) Best Accuracy', fontsize=11)
        axes[2].grid(True, alpha=0.3, axis='y', linestyle='--')

        for i, v in enumerate([no_pipe['best_acc'], with_pipe['best_acc']]):
            axes[2].text(i, v, f'{v:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.tight_layout()
        save_path = self._get_save_path('pipeline_comparison')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {save_path}")
        plt.close()

    def print_summary(self, results=None):
        """打印所有实验的性能摘要"""
        if results is None:
            results = self.load_results()

        print(f"\n{'='*80}")
        print("Performance Summary")
        print(f"{'='*80}")
        print(f"{'Experiment':<40s} {'Best Acc':>10s} {'Sparse Time':>12s} {'Comm Time':>12s}")
        print(f"{'-'*80}")

        for result in sorted(results, key=lambda x: x.get('best_acc', 0), reverse=True):
            compressor = result.get('compressor') or 'baseline'
            use_pipeline = result.get('use_pipeline', False)
            label = NAMES.get(compressor, compressor)
            if use_pipeline:
                label += ' (Pipeline)'

            best_acc = result.get('best_acc', 0)
            sparse_time = result.get('avg_sparsification_time', 0)
            comm_time = result.get('avg_communication_time', 0)

            print(f"{label:<40s} {best_acc:>9.2f}% {sparse_time:>11.2f}s {comm_time:>11.2f}s")

        print(f"{'='*80}\n")

    def compare_communication_time(self, results=None):
        """对比通信时间"""
        if results is None:
            results = self.load_results()

        print(f"\n{'='*80}")
        print("Communication Time Comparison")
        print(f"{'='*80}")
        print(f"{'Method':<40s} {'Comm Time':>12s} {'Overhead':>10s}")
        print(f"{'-'*80}")

        for result in sorted(results, key=lambda x: x.get('avg_communication_time', 0)):
            compressor = result.get('compressor') or 'baseline'
            use_pipeline = result.get('use_pipeline', False)
            label = NAMES.get(compressor, compressor)
            if use_pipeline:
                label += ' (Pipeline)'

            comm_time = result.get('avg_communication_time', 0)
            total_time = result.get('avg_epoch_time', 1.0)
            overhead = (comm_time / total_time * 100) if total_time > 0 else 0

            print(f"{label:<40s} {comm_time:>10.2f}s {overhead:>9.1f}%")

        print(f"{'='*80}\n")

    def compare_sparsification_time(self, results=None):
        """对比稀疏化时间"""
        if results is None:
            results = self.load_results()

        print(f"\n{'='*80}")
        print("Sparsification Time Comparison")
        print(f"{'='*80}")
        print(f"{'Method':<40s} {'Sparse Time':>12s} {'Overhead':>10s}")
        print(f"{'-'*80}")

        for result in sorted(results, key=lambda x: x.get('avg_sparsification_time', 0)):
            compressor = result.get('compressor') or 'baseline'
            if not compressor or compressor == 'baseline':
                continue

            use_pipeline = result.get('use_pipeline', False)
            label = NAMES.get(compressor, compressor)
            if use_pipeline:
                label += ' (Pipeline)'

            sparse_time = result.get('avg_sparsification_time', 0)
            total_time = result.get('avg_epoch_time', 1.0)
            overhead = (sparse_time / total_time * 100) if total_time > 0 else 0

            print(f"{label:<40s} {sparse_time:>10.2f}s {overhead:>9.1f}%")

        print(f"{'='*80}\n")

    def generate_report(self, results=None, output_file='performance_report.txt'):
        """生成详细性能报告"""
        if results is None:
            results = self.load_results()

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("HGG-TopK Training Performance Report\n")
            f.write("="*80 + "\n\n")

            f.write(f"Total Experiments: {len(results)}\n")
            f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # 详细信息
            for result in results:
                compressor = result.get('compressor') or 'baseline'
                use_pipeline = result.get('use_pipeline', False)
                label = NAMES.get(compressor, compressor)
                if use_pipeline:
                    label += ' (Pipeline)'

                f.write(f"\n{'-'*80}\n")
                f.write(f"Experiment: {label}\n")
                f.write(f"{'-'*80}\n")
                f.write(f"Model: {result.get('model', 'N/A')}\n")
                f.write(f"Dataset: {result.get('dataset', 'N/A')}\n")
                f.write(f"Compressor: {compressor}\n")
                f.write(f"Density: {result.get('density', 1.0):.3f}\n\n")

                f.write(f"Accuracy:\n")
                f.write(f"  Best: {result.get('best_acc', 0):.2f}%\n")
                f.write(f"  Final: {result.get('final_acc', 0):.2f}%\n\n")

                f.write(f"Time Breakdown (per epoch):\n")
                f.write(f"  Forward:        {result.get('avg_epoch_time', 0):.2f}s\n")
                f.write(f"  Sparsification: {result.get('avg_sparsification_time', 0):.2f}s\n")
                f.write(f"  Communication:  {result.get('avg_communication_time', 0):.2f}s\n")
                f.write(f"  Update:         {result.get('avg_update_time', 0):.2f}s\n\n")

                if compressor and compressor != 'baseline':
                    f.write(f"Compression Statistics:\n")
                    comp_ratios = result.get('compression_ratios', [])
                    if comp_ratios:
                        f.write(f"  Avg Compression Ratio: {np.mean(comp_ratios):.4f}\n")
                    f.write(f"  Avg Threshold Accuracy: {result.get('avg_threshold_accuracy', 0):.4f}\n\n")

        print(f"✓ Report saved: {output_file}\n")

    def generate_all(self):
        """生成所有图表"""
        print("\n" + "="*60)
        print("Generating Visualizations (PNG Format)")
        print(f"Timestamp: {self.timestamp}")
        print("="*60 + "\n")

        results = self.load_results()
        if not results:
            print("⚠ No results found!")
            return

        print("Generating training curves...")
        self.plot_training_curves(results)

        print("Generating performance comparison...")
        self.plot_performance_comparison(results)

        print("Generating pipeline comparison...")
        self.plot_pipeline_comparison(results)

        print(f"\n✓ All figures saved to: {self.output_dir}\n")
        print(f"✓ Timestamp prefix: {self.timestamp}_*.png\n")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Visualize and Analyze Results')
    parser.add_argument('--log-dir', type=str, default='./logs',
                       help='Directory containing JSON result files')
    parser.add_argument('--output-dir', type=str, default='./figures',
                       help='Directory to save PNG figures')
    parser.add_argument('--summary', action='store_true',
                       help='Print performance summary')
    parser.add_argument('--compare-comm', action='store_true',
                       help='Compare communication time')
    parser.add_argument('--compare-sparse', action='store_true',
                       help='Compare sparsification time')
    parser.add_argument('--report', action='store_true',
                       help='Generate detailed report')
    parser.add_argument('--plot', action='store_true',
                       help='Generate all plots')

    args = parser.parse_args()

    viz = Visualizer(log_dir=args.log_dir, output_dir=args.output_dir)

    # 如果没有指定任何选项，默认生成所有图表
    if not any([args.summary, args.compare_comm, args.compare_sparse, args.report, args.plot]):
        viz.generate_all()
    else:
        if args.summary:
            viz.print_summary()
        if args.compare_comm:
            viz.compare_communication_time()
        if args.compare_sparse:
            viz.compare_sparsification_time()
        if args.report:
            viz.generate_report()
        if args.plot:
            viz.generate_all()
