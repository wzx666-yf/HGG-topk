# -*- coding: utf-8 -*-
"""
可视化工具 - 生成所有分析图表

生成图表:
1. 训练曲线 (精度、损失、时间)
2. 时间分解 (forward/backward/sparsification/communication)
3. 稀疏化开销分析
4. 阈值精度分析
5. 流水线对比
6. 综合对比
"""

import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from typing import Dict, List

# 设置样式
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)

# 配色
COLORS = {
    'baseline': '#2E86AB',
    'topk': '#A23B72',
    'gaussian': '#F18F01',
    'redsync': '#06A77D',
    'hggtopk': '#E63946',
}

NAMES = {
    'baseline': 'Baseline',
    'topk': 'TopK',
    'gaussian': 'Gaussian',
    'redsync': 'RedSync',
    'hggtopk': 'HGG-TopK',
}


class Visualizer:
    """可视化器"""

    def __init__(self, log_dir='./logs', output_dir='./figures'):
        self.log_dir = log_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def load_results(self):
        """加载所有结果"""
        files = glob.glob(os.path.join(self.log_dir, '*.json'))
        results = []
        for file in files:
            with open(file, 'r') as f:
                results.append(json.load(f))
        print(f"✓ Loaded {len(results)} result files")
        return results

    def plot_training_curves(self, results):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        for result in results:
            compressor = result.get('compressor') or 'baseline'
            use_pipeline = result.get('use_pipeline', False)
            label = NAMES.get(compressor, compressor)
            if use_pipeline:
                label += ' (Pipeline)'
            color = COLORS.get(compressor, 'gray')

            # (a) Test Accuracy
            if 'test_accs' in result and result['test_accs']:
                epochs = range(1, len(result['test_accs']) + 1)
                axes[0, 0].plot(epochs, result['test_accs'], label=label, color=color, linewidth=2)

            # (b) Train Loss
            if 'train_losses' in result and result['train_losses']:
                epochs = range(1, len(result['train_losses']) + 1)
                axes[0, 1].plot(epochs, result['train_losses'], label=label, color=color, linewidth=2)

            # (c) Train Accuracy
            if 'train_accs' in result and result['train_accs']:
                epochs = range(1, len(result['train_accs']) + 1)
                axes[1, 0].plot(epochs, result['train_accs'], label=label, color=color, linewidth=2)

            # (d) Time per Epoch
            if 'train_times' in result and result['train_times']:
                epochs = range(1, len(result['train_times']) + 1)
                axes[1, 1].plot(epochs, result['train_times'], label=label, color=color, linewidth=2)

        axes[0, 0].set(xlabel='Epoch', ylabel='Test Accuracy (%)', title='(a) Test Accuracy')
        axes[0, 0].legend(fontsize=9)
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].set(xlabel='Epoch', ylabel='Train Loss', title='(b) Training Loss')
        axes[0, 1].legend(fontsize=9)
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].set(xlabel='Epoch', ylabel='Train Accuracy (%)', title='(c) Training Accuracy')
        axes[1, 0].legend(fontsize=9)
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].set(xlabel='Epoch', ylabel='Time per Epoch (s)', title='(d) Training Time')
        axes[1, 1].legend(fontsize=9)
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'training_curves.pdf')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()

    def plot_performance_comparison(self, results):
        """绘制性能对比"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        methods = []
        best_accs = []
        avg_times = []
        sparse_overheads = []
        threshold_accs = []

        for result in results:
            compressor = result.get('compressor') or 'baseline'
            use_pipeline = result.get('use_pipeline', False)
            label = NAMES.get(compressor, compressor)
            if use_pipeline:
                label += ' (Pipe)'

            methods.append(label)
            best_accs.append(result.get('best_acc', 0))
            avg_times.append(result.get('avg_epoch_time', 0))

            # 稀疏化开销
            if 'avg_sparsification_time' in result and 'avg_epoch_time' in result:
                overhead = result['avg_sparsification_time'] / result['avg_epoch_time'] * 100
                sparse_overheads.append(overhead)
            else:
                sparse_overheads.append(0)

            threshold_accs.append(result.get('avg_threshold_accuracy', 0))

        x = np.arange(len(methods))
        colors = sns.color_palette("husl", len(methods))

        # (a) Best Accuracy
        axes[0, 0].bar(x, best_accs, color=colors)
        axes[0, 0].set(ylabel='Best Test Acc (%)', title='(a) Best Accuracy')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
        axes[0, 0].grid(True, alpha=0.3, axis='y')

        # (b) Training Time
        axes[0, 1].bar(x, avg_times, color=colors)
        axes[0, 1].set(ylabel='Avg Time/Epoch (s)', title='(b) Training Time')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
        axes[0, 1].grid(True, alpha=0.3, axis='y')

        # (c) Sparsification Overhead
        axes[0, 2].bar(x, sparse_overheads, color=colors)
        axes[0, 2].set(ylabel='Sparse Overhead (%)', title='(c) Sparsification Overhead')
        axes[0, 2].set_xticks(x)
        axes[0, 2].set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
        axes[0, 2].grid(True, alpha=0.3, axis='y')

        # (d) Threshold Accuracy
        axes[1, 0].bar(x, threshold_accs, color=colors)
        axes[1, 0].set(ylabel='Threshold Error', title='(d) Threshold Accuracy')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
        axes[1, 0].grid(True, alpha=0.3, axis='y')

        # (e) Time Breakdown (堆叠柱状图)
        if any('forward_times' in r for r in results):
            fwd_times = [np.mean(r.get('forward_times', [0])) for r in results]
            bwd_times = [np.mean(r.get('backward_times', [0])) for r in results]
            sparse_times = [np.mean(r.get('sparsification_times', [0])) for r in results]

            axes[1, 1].bar(x, fwd_times, label='Forward', color='#3498db')
            axes[1, 1].bar(x, bwd_times, bottom=fwd_times, label='Backward', color='#e74c3c')
            axes[1, 1].bar(x, sparse_times, bottom=np.array(fwd_times)+np.array(bwd_times),
                          label='Sparse', color='#f39c12')
            axes[1, 1].set(ylabel='Time (s)', title='(e) Time Breakdown')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
            axes[1, 1].legend(fontsize=8)
            axes[1, 1].grid(True, alpha=0.3, axis='y')

        # (f) Accuracy vs Time
        axes[1, 2].scatter(avg_times, best_accs, s=150, c=colors, alpha=0.7, edgecolors='black', linewidth=2)
        for i, method in enumerate(methods):
            axes[1, 2].annotate(method, (avg_times[i], best_accs[i]), fontsize=7, ha='center', va='bottom')
        axes[1, 2].set(xlabel='Avg Time/Epoch (s)', ylabel='Best Acc (%)', title='(f) Accuracy vs Time')
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'performance_comparison.pdf')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
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

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

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
        axes[0].bar(x - width/2, no_pipe_times, width, label='No Pipeline', color='#3498db')
        axes[0].bar(x + width/2, with_pipe_times, width, label='With Pipeline', color='#e74c3c')
        axes[0].set(ylabel='Time (s)', title='(a) Time Breakdown')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(categories, rotation=20, ha='right')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')

        # (b) Sparsification Overhead
        total_no = sum(no_pipe_times)
        total_with = sum(with_pipe_times)
        overhead_no = no_pipe_times[2] / total_no * 100
        overhead_with = with_pipe_times[2] / total_with * 100

        axes[1].bar(['No Pipeline', 'With Pipeline'], [overhead_no, overhead_with],
                   color=['#3498db', '#e74c3c'])
        axes[1].set(ylabel='Sparse Overhead (%)', title='(b) Sparsification Overhead')
        axes[1].grid(True, alpha=0.3, axis='y')

        for i, v in enumerate([overhead_no, overhead_with]):
            axes[1].text(i, v, f'{v:.1f}%', ha='center', va='bottom', fontsize=11)

        # (c) Accuracy
        axes[2].bar(['No Pipeline', 'With Pipeline'],
                   [no_pipe['best_acc'], with_pipe['best_acc']],
                   color=['#3498db', '#e74c3c'])
        axes[2].set(ylabel='Best Test Acc (%)', title='(c) Accuracy')
        axes[2].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'pipeline_comparison.pdf')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()

    def generate_all(self):
        """生成所有图表"""
        print("\n" + "="*60)
        print("Generating Visualizations")
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


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Visualize Results')
    parser.add_argument('--log-dir', type=str, default='./logs')
    parser.add_argument('--output-dir', type=str, default='./figures')

    args = parser.parse_args()

    viz = Visualizer(log_dir=args.log_dir, output_dir=args.output_dir)
    viz.generate_all()
