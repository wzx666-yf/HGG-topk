#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对比所有压缩方法

运行：
1. Baseline (无压缩)
2. TopK
3. Gaussian
4. RedSync
5. HGG-TopK
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainers.trainer import main


def main_comparison():
    """对比所有压缩方法"""
    print("\n" + "="*60)
    print("Compression Methods Comparison")
    print("="*60 + "\n")

    # 配置
    model = 'resnet18'
    dataset = 'cifar10'
    epochs = 50
    density = 0.05
    log_dir = './logs/comparison'

    experiments = [
        {'name': 'Baseline', 'compressor': None},
        {'name': 'TopK', 'compressor': 'topk'},
        {'name': 'Gaussian', 'compressor': 'gaussian'},
        {'name': 'RedSync', 'compressor': 'redsync'},
        {'name': 'HGG-TopK', 'compressor': 'hggtopk'},
    ]

    for exp in experiments:
        print(f"\n{'='*60}")
        print(f"Running: {exp['name']}")
        print(f"{'='*60}\n")

        args = {
            'model_name': model,
            'dataset': dataset,
            'epochs': epochs,
            'batch_size': 128,
            'lr': 0.1,
            'log_dir': log_dir
        }

        if exp['compressor']:
            args['compressor'] = exp['compressor']
            args['density'] = density

        try:
            main(**args)
            print(f"\n✓ {exp['name']} completed")
        except Exception as e:
            print(f"\n✗ {exp['name']} failed: {e}")

    print("\n" + "="*60)
    print("✓ All experiments completed!")
    print("="*60)
    print(f"\nResults saved to: {log_dir}")
    print("Run 'python visualization/visualizer.py --log-dir logs/comparison' to visualize\n")


if __name__ == '__main__':
    main_comparison()
