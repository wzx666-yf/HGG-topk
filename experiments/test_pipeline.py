#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
流水线对比实验

对比HGG-TopK的流水线版本与非流水线版本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainers.trainer import main


def pipeline_comparison():
    """流水线对比"""
    print("\n" + "="*60)
    print("HGG-TopK Pipeline Comparison")
    print("="*60 + "\n")

    # 配置
    configs = [
        ('resnet18', 50),
        ('resnet50', 50),
    ]

    log_dir = './logs/pipeline_comparison'

    for model, epochs in configs:
        batch_size = 128 if model == 'resnet18' else 64

        # 不使用流水线
        print(f"\n{'='*60}")
        print(f"Running: {model} WITHOUT pipeline")
        print(f"{'='*60}\n")

        try:
            main(
                model_name=model,
                dataset='cifar10',
                epochs=epochs,
                batch_size=batch_size,
                lr=0.1,
                compressor='hggtopk',
                density=0.05,
                use_pipeline=False,
                log_dir=log_dir
            )
            print(f"\n✓ {model} (no pipeline) completed")
        except Exception as e:
            print(f"\n✗ {model} (no pipeline) failed: {e}")

        # 使用流水线
        print(f"\n{'='*60}")
        print(f"Running: {model} WITH pipeline")
        print(f"{'='*60}\n")

        try:
            main(
                model_name=model,
                dataset='cifar10',
                epochs=epochs,
                batch_size=batch_size,
                lr=0.1,
                compressor='hggtopk',
                density=0.05,
                use_pipeline=True,
                log_dir=log_dir
            )
            print(f"\n✓ {model} (with pipeline) completed")
        except Exception as e:
            print(f"\n✗ {model} (with pipeline) failed: {e}")

    print("\n" + "="*60)
    print("✓ Pipeline comparison completed!")
    print("="*60)
    print(f"\nResults saved to: {log_dir}")
    print("Run 'python visualization/visualizer.py --log-dir logs/pipeline_comparison' to visualize\n")


if __name__ == '__main__':
    pipeline_comparison()
