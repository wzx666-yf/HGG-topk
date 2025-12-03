#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速测试 - 10 epochs验证环境和代码

测试：
1. Baseline (无压缩)
2. HGG-TopK (5%稀疏度)
3. HGG-TopK Pipeline
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from trainers.trainer import main


def check_environment():
    """检查环境"""
    print("\n" + "="*60)
    print("Environment Check")
    print("="*60)

    print(f"✓ Python: {sys.version.split()[0]}")
    print(f"✓ PyTorch: {torch.__version__}")

    cuda_available = torch.cuda.is_available()
    print(f"{'✓' if cuda_available else '✗'} CUDA: {cuda_available}")

    if cuda_available:
        gpu_count = torch.cuda.device_count()
        print(f"✓ GPUs: {gpu_count}")
        for i in range(gpu_count):
            print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("✗ No GPU available!")
        return False

    print("="*60 + "\n")
    return True


def run_quick_tests():
    """运行快速测试"""
    if not check_environment():
        return False

    tests = [
        {
            'name': 'Test 1: Baseline',
            'args': {
                'model_name': 'resnet18',
                'dataset': 'cifar10',
                'epochs': 10,
                'batch_size': 128,
                'lr': 0.1,
                'log_dir': './logs/quick_test'
            }
        },
        {
            'name': 'Test 2: HGG-TopK',
            'args': {
                'model_name': 'resnet18',
                'dataset': 'cifar10',
                'epochs': 10,
                'batch_size': 128,
                'lr': 0.1,
                'compressor': 'hggtopk',
                'density': 0.05,
                'log_dir': './logs/quick_test'
            }
        },
        {
            'name': 'Test 3: HGG-TopK Pipeline',
            'args': {
                'model_name': 'resnet18',
                'dataset': 'cifar10',
                'epochs': 10,
                'batch_size': 128,
                'lr': 0.1,
                'compressor': 'hggtopk',
                'density': 0.05,
                'use_pipeline': True,
                'log_dir': './logs/quick_test'
            }
        },
    ]

    print("\n" + "="*60)
    print("Quick Tests (10 epochs each)")
    print("="*60 + "\n")

    for i, test in enumerate(tests, 1):
        print(f"\n{'='*60}")
        print(f"{test['name']}")
        print(f"{'='*60}\n")

        try:
            main(**test['args'])
            print(f"\n✓ {test['name']} completed")
        except Exception as e:
            print(f"\n✗ {test['name']} failed: {e}")
            return False

    print("\n" + "="*60)
    print("✓ All tests passed!")
    print("="*60)
    print("\nRun 'python visualization/visualizer.py --log-dir logs/quick_test' to see results\n")

    return True


if __name__ == '__main__':
    success = run_quick_tests()
    sys.exit(0 if success else 1)
