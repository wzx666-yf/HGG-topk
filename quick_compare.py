# -*- coding: utf-8 -*-
"""
快速对比脚本 - 一键运行关键压缩方法对比实验
"""

import os
import sys
import argparse
import subprocess
import time


def run_experiment(model, dataset, compressor, density, epochs, gpus):
    """运行单个实验"""
    cmd = [
        sys.executable, 'trainers/trainer.py',
        '--model', model,
        '--dataset', dataset,
        '--epochs', str(epochs),
        '--gpus', str(gpus)
    ]

    if compressor:
        cmd.extend(['--compressor', compressor])
        cmd.extend(['--density', str(density)])

    print(f"\n{'='*80}")
    exp_name = f"{compressor or 'baseline'} (d={density if compressor else 1.0})"
    print(f"Running: {exp_name}")
    print(f"{'='*80}\n")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n⚠ Warning: Experiment {exp_name} failed!")

    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description='Quick Comparison Script')
    parser.add_argument('--model', type=str, default='resnet18',
                       choices=['resnet18', 'resnet50', 'vgg11', 'vgg16', 'mobilenet'],
                       help='Model to train')
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'cifar100'],
                       help='Dataset to use')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of epochs (default: 5 for quick test)')
    parser.add_argument('--density', type=float, default=0.05,
                       help='Compression density (default: 0.05 = 5%%)')
    parser.add_argument('--gpus', type=int, default=None,
                       help='Number of GPUs to use')
    parser.add_argument('--methods', type=str, nargs='+',
                       default=['baseline', 'topk', 'hggtopk'],
                       help='Compression methods to compare')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("HGG-TopK Quick Comparison")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.epochs}")
    print(f"Density: {args.density}")
    print(f"Methods: {', '.join(args.methods)}")
    print("="*80 + "\n")

    start_time = time.time()
    results = {}

    for method in args.methods:
        compressor = None if method == 'baseline' else method
        success = run_experiment(
            args.model, args.dataset, compressor,
            args.density, args.epochs, args.gpus
        )
        results[method] = 'Success' if success else 'Failed'

    # 总结
    total_time = time.time() - start_time
    print("\n" + "="*80)
    print("Experiment Summary")
    print("="*80)
    print(f"Total Time: {total_time/60:.1f} minutes\n")

    for method, status in results.items():
        symbol = "✓" if status == 'Success' else "✗"
        print(f"{symbol} {method}: {status}")

    print("\n" + "="*80)
    print("Next Steps:")
    print("="*80)
    print("1. View results:")
    print("   python visualization/visualizer.py --summary")
    print("\n2. Compare performance:")
    print("   python visualization/visualizer.py --compare-comm --compare-sparse")
    print("\n3. Generate report and plots:")
    print("   python visualization/visualizer.py --report --plot")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
