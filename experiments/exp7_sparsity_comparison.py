# -*- coding: utf-8 -*-
"""
实验7: 不同稀疏率下的压缩算法对比
比较不同稀疏率下各压缩算法的性能表现
模型: resnet18, resnet50, vgg11, vgg16, lstm, gpt2-small
稀疏率: 1%, 2%, 5%, 10%, 20%
"""

import os
import sys
import json
import subprocess
import time

# 实验配置 - 多个模型
EXPERIMENTS = {
    'resnet18': {
        'model': 'resnet18',
        'dataset': 'cifar10',
        'batch_size': 128,
        'epochs': 20,  # 20轮用于充分训练
        'lr': 0.1
    },
    'resnet50': {
        'model': 'resnet50',
        'dataset': 'cifar10',
        'batch_size': 64,
        'epochs': 20,
        'lr': 0.1
    },
    'vgg11': {
        'model': 'vgg11',
        'dataset': 'cifar10',
        'batch_size': 128,
        'epochs': 20,
        'lr': 0.01
    },
    'vgg16': {
        'model': 'vgg16',
        'dataset': 'cifar10',
        'batch_size': 64,
        'epochs': 20,
        'lr': 0.01
    },
    'lstm': {
        'model': 'lstm',
        'dataset': 'ptb',
        'batch_size': 20,
        'epochs': 15,
        'lr': 20.0
    },
    'gpt2-small': {
        'model': 'gpt2-small',
        'dataset': 'wikitext2',
        'batch_size': 4,
        'epochs': 5,
        'lr': 5e-5,
        'seq_length': 512,
        'log_interval': 100
    }
}

# 稀疏率配置
DENSITIES = [0.01, 0.02, 0.05, 0.10, 0.20]

# 压缩算法配置（不包括baseline）
COMPRESSORS = ['topk', 'gaussian', 'redsync', 'hggtopk']


def run_experiment(model_config, compressor, density, log_dir='./logs/exp7'):
    """运行单个实验"""
    model_name = model_config['model']

    print(f"\n{'='*80}")
    print(f"Running: {model_name} + {compressor} + density={density}")
    print(f"{'='*80}\n")

    # 构建命令
    cmd = [
        sys.executable, 'trainers/trainer.py',
        '--model', model_config['model'],
        '--dataset', model_config['dataset'],
        '--batch-size', str(model_config['batch_size']),
        '--epochs', str(model_config['epochs']),
        '--lr', str(model_config['lr']),
        '--log-dir', log_dir,
        '--gpus', '1',
        '--compressor', compressor,
        '--density', str(density)
    ]

    # 添加序列长度（GPT-2）
    if 'seq_length' in model_config:
        cmd.extend(['--seq-length', str(model_config['seq_length'])])

    # 添加日志间隔（GPT-2）
    if 'log_interval' in model_config:
        cmd.extend(['--log-interval', str(model_config['log_interval'])])

    # 运行实验
    start_time = time.time()
    result = subprocess.run(cmd)
    elapsed_time = time.time() - start_time

    success = result.returncode == 0

    if success:
        print(f"\n✓ Completed in {elapsed_time/60:.1f} minutes")
    else:
        print(f"\n✗ Failed!")

    return success


def main():
    print("\n" + "="*80)
    print("实验7: 不同稀疏率下的压缩算法对比")
    print("="*80)

    log_dir = './logs/exp7_sparsity_comparison'
    os.makedirs(log_dir, exist_ok=True)

    results = {}

    # 统计总实验数
    total_experiments = len(EXPERIMENTS) * len(COMPRESSORS) * len(DENSITIES)
    current_exp = 0

    # 运行所有实验
    for model_key, model_config in EXPERIMENTS.items():
        results[model_key] = {}

        print(f"\n{'#'*80}")
        print(f"# Model: {model_key}")
        print(f"{'#'*80}")

        for density in DENSITIES:
            results[model_key][f'density_{density}'] = {}

            print(f"\n{'-'*80}")
            print(f"Density: {density*100}%")
            print(f"{'-'*80}")

            for compressor in COMPRESSORS:
                current_exp += 1
                print(f"\nProgress: {current_exp}/{total_experiments}")

                success = run_experiment(model_config, compressor, density, log_dir)
                results[model_key][f'density_{density}'][compressor] = 'Success' if success else 'Failed'

    # 保存结果摘要
    summary_file = os.path.join(log_dir, 'experiment_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)

    # 打印摘要
    print("\n" + "="*80)
    print("实验摘要")
    print("="*80)
    for model_key, density_results in results.items():
        print(f"\n{model_key}:")
        for density_key, comp_results in density_results.items():
            print(f"  {density_key}:")
            for comp_name, status in comp_results.items():
                symbol = "✓" if status == 'Success' else "✗"
                print(f"    {symbol} {comp_name}: {status}")

    print("\n" + "="*80)
    print("下一步:")
    print("="*80)
    print("1. 生成可视化:")
    print(f"   python experiments/visualize_exp7.py --log-dir {log_dir}")
    print("\n2. 查看详细结果:")
    print(f"   python visualization/visualizer.py --log-dir {log_dir} --summary")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
