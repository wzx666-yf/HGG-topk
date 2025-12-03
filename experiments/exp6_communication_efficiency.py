# -*- coding: utf-8 -*-
"""
实验6: 通信效率分析
对比不同压缩率(density)下的通信量、训练时间和精度
"""

import os
import sys
import json
import subprocess
import time

# 实验配置
EXPERIMENTS = {
    'resnet18': {
        'model': 'resnet18',
        'dataset': 'cifar10',
        'batch_size': 128,
        'epochs': 30,
        'lr': 0.1
    }
}

# 不同压缩率配置
DENSITY_CONFIGS = [
    {'name': 'baseline', 'compressor': None, 'density': 1.0},
    {'name': 'density_01', 'compressor': 'hggtopk', 'density': 0.01},  # 1%
    {'name': 'density_05', 'compressor': 'hggtopk', 'density': 0.05},  # 5%
    {'name': 'density_10', 'compressor': 'hggtopk', 'density': 0.10},  # 10%
    {'name': 'density_20', 'compressor': 'hggtopk', 'density': 0.20},  # 20%
    {'name': 'density_50', 'compressor': 'hggtopk', 'density': 0.50},  # 50%
]


def run_experiment(model_config, density_config, log_dir='./logs/exp6'):
    """运行单个实验"""
    model_name = model_config['model']
    config_name = density_config['name']

    print(f"\n{'='*80}")
    print(f"Running: {model_name} + {config_name}")
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
        '--gpus', '1'
    ]

    # 添加压缩器
    if density_config['compressor']:
        cmd.extend(['--compressor', density_config['compressor']])
        cmd.extend(['--density', str(density_config['density'])])

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
    print("实验6: 通信效率分析 - 不同压缩率对比")
    print("="*80)

    log_dir = './logs/exp6_communication_efficiency'
    os.makedirs(log_dir, exist_ok=True)

    results = {}

    # 运行所有实验
    for model_key, model_config in EXPERIMENTS.items():
        results[model_key] = {}

        print(f"\n{'#'*80}")
        print(f"# Model: {model_key}")
        print(f"{'#'*80}")

        for density_config in DENSITY_CONFIGS:
            success = run_experiment(model_config, density_config, log_dir)
            results[model_key][density_config['name']] = 'Success' if success else 'Failed'

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
        for density_name, status in density_results.items():
            symbol = "✓" if status == 'Success' else "✗"
            print(f"  {symbol} {density_name}: {status}")

    print("\n" + "="*80)
    print("下一步:")
    print("="*80)
    print("1. 生成可视化:")
    print(f"   python experiments/visualize_exp6.py --log-dir {log_dir}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
