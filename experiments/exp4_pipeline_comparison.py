# -*- coding: utf-8 -*-
"""
实验4: HGG-TopK流水线对比
对比使用流水线掩盖梯度压缩时间和不使用流水线的时间差异
"""

import os
import sys
import json
import subprocess
import time

# 实验配置
EXPERIMENTS = {
    'resnet50': {
        'model': 'resnet50',
        'dataset': 'cifar10',
        'batch_size': 64,
        'epochs': 20,
        'lr': 0.1
    },
    'vgg16': {
        'model': 'vgg16',
        'dataset': 'cifar10',
        'batch_size': 64,
        'epochs': 20,
        'lr': 0.01
    },
    'mobilenet': {
        'model': 'mobilenet',
        'dataset': 'cifar10',
        'batch_size': 128,
        'epochs': 20,
        'lr': 0.1
    }
}

# 流水线配置
PIPELINE_CONFIGS = [
    {'name': 'no_pipeline', 'use_pipeline': False},
    {'name': 'with_pipeline', 'use_pipeline': True}
]


def run_experiment(model_config, pipeline_config, log_dir='./logs/exp4'):
    """运行单个实验"""
    model_name = model_config['model']
    config_name = pipeline_config['name']

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
        '--compressor', 'hggtopk',
        '--density', '0.05',
        '--log-dir', log_dir,
        '--gpus', '1'
    ]

    # 添加流水线标志
    if pipeline_config['use_pipeline']:
        cmd.append('--use-pipeline')

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
    print("实验4: HGG-TopK流水线对比")
    print("="*80)

    log_dir = './logs/exp4_pipeline_comparison'
    os.makedirs(log_dir, exist_ok=True)

    results = {}

    # 运行所有实验
    for model_key, model_config in EXPERIMENTS.items():
        results[model_key] = {}

        print(f"\n{'#'*80}")
        print(f"# Model: {model_key}")
        print(f"{'#'*80}")

        for config in PIPELINE_CONFIGS:
            success = run_experiment(model_config, config, log_dir)
            results[model_key][config['name']] = 'Success' if success else 'Failed'

    # 保存结果摘要
    summary_file = os.path.join(log_dir, 'experiment_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)

    # 打印摘要
    print("\n" + "="*80)
    print("实验摘要")
    print("="*80)
    for model_key, config_results in results.items():
        print(f"\n{model_key}:")
        for config_name, status in config_results.items():
            symbol = "✓" if status == 'Success' else "✗"
            print(f"  {symbol} {config_name}: {status}")

    print("\n" + "="*80)
    print("下一步:")
    print("="*80)
    print("1. 生成可视化:")
    print(f"   python experiments/visualize_exp4.py --log-dir {log_dir}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
