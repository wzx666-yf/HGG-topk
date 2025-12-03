# -*- coding: utf-8 -*-
"""
实验1: 不压缩和各压缩算法对比
比较计算时间、通信时间和稀疏开销
模型: resnet50, vgg16, lstm, gpt2-medium
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
        'epochs': 1,  # 只跑1个epoch用于快速对比
        'lr': 0.1
    },
    'vgg16': {
        'model': 'vgg16',
        'dataset': 'cifar10',
        'batch_size': 64,
        'epochs': 1,
        'lr': 0.01
    },
    'lstm': {
        'model': 'lstm',
        'dataset': 'ptb',
        'batch_size': 20,
        'epochs': 1,
        'lr': 20.0
    },
    'gpt2-medium': {
        'model': 'gpt2-medium',
        'dataset': 'wikitext2',
        'batch_size': 2,
        'epochs': 1,
        'lr': 5e-5,
        'seq_length': 512,
        'log_interval': 50
    }
}

# 压缩算法配置
COMPRESSORS = [
    {'name': 'baseline', 'compressor': None, 'density': 1.0},
    {'name': 'topk', 'compressor': 'topk', 'density': 0.05},
    {'name': 'gaussian', 'compressor': 'gaussian', 'density': 0.05},
    {'name': 'redsync', 'compressor': 'redsync', 'density': 0.05},
    {'name': 'hggtopk', 'compressor': 'hggtopk', 'density': 0.05},
]


def run_experiment(model_config, compressor_config, log_dir='./logs/exp1'):
    """运行单个实验"""
    model_name = model_config['model']
    comp_name = compressor_config['name']

    print(f"\n{'='*80}")
    print(f"Running: {model_name} + {comp_name}")
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

    # 添加序列长度（GPT-2）
    if 'seq_length' in model_config:
        cmd.extend(['--seq-length', str(model_config['seq_length'])])

    # 添加日志间隔（GPT-2）
    if 'log_interval' in model_config:
        cmd.extend(['--log-interval', str(model_config['log_interval'])])

    # 添加压缩器
    if compressor_config['compressor']:
        cmd.extend(['--compressor', compressor_config['compressor']])
        cmd.extend(['--density', str(compressor_config['density'])])

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
    print("实验1: 算法对比实验 - 计算时间、通信时间、稀疏开销")
    print("="*80)

    log_dir = './logs/exp1_algorithm_comparison'
    os.makedirs(log_dir, exist_ok=True)

    results = {}

    # 运行所有实验
    for model_key, model_config in EXPERIMENTS.items():
        results[model_key] = {}

        print(f"\n{'#'*80}")
        print(f"# Model: {model_key}")
        print(f"{'#'*80}")

        for comp_config in COMPRESSORS:
            success = run_experiment(model_config, comp_config, log_dir)
            results[model_key][comp_config['name']] = 'Success' if success else 'Failed'

    # 保存结果摘要
    summary_file = os.path.join(log_dir, 'experiment_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)

    # 打印摘要
    print("\n" + "="*80)
    print("实验摘要")
    print("="*80)
    for model_key, comp_results in results.items():
        print(f"\n{model_key}:")
        for comp_name, status in comp_results.items():
            symbol = "✓" if status == 'Success' else "✗"
            print(f"  {symbol} {comp_name}: {status}")

    print("\n" + "="*80)
    print("下一步:")
    print("="*80)
    print("1. 生成可视化:")
    print(f"   python experiments/visualize_exp1.py --log-dir {log_dir}")
    print("\n2. 查看详细结果:")
    print(f"   python visualization/visualizer.py --log-dir {log_dir} --summary")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
