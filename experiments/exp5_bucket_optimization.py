# -*- coding: utf-8 -*-
"""
实验5: HGG-TopK最优分桶数分析
对比不同NUM_BINS值对稀疏化时间和精度的影响
找到最优的分桶数B
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
        'epochs': 20,
        'lr': 0.1
    },
    'vgg11': {
        'model': 'vgg11',
        'dataset': 'cifar10',
        'batch_size': 128,
        'epochs': 20,
        'lr': 0.01
    }
}

# 不同的分桶数配置
BUCKET_CONFIGS = [
    {'name': 'bins_256', 'num_bins': 256},
    {'name': 'bins_512', 'num_bins': 512},
    {'name': 'bins_1024', 'num_bins': 1024},
    {'name': 'bins_2048', 'num_bins': 2048},
    {'name': 'bins_4096', 'num_bins': 4096},
]


def modify_compression_bins(num_bins):
    """修改compression.py中的NUM_BINS值"""
    compression_file = 'core/compression.py'

    with open(compression_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 替换NUM_BINS的值
    import re
    pattern = r'NUM_BINS\s*=\s*\d+'
    new_value = f'NUM_BINS = {num_bins}'

    new_content = re.sub(pattern, new_value, content)

    with open(compression_file, 'w', encoding='utf-8') as f:
        f.write(new_content)

    print(f"  Modified NUM_BINS to {num_bins}")


def run_experiment(model_config, bucket_config, log_dir='./logs/exp5'):
    """运行单个实验"""
    model_name = model_config['model']
    config_name = bucket_config['name']
    num_bins = bucket_config['num_bins']

    print(f"\n{'='*80}")
    print(f"Running: {model_name} + {config_name} (NUM_BINS={num_bins})")
    print(f"{'='*80}\n")

    # 修改NUM_BINS
    modify_compression_bins(num_bins)

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


def restore_default_bins():
    """恢复默认的NUM_BINS值"""
    modify_compression_bins(1024)
    print("\n✓ Restored NUM_BINS to default (1024)")


def main():
    print("\n" + "="*80)
    print("实验5: HGG-TopK最优分桶数分析")
    print("="*80)

    log_dir = './logs/exp5_bucket_optimization'
    os.makedirs(log_dir, exist_ok=True)

    results = {}

    try:
        # 运行所有实验
        for model_key, model_config in EXPERIMENTS.items():
            results[model_key] = {}

            print(f"\n{'#'*80}")
            print(f"# Model: {model_key}")
            print(f"{'#'*80}")

            for bucket_config in BUCKET_CONFIGS:
                success = run_experiment(model_config, bucket_config, log_dir)
                results[model_key][bucket_config['name']] = 'Success' if success else 'Failed'

    finally:
        # 确保恢复默认值
        restore_default_bins()

    # 保存结果摘要
    summary_file = os.path.join(log_dir, 'experiment_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)

    # 打印摘要
    print("\n" + "="*80)
    print("实验摘要")
    print("="*80)
    for model_key, bucket_results in results.items():
        print(f"\n{model_key}:")
        for bucket_name, status in bucket_results.items():
            symbol = "✓" if status == 'Success' else "✗"
            print(f"  {symbol} {bucket_name}: {status}")

    print("\n" + "="*80)
    print("下一步:")
    print("="*80)
    print("1. 生成可视化:")
    print(f"   python experiments/visualize_exp5.py --log-dir {log_dir}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
