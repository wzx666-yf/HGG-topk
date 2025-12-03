# -*- coding: utf-8 -*-
"""
GPT-2快速测试脚本
测试HGG-TopK在GPT-2上的性能
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """运行命令并打印输出"""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}\n")

    print(f"Command: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\n❌ Error: {description} failed!")
        return False
    else:
        print(f"\n✓ {description} completed successfully!")
        return True


def main():
    print("\n" + "="*80)
    print("GPT-2 + HGG-TopK Performance Test")
    print("="*80)

    tests = [
        {
            'cmd': [
                sys.executable, 'trainers/trainer.py',
                '--model', 'gpt2-small',
                '--dataset', 'wikitext2',
                '--batch-size', '4',
                '--epochs', '2',
                '--log-interval', '50',
                '--gpus', '1'
            ],
            'desc': 'Test 1: GPT-2 Small Baseline (2 epochs)'
        },
        {
            'cmd': [
                sys.executable, 'trainers/trainer.py',
                '--model', 'gpt2-small',
                '--dataset', 'wikitext2',
                '--compressor', 'topk',
                '--density', '0.05',
                '--batch-size', '4',
                '--epochs', '2',
                '--log-interval', '50',
                '--gpus', '1'
            ],
            'desc': 'Test 2: GPT-2 Small + TopK (5% density, 2 epochs)'
        },
        {
            'cmd': [
                sys.executable, 'trainers/trainer.py',
                '--model', 'gpt2-small',
                '--dataset', 'wikitext2',
                '--compressor', 'hggtopk',
                '--density', '0.05',
                '--batch-size', '4',
                '--epochs', '2',
                '--log-interval', '50',
                '--gpus', '1'
            ],
            'desc': 'Test 3: GPT-2 Small + HGG-TopK (5% density, 2 epochs)'
        },
    ]

    results = {}

    for test in tests:
        success = run_command(test['cmd'], test['desc'])
        results[test['desc']] = 'Success' if success else 'Failed'

    # 总结
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    for desc, status in results.items():
        symbol = "✓" if status == 'Success' else "✗"
        print(f"{symbol} {desc}: {status}")

    print("\n" + "="*80)
    print("Next Steps")
    print("="*80)
    print("1. View results:")
    print("   python visualization/visualizer.py --summary")
    print("\n2. Compare performance:")
    print("   python visualization/visualizer.py --compare-comm --compare-sparse")
    print("\n3. Generate report:")
    print("   python visualization/visualizer.py --report --plot")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
