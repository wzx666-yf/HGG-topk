# -*- coding: utf-8 -*-
"""
实验8: 不同梯度量下的稀疏化时间对比
测试梯度量（张量大小）不断增加时，各压缩算法的稀疏化时间
重点对比HGG-TopK与其他算法的时间复杂度表现
"""

import os
import json
import time
import torch
import numpy as np
import sys

# 添加项目路径
sys.path.insert(0, os.path.abspath('.'))

from core.compression import (
    TopKCompressor,
    GaussianCompressor,
    RedSyncCompressor,
    HGGTopKCompressor
)


# 梯度量配置（元素数量）- 从小到大递增
GRADIENT_SIZES = [
    10**4,      # 10K
    5 * 10**4,  # 50K
    10**5,      # 100K
    5 * 10**5,  # 500K
    10**6,      # 1M
    5 * 10**6,  # 5M
    10**7,      # 10M
    2 * 10**7,  # 20M
    5 * 10**7,  # 50M
]

# 压缩算法配置
COMPRESSORS = {
    'TopK': TopKCompressor,
    'Gaussian': GaussianCompressor,
    'RedSync': RedSyncCompressor,
    'HGG-TopK': HGGTopKCompressor,
}

# 稀疏率
DENSITY = 0.05

# 预热次数和测试次数
WARMUP_ITERATIONS = 5
TEST_ITERATIONS = 20


def benchmark_compressor(compressor_class, tensor_size, density, warmup=5, iterations=20):
    """
    基准测试单个压缩器
    返回平均稀疏化时间（毫秒）
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建随机梯度张量
    tensor = torch.randn(tensor_size, device=device)

    # 预热
    for _ in range(warmup):
        _ = compressor_class.compress(tensor, name='benchmark', ratio=density)

    # 同步GPU
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # 正式测试
    times = []
    for _ in range(iterations):
        start_time = time.perf_counter()

        _ = compressor_class.compress(tensor, name='benchmark', ratio=density)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start_time
        times.append(elapsed * 1000)  # 转换为毫秒

    avg_time = np.mean(times)
    std_time = np.std(times)

    return avg_time, std_time


def run_experiments():
    """运行所有实验"""
    print("\n" + "="*80)
    print("实验8: 不同梯度量下的稀疏化时间对比")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    print(f"Density: {DENSITY*100}%")
    print(f"Warmup iterations: {WARMUP_ITERATIONS}")
    print(f"Test iterations: {TEST_ITERATIONS}")

    results = {
        'config': {
            'device': str(device),
            'density': DENSITY,
            'warmup_iterations': WARMUP_ITERATIONS,
            'test_iterations': TEST_ITERATIONS,
        },
        'data': {}
    }

    total_tests = len(GRADIENT_SIZES) * len(COMPRESSORS)
    current_test = 0

    # 对每个梯度量进行测试
    for size in GRADIENT_SIZES:
        size_str = format_size(size)
        print(f"\n{'='*80}")
        print(f"Gradient Size: {size_str} ({size:,} elements)")
        print(f"{'='*80}")

        results['data'][size] = {}

        for comp_name, comp_class in COMPRESSORS.items():
            current_test += 1
            print(f"\n[{current_test}/{total_tests}] Testing {comp_name}...")

            try:
                avg_time, std_time = benchmark_compressor(
                    comp_class, size, DENSITY,
                    warmup=WARMUP_ITERATIONS,
                    iterations=TEST_ITERATIONS
                )

                results['data'][size][comp_name] = {
                    'avg_time_ms': float(avg_time),
                    'std_time_ms': float(std_time),
                    'size': size,
                    'size_str': size_str
                }

                print(f"  ✓ {comp_name}: {avg_time:.2f} ± {std_time:.2f} ms")

            except Exception as e:
                print(f"  ✗ {comp_name} failed: {e}")
                results['data'][size][comp_name] = {
                    'error': str(e),
                    'size': size,
                    'size_str': size_str
                }

            # 清理内存
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    # 保存结果
    log_dir = './logs/exp8_gradient_scaling'
    os.makedirs(log_dir, exist_ok=True)

    output_file = os.path.join(log_dir, 'benchmark_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*80)
    print("实验完成")
    print("="*80)
    print(f"\n结果已保存到: {output_file}")

    # 打印摘要表格
    print_summary_table(results)

    print("\n" + "="*80)
    print("下一步:")
    print("="*80)
    print("1. 生成可视化:")
    print(f"   python experiments/visualize_exp8.py --log-dir {log_dir}")
    print("="*80 + "\n")

    return results


def format_size(size):
    """格式化张量大小"""
    if size >= 10**6:
        return f"{size/10**6:.1f}M"
    elif size >= 10**3:
        return f"{size/10**3:.1f}K"
    else:
        return str(size)


def print_summary_table(results):
    """打印结果摘要表格"""
    print("\n" + "="*80)
    print("结果摘要")
    print("="*80)

    # 表头
    header = f"{'Size':<12}"
    for comp_name in COMPRESSORS.keys():
        header += f"{comp_name:<18}"
    print(header)
    print("-" * 80)

    # 数据行
    for size in GRADIENT_SIZES:
        if size not in results['data']:
            continue

        row = f"{format_size(size):<12}"
        for comp_name in COMPRESSORS.keys():
            if comp_name in results['data'][size]:
                data = results['data'][size][comp_name]
                if 'avg_time_ms' in data:
                    time_val = data['avg_time_ms']
                    row += f"{time_val:>8.2f} ms        "
                else:
                    row += f"{'FAILED':<18}"
            else:
                row += f"{'N/A':<18}"

        print(row)

    print("="*80)

    # 计算加速比（相对于TopK）
    if 'TopK' in COMPRESSORS and 'HGG-TopK' in COMPRESSORS:
        print("\nHGG-TopK相对于TopK的加速比:")
        print("-" * 80)
        for size in GRADIENT_SIZES:
            if size not in results['data']:
                continue

            topk_data = results['data'][size].get('TopK', {})
            hgg_data = results['data'][size].get('HGG-TopK', {})

            if 'avg_time_ms' in topk_data and 'avg_time_ms' in hgg_data:
                topk_time = topk_data['avg_time_ms']
                hgg_time = hgg_data['avg_time_ms']
                speedup = topk_time / hgg_time if hgg_time > 0 else 0
                comparison = "faster" if speedup > 1 else "slower"

                print(f"{format_size(size):<12}: {speedup:.2f}x ({comparison})")

        print("="*80)


def main():
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        print("\n⚠ Warning: CUDA not available, running on CPU (will be slower)")
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return

    # 运行实验
    results = run_experiments()


if __name__ == '__main__':
    main()
