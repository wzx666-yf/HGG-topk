#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
一键运行脚本 - HGG-TopK实验框架

提供简单的命令行界面运行各种实验
"""

import sys
import subprocess


def print_menu():
    """打印菜单"""
    print("\n" + "="*60)
    print("HGG-TopK Training Framework")
    print("="*60)
    print("\n选择实验:")
    print("  [1] 快速测试 (10 epochs, 3个实验, ~30分钟)")
    print("  [2] 对比所有压缩方法 (50 epochs, 6个实验, ~6小时)")
    print("  [3] 流水线对比实验 (50 epochs, 4个实验, ~6小时)")
    print("  [4] 单次训练 (自定义参数)")
    print("  [5] 生成可视化图表")
    print("  [0] 退出")
    print("="*60)


def run_command(cmd):
    """运行命令"""
    print(f"\n执行: {cmd}\n")
    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0


def quick_test():
    """快速测试"""
    return run_command("python experiments/quick_test.py")


def compare_all():
    """对比所有方法"""
    return run_command("python experiments/compare_all_methods.py")


def test_pipeline():
    """测试流水线"""
    return run_command("python experiments/test_pipeline.py")


def single_train():
    """单次训练"""
    print("\n自定义训练参数:")
    print("示例: python trainers/trainer.py --model resnet18 --dataset cifar10 --epochs 50 --compressor hggtopk --density 0.05\n")
    print("可用压缩器: topk, gaussian, dgcsampling, redsync, hggtopk (推荐), randomk\n")

    model = input("Model (resnet18/resnet50/vgg11/vgg16) [resnet18]: ").strip() or 'resnet18'
    dataset = input("Dataset (cifar10/cifar100) [cifar10]: ").strip() or 'cifar10'
    epochs = input("Epochs [100]: ").strip() or '100'
    compressor = input("Compressor (none/topk/gaussian/dgcsampling/redsync/hggtopk) [hggtopk]: ").strip() or 'hggtopk'
    density = input("Density (0.01-1.0) [0.05]: ").strip() or '0.05'
    use_pipeline = input("Use pipeline? (y/n) [n]: ").strip().lower() == 'y'

    cmd = f"python trainers/trainer.py --model {model} --dataset {dataset} --epochs {epochs}"

    if compressor != 'none':
        cmd += f" --compressor {compressor} --density {density}"
        if use_pipeline:
            cmd += " --use-pipeline"

    return run_command(cmd)


def visualize():
    """生成可视化"""
    log_dir = input("Log directory [./logs]: ").strip() or './logs'
    output_dir = input("Output directory [./figures]: ").strip() or './figures'

    return run_command(f"python visualization/visualizer.py --log-dir {log_dir} --output-dir {output_dir}")


def main():
    """主函数"""
    while True:
        print_menu()
        choice = input("\n请选择 (0-5): ").strip()

        if choice == '0':
            print("\n再见!")
            break
        elif choice == '1':
            if quick_test():
                print("\n✓ 快速测试完成!")
            else:
                print("\n✗ 快速测试失败!")
        elif choice == '2':
            if compare_all():
                print("\n✓ 对比实验完成!")
            else:
                print("\n✗ 对比实验失败!")
        elif choice == '3':
            if test_pipeline():
                print("\n✓ 流水线实验完成!")
            else:
                print("\n✗ 流水线实验失败!")
        elif choice == '4':
            if single_train():
                print("\n✓ 训练完成!")
            else:
                print("\n✗ 训练失败!")
        elif choice == '5':
            if visualize():
                print("\n✓ 可视化完成!")
            else:
                print("\n✗ 可视化失败!")
        else:
            print("\n✗ 无效选择，请重试")

        input("\n按Enter继续...")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n中断，退出")
        sys.exit(0)
