# -*- coding: utf-8 -*-
"""
实验总览和一键运行脚本
提供交互式菜单，方便运行各个实验
"""

import os
import sys
import subprocess


def print_header():
    """打印标题"""
    print("\n" + "="*80)
    print(" "*20 + "HGG-TopK实验套件")
    print("="*80)


def print_menu():
    """打印菜单"""
    print("\n可用实验:")
    print("-"*80)
    print("1. 实验1: 算法时间对比 (ResNet50, VGG16, LSTM, GPT2-Medium)")
    print("   - 对比不压缩和各压缩算法的计算时间、通信时间、稀疏开销")
    print("   - 生成堆叠柱形图和开销对比图")
    print()
    print("2. 实验2: HGG-TopK历史阈值 vs 全局二分搜索")
    print("   - 对比使用历史阈值(Galloping)和每轮全局搜索的性能差异")
    print("   - 展示稀疏化时间和总训练时间对比")
    print()
    print("3. 实验3: 精度和损失曲线")
    print("   - 绘制所有模型在不同压缩算法下的精度和损失曲线")
    print("   - 每个模型生成一张精度图和一张损失图")
    print()
    print("4. 实验4: HGG-TopK流水线对比")
    print("   - 对比使用流水线掩盖压缩时间和不使用流水线的差异")
    print("   - 展示加速比和开销降低")
    print()
    print("5. 实验5: 最优分桶数分析")
    print("   - 对比不同NUM_BINS值对性能和精度的影响")
    print("   - 找到最优的分桶数B")
    print()
    print("6. 实验6: 通信效率分析")
    print("   - 对比不同压缩率下的通信量节省和精度权衡")
    print("   - 生成权衡分析图和推荐配置")
    print()
    print("7. 实验7: 不同稀疏率下的压缩算法对比 ⭐NEW")
    print("   - 系统性对比1%, 2%, 5%, 10%, 20%稀疏率下各算法性能")
    print("   - 每个稀疏率独立成图，多模型全面对比")
    print()
    print("8. 实验8: 不同梯度量下的稀疏化时间对比 ⭐NEW")
    print("   - 分析梯度量增加时各算法的时间复杂度表现")
    print("   - 单图展示所有算法的可扩展性")
    print()
    print("9. 运行所有实验 (需要较长时间)")
    print()
    print("0. 退出")
    print("-"*80)


def run_experiment(exp_num):
    """运行指定实验"""
    experiments = {
        1: {
            'name': '实验1: 算法时间对比',
            'runner': 'experiments/exp1_algorithm_comparison.py',
            'visualizer': 'experiments/visualize_exp1.py',
            'log_dir': './logs/exp1_algorithm_comparison'
        },
        2: {
            'name': '实验2: HGG-TopK历史阈值 vs 全局二分',
            'runner': 'experiments/exp2_galloping_vs_binary.py',
            'visualizer': 'experiments/visualize_exp2.py',
            'log_dir': './logs/exp2_galloping_vs_binary'
        },
        3: {
            'name': '实验3: 精度和损失曲线',
            'runner': 'experiments/exp3_accuracy_loss_curves.py',
            'visualizer': 'experiments/visualize_exp3.py',
            'log_dir': './logs/exp3_accuracy_loss_curves'
        },
        4: {
            'name': '实验4: HGG-TopK流水线对比',
            'runner': 'experiments/exp4_pipeline_comparison.py',
            'visualizer': 'experiments/visualize_exp4.py',
            'log_dir': './logs/exp4_pipeline_comparison'
        },
        5: {
            'name': '实验5: 最优分桶数分析',
            'runner': 'experiments/exp5_bucket_optimization.py',
            'visualizer': 'experiments/visualize_exp5.py',
            'log_dir': './logs/exp5_bucket_optimization'
        },
        6: {
            'name': '实验6: 通信效率分析',
            'runner': 'experiments/exp6_communication_efficiency.py',
            'visualizer': 'experiments/visualize_exp6.py',
            'log_dir': './logs/exp6_communication_efficiency'
        },
        7: {
            'name': '实验7: 不同稀疏率下的压缩算法对比',
            'runner': 'experiments/exp7_sparsity_comparison.py',
            'visualizer': 'experiments/visualize_exp7.py',
            'log_dir': './logs/exp7_sparsity_comparison'
        },
        8: {
            'name': '实验8: 不同梯度量下的稀疏化时间对比',
            'runner': 'experiments/exp8_gradient_scaling.py',
            'visualizer': 'experiments/visualize_exp8.py',
            'log_dir': './logs/exp8_gradient_scaling'
        }
    }

    if exp_num not in experiments:
        print("❌ 无效的实验编号!")
        return

    exp = experiments[exp_num]

    print(f"\n{'='*80}")
    print(f"开始运行: {exp['name']}")
    print(f"{'='*80}\n")

    # 运行实验
    print(f"第1步: 运行训练实验...")
    print(f"命令: python {exp['runner']}")
    result = subprocess.run([sys.executable, exp['runner']])

    if result.returncode != 0:
        print(f"\n❌ 实验运行失败!")
        return

    # 生成可视化
    print(f"\n第2步: 生成可视化...")
    print(f"命令: python {exp['visualizer']} --log-dir {exp['log_dir']}")
    result = subprocess.run([sys.executable, exp['visualizer'], '--log-dir', exp['log_dir']])

    if result.returncode != 0:
        print(f"\n⚠ 可视化生成失败!")
        return

    print(f"\n{'='*80}")
    print(f"✓ {exp['name']} 完成!")
    print(f"{'='*80}\n")


def run_all_experiments():
    """运行所有实验"""
    print("\n" + "="*80)
    print("将依次运行所有8个实验，这可能需要10-15小时时间")
    print("="*80)

    choice = input("\n确认继续? (y/n): ")
    if choice.lower() != 'y':
        print("已取消")
        return

    for exp_num in range(1, 9):
        run_experiment(exp_num)
        print("\n" + "-"*80 + "\n")

    print("\n" + "="*80)
    print("✓ 所有实验完成!")
    print("="*80)


def show_quick_commands():
    """显示快速命令"""
    print("\n" + "="*80)
    print("快速命令参考")
    print("="*80)
    print("\n# 单独运行某个实验:")
    print("python experiments/exp1_algorithm_comparison.py")
    print("python experiments/visualize_exp1.py --log-dir ./logs/exp1_algorithm_comparison")
    print("\n# 实验7: 不同稀疏率对比")
    print("python experiments/exp7_sparsity_comparison.py")
    print("python experiments/visualize_exp7.py --log-dir ./logs/exp7_sparsity_comparison")
    print("\n# 实验8: 梯度量可扩展性测试")
    print("python experiments/exp8_gradient_scaling.py")
    print("python experiments/visualize_exp8.py --log-dir ./logs/exp8_gradient_scaling")
    print("\n# 查看结果摘要:")
    print("python visualization/visualizer.py --summary")
    print("\n# 生成完整报告:")
    print("python visualization/visualizer.py --report")
    print("="*80 + "\n")


def main():
    print_header()

    while True:
        print_menu()

        try:
            choice = input("\n请选择实验编号 (0-9, 或输入'h'查看快速命令): ")

            if choice.lower() == 'h':
                show_quick_commands()
                continue

            choice = int(choice)

            if choice == 0:
                print("\n再见!")
                break
            elif choice == 9:
                run_all_experiments()
            elif 1 <= choice <= 8:
                run_experiment(choice)
            else:
                print("❌ 无效的选择，请输入0-9之间的数字")

        except ValueError:
            print("❌ 请输入有效的数字")
        except KeyboardInterrupt:
            print("\n\n用户中断，退出")
            break


if __name__ == '__main__':
    main()
