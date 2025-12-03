# HGG-TopK 梯度稀疏化训练框架

> **O(N)时间复杂度的梯度稀疏化算法 + 异步流水线优化**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

## 🎯 核心特性

- **O(N)时间复杂度** - 对数域分桶 + 直方图搜索
- **异步流水线** - 双CUDA流重叠压缩与计算
- **多模型支持** - ResNet, VGG, LSTM
- **详细性能分析** - 时间分解、阈值精度跟踪
- **科研级可视化** - 自动生成论文质量图表

## 📦 安装

```bash
# 克隆项目
cd D:\python\SGD\HGG-TopK-Training

# 安装依赖
pip install -r requirements.txt

# 验证安装
python experiments/quick_test.py
```

## 🚀 快速开始

### 方式1: 一键运行（推荐）

```bash
python run.py
```

选择菜单中的实验即可。

### 方式2: 命令行

```bash
# 快速测试 (10 epochs, ~30分钟)
python experiments/quick_test.py

# Baseline训练
python trainers/trainer.py --model resnet18 --dataset cifar10 --epochs 50

# HGG-TopK (5%稀疏度)
python trainers/trainer.py --model resnet18 --dataset cifar10 --epochs 50 \
    --compressor hggtopk --density 0.05

# HGG-TopK + 流水线
python trainers/trainer.py --model resnet18 --dataset cifar10 --epochs 50 \
    --compressor hggtopk --density 0.05 --use-pipeline

# 生成图表
python visualization/visualizer.py
```

## 📁 项目结构

```
HGG-TopK-Training/
├── README.md                    # 本文档
├── QUICKSTART.md                # 5分钟快速上手
├── requirements.txt             # 依赖列表
├── run.py                       # 一键运行脚本 ⭐
│
├── core/                        # 核心算法
│   ├── compression.py           # 所有压缩算法
│   ├── hgg_pipeline.py          # 异步流水线
│   └── models.py                # LSTM模型定义
│
├── trainers/                    # 训练器
│   └── trainer.py               # 统一训练器 ⭐
│
├── data_utils/                  # 数据处理
│   └── ptb_reader.py            # PTB数据读取
│
├── visualization/               # 可视化
│   └── visualizer.py            # 图表生成 ⭐
│
├── experiments/                 # 实验脚本 ⭐
│   ├── quick_test.py            # 快速测试
│   ├── compare_all_methods.py  # 对比所有压缩方法
│   └── test_pipeline.py         # 流水线对比
│
├── data/                        # 数据目录 (自动创建)
├── logs/                        # 日志目录 (自动创建)
└── figures/                     # 图表目录 (自动创建)
```

## 🧪 预设实验

### 实验1: 快速测试 (~30分钟)

```bash
python experiments/quick_test.py
```

验证环境和代码，运行3个10-epoch实验。

### 实验2: 压缩方法对比 (~5小时)

```bash
python experiments/compare_all_methods.py
```

对比5种方法：Baseline, TopK, Gaussian, RedSync, HGG-TopK。

### 实验3: 流水线对比 (~6小时)

```bash
python experiments/test_pipeline.py
```

对比HGG-TopK的流水线版本与非流水线版本。

### 自定义实验

```bash
python trainers/trainer.py \
    --model resnet50 \
    --dataset cifar10 \
    --epochs 100 \
    --compressor hggtopk \
    --density 0.05 \
    --use-pipeline \
    --batch-size 64
```

## 📊 主要参数

| 参数 | 说明 | 默认值 | 可选值 |
|------|------|--------|--------|
| `--model` | 模型架构 | resnet18 | resnet18/50, vgg11/16, lstm |
| `--dataset` | 数据集 | cifar10 | cifar10, cifar100, ptb |
| `--epochs` | 训练轮数 | 100 | 任意正整数 |
| `--compressor` | 压缩器 | None | topk, topk2, gaussian, gaussian2, randomk, randomkec, dgcsampling, redsync, hggtopk |
| `--density` | 梯度密度 | 1.0 | 0.001~1.0 (推荐0.05) |
| `--use-pipeline` | 使用流水线 | False | 加上此标志启用（仅hggtopk） |
| `--batch-size` | 批大小 | 128 | 根据显存调整 |

### 可用压缩器说明

| 压缩器 | 说明 | 特点 |
|--------|------|------|
| `topk` | 标准 TopK | 带误差补偿 |
| `topk2` | TopK | 无误差补偿 |
| `gaussian` | 高斯分布 | 基于标准差阈值 + 误差补偿 |
| `gaussian2` | 高斯分布 | 基于标准差阈值，无误差补偿 |
| `randomk` | 随机K | 随机选择k个梯度 |
| `randomkec` | 随机K | 随机选择 + 误差补偿 |
| `dgcsampling` | DGC采样 | 基于采样估计阈值 |
| `redsync` | RedSync | 自适应阈值二分搜索 |
| `hggtopk` | HGG-TopK | **O(N)时间复杂度** + 历史引导搜索 |

## 📈 可视化

运行实验后，自动生成以下图表：

1. **training_curves.pdf** - 训练曲线（精度、损失、时间）
2. **performance_comparison.pdf** - 性能对比（6个子图）
3. **pipeline_comparison.pdf** - 流水线对比

```bash
# 生成图表
python visualization/visualizer.py --log-dir logs --output-dir figures
```

## 🎓 HGG-TopK优势

| 指标 | 目标 | 说明 |
|------|------|------|
| **稀疏化开销** | < 5% | TopK通常10-15% |
| **精度保持** | > 95% Baseline | 接近无压缩精度 |
| **阈值精度** | < 1% 相对误差 | 接近真实TopK阈值 |
| **流水线收益** | > 50% 开销降低 | 异步流水线效果 |

## 📖 进阶使用

详细文档请查看：

- **[QUICKSTART.md](QUICKSTART.md)** - 5分钟快速上手指南
- **代码注释** - 每个模块都有详细注释

### 修改超参数

```python
# 在训练前修改HGG-TopK的超参数
from core.compression import HGGTopKCompressor

HGGTopKCompressor.NUM_BINS = 2048  # 直方图桶数（默认1024）
HGGTopKCompressor.GAMMA = 500.0    # 对数缩放因子（默认1000.0）
```

## ❓ 常见问题

**Q: CUDA Out of Memory?**
A: 减小`--batch-size`参数，如`--batch-size 64`或`--batch-size 32`

**Q: 如何使用部分GPU?**
A: 使用`--gpus`参数或`CUDA_VISIBLE_DEVICES`环境变量
```bash
python trainers/trainer.py --gpus 2 ...
# 或
CUDA_VISIBLE_DEVICES=0,1 python trainers/trainer.py ...
```

**Q: 训练太慢?**
A:
- 减少epochs: `--epochs 50`
- 使用更小模型: `--model resnet18`
- 使用更多GPU: `--gpus 4`

**Q: 流水线不生效?**
A: 确保同时使用`--compressor hggtopk`和`--use-pipeline`

## 📧 技术支持

如有问题：
1. 查看本README的常见问题部分
2. 查看代码注释
3. 运行`python experiments/quick_test.py`验证环境

## 📄 引用

```bibtex
@article{hggtopk2024,
  title={HGG-TopK: Efficient Gradient Sparsification via History-Guided Adaptive Galloping Search},
  author={Your Name},
  year={2024}
}
```

## 📜 许可证

Apache 2.0 License

---

**提示**: 首次使用建议运行`python run.py`体验一键运行功能！
