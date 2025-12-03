# 🚀 5分钟快速上手

> 快速开始使用HGG-TopK训练框架

## 第一步: 安装 (1分钟)

```bash
cd D:\python\SGD\HGG-TopK-Training

# 安装依赖
pip install -r requirements.txt
```

## 第二步: 快速测试 (30分钟)

### 方法A: 使用一键脚本（推荐）

```bash
python run.py
```

选择 `[1] 快速测试`

### 方法B: 直接运行

```bash
python experiments/quick_test.py
```

这将运行3个10-epoch的实验：
- ✓ Baseline (无压缩)
- ✓ HGG-TopK (5%稀疏度)
- ✓ HGG-TopK + 流水线

## 第三步: 查看结果 (2分钟)

```bash
# 生成可视化图表
python visualization/visualizer.py --log-dir logs/quick_test --output-dir figures/quick_test
```

打开`figures/quick_test/`目录查看生成的PDF图表。

---

## 常用命令

### 1. 单次训练

```bash
# Baseline
python trainers/trainer.py --model resnet18 --dataset cifar10 --epochs 50

# HGG-TopK
python trainers/trainer.py --model resnet18 --dataset cifar10 --epochs 50 \
    --compressor hggtopk --density 0.05

# HGG-TopK + 流水线
python trainers/trainer.py --model resnet18 --dataset cifar10 --epochs 50 \
    --compressor hggtopk --density 0.05 --use-pipeline
```

### 2. 对比所有压缩方法

```bash
python experiments/compare_all_methods.py
```

### 3. 测试流水线效果

```bash
python experiments/test_pipeline.py
```

### 4. 生成可视化

```bash
python visualization/visualizer.py
```

---

## 快速参数参考

| 参数 | 说明 | 示例 |
|------|------|------|
| `--model` | 模型 | resnet18, resnet50, vgg11, vgg16, lstm |
| `--dataset` | 数据集 | cifar10, cifar100, ptb |
| `--epochs` | 轮数 | 50, 100 |
| `--compressor` | 压缩器 | topk, gaussian, redsync, hggtopk |
| `--density` | 密度 | 0.05 (推荐), 0.01~1.0 |
| `--use-pipeline` | 流水线 | 添加此标志 |
| `--batch-size` | 批大小 | 128 (resnet18), 64 (resnet50) |

---

## 示例场景

### 场景1: 我想快速验证代码

```bash
python experiments/quick_test.py
```

### 场景2: 我想对比HGG-TopK和其他方法

```bash
python experiments/compare_all_methods.py
python visualization/visualizer.py --log-dir logs/comparison
```

### 场景3: 我想测试流水线的效果

```bash
python experiments/test_pipeline.py
python visualization/visualizer.py --log-dir logs/pipeline_comparison
```

### 场景4: 我想训练自己的配置

```bash
# 交互式
python run.py
# 选择 [4] 单次训练

# 或命令行
python trainers/trainer.py \
    --model vgg16 \
    --dataset cifar100 \
    --epochs 100 \
    --compressor hggtopk \
    --density 0.05 \
    --batch-size 64
```

---

## 常见问题速查

**Q: 显存不足?**
```bash
python trainers/trainer.py --batch-size 64 ...  # 或更小
```

**Q: 只用2块GPU?**
```bash
python trainers/trainer.py --gpus 2 ...
```

**Q: 训练太慢，先测试?**
```bash
python trainers/trainer.py --epochs 10 ...  # 减少epochs
```

**Q: 修改超参数?**
```python
from core.compression import HGGTopKCompressor
HGGTopKCompressor.NUM_BINS = 2048
HGGTopKCompressor.GAMMA = 500.0
```

---

## 下一步

- 📖 查看完整文档: `README.md`
- 🔬 运行完整实验: `python experiments/compare_all_methods.py`
- 📊 查看更多可视化: `python visualization/visualizer.py`

祝实验顺利！ 🎉
