# 项目结构说明

## 📁 目录结构

```
HGG-TopK-Training/
│
├── 📄 README.md                    主文档
├── 📄 QUICKSTART.md                快速上手指南
├── 📄 COMPRESSION_UPDATE.md        压缩器更新说明
├── 📄 CHANGELOG.md                 版本更新历史
├── 📄 requirements.txt             依赖列表
├── ⭐ run.py                       一键运行脚本
│
├── 📦 core/                        核心算法
│   ├── compression.py              10种压缩算法实现
│   ├── hgg_pipeline.py             HGG-TopK 异步流水线
│   └── models.py                   LSTM 模型定义
│
├── 📦 trainers/                    训练器
│   └── trainer.py                  统一训练器（支持所有模型和压缩器）
│
├── 📦 data_utils/                  数据处理
│   └── ptb_reader.py               PTB 数据集读取
│
├── 📦 visualization/               可视化
│   └── visualizer.py               自动生成 PDF 图表
│
├── 📦 experiments/                 实验脚本
│   ├── quick_test.py               快速测试（10 epochs）
│   ├── compare_all_methods.py     对比所有压缩方法
│   └── test_pipeline.py            流水线对比实验
│
├── 📂 data/                        数据目录（自动创建）
├── 📂 logs/                        日志目录（自动生成）
└── 📂 figures/                     图表目录（自动生成）
```

## 🎯 核心文件说明

### 1. run.py ⭐
**最简单的使用方式**
- 交互式菜单界面
- 集成所有实验
- 零学习成本

### 2. core/compression.py
**10种压缩算法**
- `hggtopk` - **HGG-TopK** (O(N), 推荐) ⭐
- `topk`, `topk2` - 标准 TopK
- `gaussian`, `gaussian2` - 高斯分布
- `redsync` - RedSync 自适应
- `dgcsampling` - DGC 采样
- `randomk`, `randomkec` - 随机K

### 3. trainers/trainer.py
**统一训练器**
- 支持多种模型：ResNet18/50, VGG11/16, LSTM
- 支持所有压缩算法
- 详细性能测量
- 分布式数据并行（DDP）

### 4. experiments/
**预设实验脚本**
| 脚本 | 时间 | 说明 |
|------|------|------|
| quick_test.py | ~30分钟 | 快速验证环境 |
| compare_all_methods.py | ~6小时 | 对比6种压缩方法 |
| test_pipeline.py | ~6小时 | 流水线效果测试 |

### 5. visualization/visualizer.py
**自动可视化**
- 训练曲线图
- 性能对比图
- 流水线对比图
- 科研级质量（300 DPI）

## 🚀 使用方式

### 方式1：一键运行（推荐）
```bash
python run.py
```
选择菜单中的选项即可。

### 方式2：直接命令
```bash
# 快速测试
python experiments/quick_test.py

# 单次训练
python trainers/trainer.py --model resnet18 --dataset cifar10 \
    --epochs 50 --compressor hggtopk --density 0.05

# 对比实验
python experiments/compare_all_methods.py

# 生成图表
python visualization/visualizer.py
```

## 📊 输出说明

### logs/ - 训练日志
- JSON 格式
- 包含完整训练记录
- 详细性能数据（时间分解、阈值精度等）

### figures/ - 图表输出
- `training_curves.pdf` - 训练曲线
- `performance_comparison.pdf` - 性能对比
- `pipeline_comparison.pdf` - 流水线对比

### data/ - 数据集
- CIFAR-10/100 自动下载
- PTB 数据集

## 📖 文档说明

| 文档 | 用途 | 目标读者 |
|------|------|---------|
| **README.md** | 项目主文档 | 所有用户 |
| **QUICKSTART.md** | 5分钟快速上手 | 新用户 |
| **COMPRESSION_UPDATE.md** | 压缩器技术说明 | 研究者/开发者 |
| **CHANGELOG.md** | 版本更新历史 | 维护者 |
| **PROJECT_STRUCTURE.md** | 项目结构（本文档） | 开发者 |

## 💡 推荐工作流

### 新用户
```bash
1. pip install -r requirements.txt
2. python run.py
3. 选择 [1] 快速测试
4. 查看 logs/ 和 figures/ 目录
```

### 研究者
```bash
1. 运行 python experiments/compare_all_methods.py
2. 生成图表 python visualization/visualizer.py --log-dir logs/comparison
3. 查看 COMPRESSION_UPDATE.md 了解技术细节
4. 自定义超参数后重新实验
```

### 开发者
```bash
1. 查看 core/compression.py 了解算法实现
2. 查看 trainers/trainer.py 了解训练流程
3. 参考 experiments/ 目录设计自己的实验
4. 使用 visualization/visualizer.py 生成图表
```

## ⚙️ 主要参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | resnet18 | resnet18/50, vgg11/16, lstm |
| `--dataset` | cifar10 | cifar10, cifar100, ptb |
| `--epochs` | 100 | 训练轮数 |
| `--compressor` | None | 压缩器选择 |
| `--density` | 1.0 | 梯度密度（0.05推荐） |
| `--use-pipeline` | False | 启用流水线（仅HGG-TopK） |
| `--batch-size` | 128 | 批大小 |
| `--gpus` | auto | GPU数量（自动使用所有） |

## 🔍 目录职责

| 目录 | 职责 | 关键文件 |
|------|------|---------|
| `core/` | 核心算法 | compression.py, hgg_pipeline.py |
| `trainers/` | 训练逻辑 | trainer.py |
| `data_utils/` | 数据处理 | ptb_reader.py |
| `visualization/` | 图表生成 | visualizer.py |
| `experiments/` | 实验脚本 | *.py |
| `logs/` | 训练日志 | *.json |
| `figures/` | 输出图表 | *.pdf |
| `data/` | 数据集 | CIFAR, PTB |

## 📧 获取帮助

1. 查看 **README.md** 完整文档
2. 查看 **QUICKSTART.md** 快速指南
3. 查看代码注释（所有代码都有详细注释）
4. 运行 `python experiments/quick_test.py` 验证环境

---

**项目特点**：清晰的结构 + 统一的接口 + 完整的文档 = 优秀的用户体验 ✨
