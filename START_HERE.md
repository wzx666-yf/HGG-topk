# 🚀 从这里开始

> **新用户？只需3步即可开始！**

---

## ⚡ 快速开始（3步）

### 1️⃣ 安装依赖（1分钟）
```bash
cd D:\python\SGD\HGG-TopK-Training
pip install -r requirements.txt
```

### 2️⃣ 运行快速测试（30分钟）
```bash
python run.py
# 选择 [1] 快速测试
```

### 3️⃣ 查看结果
```
logs/quick_test/          # 训练日志（JSON）
figures/quick_test/       # 可视化图表（PDF）
```

**就这么简单！** ✨

---

## 📚 我想了解更多

### 🎯 我想快速上手
👉 阅读 **QUICKSTART.md**（5分钟）

### 🔬 我想做压缩器对比实验
👉 阅读 **COMPRESSION_UPDATE.md**（技术细节）
```bash
python experiments/compare_all_methods.py
```

### 📁 我想了解项目结构
👉 阅读 **PROJECT_STRUCTURE.md**（项目结构）

### 📖 我想了解所有功能
👉 阅读 **README.md**（完整文档）

---

## 🎓 推荐学习路径

### 路径1：快速验证（30分钟）
```
安装依赖 → 运行 run.py → 选择快速测试 → 查看结果
```

### 路径2：对比实验（6小时）
```
阅读 COMPRESSION_UPDATE.md → 运行 compare_all_methods.py → 生成图表
```

### 路径3：深入研究（自定义）
```
阅读 PROJECT_STRUCTURE.md → 查看代码 → 修改超参数 → 自定义实验
```

---

## 💡 常用命令速查

```bash
# 一键运行（推荐）
python run.py

# 快速测试
python experiments/quick_test.py

# 单次训练
python trainers/trainer.py --model resnet18 --dataset cifar10 \
    --epochs 50 --compressor hggtopk --density 0.05

# 对比所有方法
python experiments/compare_all_methods.py

# 生成图表
python visualization/visualizer.py
```

---

## ❓ 常见问题

**Q: 显存不足？**
```bash
--batch-size 64  # 或更小
```

**Q: 只用部分GPU？**
```bash
--gpus 2  # 或设置 CUDA_VISIBLE_DEVICES=0,1
```

**Q: 训练太慢？**
```bash
--epochs 10  # 先用更少的epochs测试
```

---

## 📦 项目特性

- ✅ **10种压缩算法** - 业界最全
- ✅ **O(N)时间复杂度** - HGG-TopK 最优
- ✅ **分布式训练** - 自动多GPU
- ✅ **异步流水线** - 降低开销50%
- ✅ **一键运行** - 零学习成本
- ✅ **自动可视化** - 科研级图表

---

## 🎯 下一步建议

1. ✅ 运行快速测试验证环境
2. ✅ 阅读 QUICKSTART.md 了解基本用法
3. ✅ 运行对比实验查看效果
4. ✅ 阅读 COMPRESSION_UPDATE.md 深入了解

---

**准备好了吗？运行 `python run.py` 开始吧！** 🚀
