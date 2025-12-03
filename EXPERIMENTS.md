# HGG-TopK 实验指南

本文档详细说明所有实验的设计、运行方式和预期输出。

## 🚀 快速开始

### 方式1: 交互式菜单
```bash
python experiments/run_experiments.py
```
选择要运行的实验编号(1-6)，或选择7运行所有实验。

### 方式2: 直接运行
```bash
# 运行实验
python experiments/exp1_algorithm_comparison.py

# 生成可视化
python experiments/visualize_exp1.py --log-dir ./logs/exp1_algorithm_comparison
```

---

## 📊 实验详情

### 实验1: 算法时间对比

**目的**: 对比不同压缩算法的时间开销分解

**实验配置**:
- **模型**: ResNet50, VGG16, LSTM, GPT2-Medium
- **压缩算法**: Baseline, TopK, Gaussian, RedSync, HGG-TopK
- **压缩率**: 5% (density=0.05)
- **训练轮数**: 1 epoch (快速对比)

**测量指标**:
- Forward时间
- Backward时间
- 稀疏化时间
- 通信时间
- 参数更新时间

**输出文件**:
- `exp1_time_breakdown.png` - 堆叠柱形图，展示每个算法的时间分解
- `exp1_overhead_comparison.png` - 稀疏化和通信开销百分比对比

**运行时间**: ~30-60分钟（取决于GPU）

**命令**:
```bash
python experiments/exp1_algorithm_comparison.py
python experiments/visualize_exp1.py
```

**科研意义**: 展示HGG-TopK在时间开销上的优势，证明稀疏化开销已降至可接受范围

---

### 实验2: HGG-TopK历史阈值 vs 全局二分搜索

**目的**: 验证使用历史阈值(Galloping Search)的优化效果

**实验配置**:
- **模型**: ResNet18, VGG11
- **对比方案**:
  - 使用历史阈值的HGG-TopK (当前实现)
  - 每轮都进行全局二分搜索的HGG-TopK
- **训练轮数**: 20 epochs

**测量指标**:
- 稀疏化时间
- 总训练时间
- 稀疏化开销百分比

**输出文件**:
- `exp2_galloping_vs_binary.png` - 稀疏化时间和总时间对比
- `exp2_overhead.png` - 稀疏化开销百分比对比

**运行时间**: ~40-60分钟

**命令**:
```bash
python experiments/exp2_galloping_vs_binary.py
python experiments/visualize_exp2.py
```

**科研意义**: 证明Galloping Search优化的有效性，展示历史阈值如何加速收敛

---

### 实验3: 精度和损失曲线

**目的**: 对比不同压缩算法对模型精度的影响

**实验配置**:
- **模型**: ResNet50, VGG16, LSTM, GPT2-Medium
- **压缩算法**: Baseline, TopK, Gaussian, RedSync, HGG-TopK
- **训练轮数**:
  - ResNet50/VGG16: 50 epochs
  - LSTM: 40 epochs
  - GPT2-Medium: 10 epochs

**测量指标**:
- 每轮测试精度
- 每轮训练损失
- 最佳测试精度
- (语言模型) 困惑度Perplexity

**输出文件**:
- `{model}_accuracy.png` - 精度曲线图
- `{model}_loss.png` - 损失曲线图
- `{model}_perplexity.png` - 困惑度曲线(仅语言模型)
- `final_accuracy_comparison.png` - 所有模型最终精度对比

**运行时间**: ~3-6小时（最长的实验）

**命令**:
```bash
python experiments/exp3_accuracy_loss_curves.py
python experiments/visualize_exp3.py
```

**科研意义**: 证明压缩算法不会严重损害模型精度，展示HGG-TopK在精度保持上的优势

---

### 实验4: HGG-TopK流水线对比

**目的**: 验证异步流水线掩盖压缩开销的效果

**实验配置**:
- **模型**: ResNet50, VGG16, MobileNet
- **对比方案**:
  - 不使用流水线的HGG-TopK
  - 使用流水线的HGG-TopK (双CUDA流)
- **训练轮数**: 20 epochs

**测量指标**:
- 总训练时间
- 稀疏化+通信时间
- 加速比
- 开销降低百分比

**输出文件**:
- `exp4_pipeline_comparison.png` - 时间分解对比
- `exp4_speedup_analysis.png` - 加速比分析

**运行时间**: ~40-60分钟

**命令**:
```bash
python experiments/exp4_pipeline_comparison.py
python experiments/visualize_exp4.py
```

**科研意义**: 展示流水线优化如何进一步降低通信开销，提升整体训练效率

---

### 实验5: 最优分桶数分析

**目的**: 找到HGG-TopK的最优NUM_BINS参数

**实验配置**:
- **模型**: ResNet18, VGG11
- **分桶数**: 256, 512, 1024, 2048, 4096
- **训练轮数**: 20 epochs

**测量指标**:
- 稀疏化时间
- 最佳测试精度
- 压缩率精确度
- 阈值搜索误差
- 稀疏化开销百分比

**输出文件**:
- `exp5_sparse_time_vs_bins.png` - 稀疏化时间 vs 分桶数
- `exp5_accuracy_vs_bins.png` - 精度 vs 分桶数
- `exp5_compression_quality_vs_bins.png` - 压缩质量分析
- `exp5_overhead_vs_bins.png` - 开销 vs 分桶数
- `bucket_recommendation.txt` - 最优分桶数推荐

**运行时间**: ~1-2小时

**命令**:
```bash
python experiments/exp5_bucket_optimization.py
python experiments/visualize_exp5.py
```

**注意**:
- 实验会临时修改`core/compression.py`中的NUM_BINS值
- 实验结束后会自动恢复为默认值1024

**科研意义**: 为HGG-TopK算法提供参数选择指导，找到速度和精度的最佳平衡点

---

### 实验6: 通信效率分析

**目的**: 分析不同压缩率下的精度-通信量权衡

**实验配置**:
- **模型**: ResNet18
- **压缩率**: 1%, 5%, 10%, 20%, 50%, 100% (Baseline)
- **训练轮数**: 30 epochs

**测量指标**:
- 测试精度
- 通信时间
- 通信量节省百分比
- 精度损失

**输出文件**:
- `exp6_accuracy_vs_density.png` - 精度 vs 压缩率
- `exp6_communication_savings.png` - 通信量节省柱状图
- `exp6_time_vs_density.png` - 训练时间和通信时间 vs 压缩率
- `exp6_tradeoff_analysis.png` - 精度-通信量权衡散点图
- `communication_efficiency_report.txt` - 通信效率报告和推荐配置

**运行时间**: ~1-1.5小时

**命令**:
```bash
python experiments/exp6_communication_efficiency.py
python experiments/visualize_exp6.py
```

**科研意义**: 为实际应用提供压缩率选择指导，展示不同带宽限制下的最佳配置

---

## 📈 实验输出位置

所有实验结果保存在以下位置：

```
logs/
├── exp1_algorithm_comparison/    # 实验1日志和JSON数据
├── exp2_galloping_vs_binary/     # 实验2日志和JSON数据
├── exp3_accuracy_loss_curves/    # 实验3日志和JSON数据
├── exp4_pipeline_comparison/     # 实验4日志和JSON数据
├── exp5_bucket_optimization/     # 实验5日志和JSON数据
└── exp6_communication_efficiency/ # 实验6日志和JSON数据

figures/
├── exp1/    # 实验1可视化图表
├── exp2/    # 实验2可视化图表
├── exp3/    # 实验3可视化图表
├── exp4/    # 实验4可视化图表
├── exp5/    # 实验5可视化图表
└── exp6/    # 实验6可视化图表
```

---

## 🔬 实验组合建议

### 快速验证 (1-2小时)
推荐运行实验1和实验2，快速验证算法性能。
```bash
python experiments/exp1_algorithm_comparison.py && python experiments/visualize_exp1.py
python experiments/exp2_galloping_vs_binary.py && python experiments/visualize_exp2.py
```

### 完整论文实验 (6-10小时)
运行所有6个实验，获得完整的实验数据和图表。
```bash
python experiments/run_experiments.py
# 选择选项 7: 运行所有实验
```

### 参数优化研究 (2-3小时)
专注于实验5和实验6，研究最佳参数配置。
```bash
python experiments/exp5_bucket_optimization.py && python experiments/visualize_exp5.py
python experiments/exp6_communication_efficiency.py && python experiments/visualize_exp6.py
```

---

## 💡 常见问题

**Q: 实验运行失败怎么办?**
A: 检查GPU内存，可以减小batch size或使用更小的模型。

**Q: 如何修改实验配置?**
A: 直接编辑对应的`expN_*.py`文件，修改EXPERIMENTS或COMPRESSORS字典。

**Q: 可以只运行部分模型吗?**
A: 可以，编辑实验脚本，注释掉不需要的模型。

**Q: 如何使用多GPU?**
A: 修改实验脚本中的`--gpus`参数，或设置环境变量`CUDA_VISIBLE_DEVICES`。

**Q: 生成的图表不清晰?**
A: 可以修改visualize脚本中的`dpi`参数，默认300，可提升至600。

---

## 📊 实验结果示例

所有实验完成后，您将获得：

1. **20+张高质量PNG图表** (300 DPI，适合论文)
2. **6个详细的JSON数据文件** (包含所有性能指标)
3. **2-3份文本报告** (实验5和实验6的推荐配置)
4. **完整的训练日志** (可用于进一步分析)

这些结果可以直接用于：
- 学术论文图表
- 技术报告
- 性能对比分析
- 参数调优决策

---

**祝实验顺利！如有问题，请查阅README.md或提交issue。**
