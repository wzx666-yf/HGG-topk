# 实验7和实验8扩展说明

## 新增实验概览

本次扩展为 HGG-TopK-Training 项目添加了两个重要的新实验：

### 实验7: 不同稀疏率下的压缩算法对比

**特点**:
- ✅ 系统性对比5个不同稀疏率（1%, 2%, 5%, 10%, 20%）
- ✅ 测试6个模型（ResNet18/50, VGG11/16, LSTM, GPT2-Small）
- ✅ 每个稀疏率独立生成一张图表，包含4个子图
- ✅ 额外生成汇总热力图和效率权衡分析
- ✅ 符合科研论文发表标准

**输出图表**:
1. `exp7_sparsity_1percent.png` - 1%稀疏率的对比图
2. `exp7_sparsity_2percent.png` - 2%稀疏率的对比图
3. `exp7_sparsity_5percent.png` - 5%稀疏率的对比图
4. `exp7_sparsity_10percent.png` - 10%稀疏率的对比图
5. `exp7_sparsity_20percent.png` - 20%稀疏率的对比图
6. `exp7_summary_heatmap.png` - 精度汇总热力图
7. `exp7_efficiency_tradeoff.png` - 效率权衡分析

**每个稀疏率图表包含**:
- (a) 训练时间对比
- (b) 稀疏化开销对比
- (c) 通信开销对比
- (d) 测试精度对比

### 实验8: 不同梯度量下的稀疏化时间对比

**特点**:
- ✅ 测试9个不同梯度量级别（10K到50M元素）
- ✅ 纯基准测试，精确测量稀疏化时间
- ✅ 包含预热阶段和多次重复测试
- ✅ 所有算法在同一图中展示，便于对比
- ✅ 采用对数坐标，清晰展示可扩展性

**输出图表**:
1. `exp8_scaling_performance.png` - 主图：时间-梯度量曲线（对数坐标）
2. `exp8_speedup_comparison.png` - 相对TopK的加速比对比
3. `exp8_complexity_analysis.png` - 时间复杂度分析（归一化）
4. `exp8_summary_report.txt` - 详细数值报告

## 科研风格设计

### 配色方案
- **TopK**: 红色 (#E74C3C)
- **Gaussian**: 蓝色 (#3498DB)
- **RedSync**: 橙色 (#F39C12)
- **HGG-TopK**: 绿色 (#2ECC71)

### 图表特性
- 300 DPI 高分辨率
- Times New Roman 字体
- 清晰的网格线和边框
- 标准化的图例和标签
- 专业的误差带显示
- 适合直接用于论文发表

## 运行指南

### 实验7 - 稀疏率对比
```bash
# 运行实验（预计4-8小时）
python experiments/exp7_sparsity_comparison.py

# 生成可视化
python experiments/visualize_exp7.py --log-dir ./logs/exp7_sparsity_comparison
```

### 实验8 - 梯度量可扩展性
```bash
# 运行实验（预计30-60分钟）
python experiments/exp8_gradient_scaling.py

# 生成可视化
python experiments/visualize_exp8.py --log-dir ./logs/exp8_gradient_scaling
```

### 使用交互式菜单
```bash
python experiments/run_experiments.py
# 选择 7 运行实验7
# 选择 8 运行实验8
# 选择 9 运行所有实验（包括新增的）
```

## 实验配置详情

### 实验7配置
- **ResNet18/50**: 128/64 batch size, 20 epochs
- **VGG11/16**: 128/64 batch size, 20 epochs
- **LSTM**: 20 batch size, 15 epochs
- **GPT2-Small**: 4 batch size, 5 epochs
- **稀疏率**: 1%, 2%, 5%, 10%, 20%
- **压缩算法**: TopK, Gaussian, RedSync, HGG-TopK

### 实验8配置
- **梯度量**: 10K, 50K, 100K, 500K, 1M, 5M, 10M, 20M, 50M
- **稀疏率**: 5%（固定）
- **预热**: 5次迭代
- **测试**: 20次迭代取平均
- **设备**: 自动检测GPU/CPU

## 预期结果

### 实验7预期展示
1. **低稀疏率（1%-2%）**:
   - 通信量最小，但可能影响精度
   - HGG-TopK在稀疏化时间上仍保持优势

2. **中等稀疏率（5%-10%）**:
   - 精度和通信量的最佳平衡点
   - 各算法性能差异最明显

3. **高稀疏率（20%）**:
   - 精度接近baseline
   - 通信开销降低有限

### 实验8预期展示
1. **HGG-TopK**: 时间复杂度接近O(n)，可扩展性最佳
2. **TopK**: 标准实现，作为baseline对比
3. **Gaussian/RedSync**: 中等性能，随梯度量增加而增长

## 科研应用

这两个实验的图表可以直接用于：

1. **学术论文**:
   - 多维度性能分析
   - 算法可扩展性验证
   - 参数敏感性研究

2. **技术报告**:
   - 不同场景下的算法选择指导
   - 性能优化建议

3. **会议演示**:
   - 清晰的视觉对比
   - 专业的科研风格

## 文件结构

```
HGG-TopK-Training/
├── experiments/
│   ├── exp7_sparsity_comparison.py      # 实验7主脚本
│   ├── visualize_exp7.py                # 实验7可视化
│   ├── exp8_gradient_scaling.py         # 实验8主脚本
│   ├── visualize_exp8.py                # 实验8可视化
│   └── run_experiments.py               # 已更新：包含实验7和8
├── logs/
│   ├── exp7_sparsity_comparison/        # 实验7日志
│   └── exp8_gradient_scaling/           # 实验8日志
├── figures/
│   ├── exp7/                            # 实验7图表（7张图）
│   └── exp8/                            # 实验8图表（3张图+1份报告）
└── EXPERIMENTS.md                       # 已更新：包含实验7和8说明
```

## 注意事项

1. **GPU要求**: 实验7需要GPU进行完整训练，实验8可在CPU上运行但建议使用GPU
2. **时间估算**: 实验7约4-8小时，实验8约30-60分钟
3. **存储空间**: 确保有足够空间存储日志和模型检查点
4. **依赖检查**: 确保已安装所有requirements.txt中的依赖

## 问题排查

### CUDA Out of Memory
- 减小batch size
- 减少测试的模型数量
- 使用更小的模型（如ResNet18代替ResNet50）

### 实验运行失败
- 检查GPU可用性
- 确认数据集已正确下载
- 查看日志文件获取详细错误信息

## 更新日志

- 2025-12-04: 添加实验7和实验8
- 更新EXPERIMENTS.md文档
- 更新run_experiments.py交互式菜单
- 所有图表符合科研论文发表标准

---

**享受新实验！如有任何问题，请查阅EXPERIMENTS.md或提交issue。**
