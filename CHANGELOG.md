# 更新日志 (CHANGELOG)

## [2.0.0] - 2025-12-03

### 新增 (Added)
- ✨ 新增 6 种梯度压缩算法：
  - `topk2` - TopK（无误差补偿）
  - `gaussian` - 高斯分布压缩
  - `gaussian2` - 高斯分布（无误差补偿）
  - `randomk` - 随机K选择
  - `randomkec` - 随机K + 误差补偿
  - `dgcsampling` - DGC采样压缩
- 📄 新增 `COMPRESSION_UPDATE.md` 详细更新说明
- 📄 新增 `PROJECT_UPDATE_SUMMARY.md` 项目更新总结
- 📄 新增 `CHANGELOG.md` 更新日志

### 修复 (Fixed)
- 🐛 修复所有压缩器的多维张量索引问题
- 🐛 修复 `RuntimeError: selected index k out of range` 错误
- 🐛 修复 Gaussian 压缩器的阈值计算问题

### 改进 (Changed)
- 🔧 统一所有压缩器接口，使用 `ratio` 参数
- 🔧 移除 `settings` 和 `utils` 外部依赖
- 📝 更新 README.md，添加完整的压缩器列表和说明
- 🚀 所有压缩器现在正确处理任意形状的张量

### 移植 (Ported)
- ➕ HGG-TopK 算法已添加到 `D:\python\SGD\compression.py`
- ➕ 原项目现在可以使用 O(N) 复杂度的 HGG-TopK 算法

---

## [1.0.0] - 2024

### 初始版本
- ✨ 实现 HGG-TopK 核心算法
- ✨ 实现异步流水线优化
- ✨ 支持 ResNet, VGG, LSTM 模型
- 📊 详细性能分析和可视化

---

## 版本说明

### 主版本号
- **2.x**: 统一压缩器库版本
- **1.x**: 原始 HGG-TopK 版本

### 向后兼容性
- ✅ 2.0.0 完全向后兼容 1.0.0
- ✅ 所有原有功能保持不变
- ✅ 新增功能不影响现有代码
