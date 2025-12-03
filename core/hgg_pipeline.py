# -*- coding: utf-8 -*-
"""
HGG-TopK 异步流水线实现

双流架构：
- 主流 (Stream A): 处理反向传播计算
- 通信流 (Stream B): 处理梯度压缩和通信

流水线重叠压缩与计算，隐藏通信开销
"""
from __future__ import print_function
import torch
import torch.cuda as cuda
import numpy as np
import time
from compression import HGGTopKCompressor


class AsyncCompressionTask:
    """异步压缩任务"""
    def __init__(self, tensor, name, ratio, device):
        self.tensor = tensor
        self.name = name
        self.ratio = ratio
        self.device = device
        self.indexes = None
        self.values = None
        self.compressed = False
        self.event = cuda.Event() if cuda.is_available() else None


class HGGPipelineCompressor:
    """
    异步流水线压缩器 - 使用双CUDA流

    架构:
    - 主流: 反向传播立即继续
    - 通信流: 梯度统计、压缩、通信并行进行

    通信流中的流水线阶段:
    1. Max & 直方图计算 (GPU, 访存密集)
    2. CPU阈值搜索 (CPU, 计算密集)
    3. 压缩内核 (GPU, 访存密集)
    4. 通信 (网络)
    """

    def __init__(self, enable_pipeline=True):
        """
        初始化流水线压缩器

        Args:
            enable_pipeline: 如果为True，使用双流流水线；否则同步运行
        """
        self.enable_pipeline = enable_pipeline
        self.comm_stream = None
        self.main_stream = None

        if self.enable_pipeline and cuda.is_available():
            # 创建通信流用于压缩和通信
            self.comm_stream = cuda.Stream()
            # 主流是默认流 (Stream 0)
            self.main_stream = cuda.current_stream()
            print('[HGG Pipeline] Initialized with dual CUDA streams')
        else:
            print('[HGG Pipeline] Running in synchronous mode')

        # 任务队列
        self.active_tasks = {}

        # 压缩统计
        self.compression_times = {}
        self.histogram_times = {}
        self.search_times = {}
        self.mask_times = {}

    def _compute_max_and_histogram_fused(self, abs_values, num_bins, gamma, stream=None):
        """
        融合内核：单次遍历计算最大值和直方图

        Args:
            abs_values: 梯度绝对值
            num_bins: 直方图桶数
            gamma: 对数映射缩放因子
            stream: 使用的CUDA流

        Returns:
            max_val: 最大绝对值
            bin_indices: 每个梯度的桶索引
        """
        with cuda.stream(stream) if stream else torch.no_grad():
            # 计算最大值
            max_val = torch.max(abs_values).item()

            # 使用对数域映射计算桶索引
            if max_val < 1e-10:
                bin_indices = torch.zeros_like(abs_values, dtype=torch.long)
            else:
                import math
                numerator = torch.log(1.0 + gamma * abs_values)
                denominator = math.log(1.0 + gamma * max_val)
                bin_indices = torch.floor(num_bins * numerator / denominator)
                bin_indices = torch.clamp(bin_indices, 0, num_bins - 1).long()

        return max_val, bin_indices

    def _async_compress_worker(self, task):
        """
        通信流中的异步压缩工作函数

        在通信流(Stream B)中运行，主流(Stream A)继续反向传播

        流水线阶段:
        1. 融合 Max + 直方图内核 (GPU, 访存密集)
        2. 同步直方图到CPU (DMA传输)
        3. CPU阈值搜索 (CPU, 计算密集, 与GPU重叠)
        4. 压缩内核 (GPU, 访存密集)

        Args:
            task: 要处理的AsyncCompressionTask
        """
        start_time = time.time()

        with torch.no_grad():
            # 设置当前流为通信流
            if self.comm_stream:
                cuda.set_stream(self.comm_stream)

            tensor = task.tensor
            name = task.name
            ratio = task.ratio

            # 阶段1: 融合Max和直方图计算
            hist_start = time.time()

            abs_values = torch.abs(tensor.data)
            max_val, bin_indices = self._compute_max_and_histogram_fused(
                abs_values,
                HGGTopKCompressor.NUM_BINS,
                HGGTopKCompressor.GAMMA,
                stream=self.comm_stream
            )

            # 构建直方图和后缀和
            histogram, suffix_sum = HGGTopKCompressor._build_histogram(
                bin_indices, HGGTopKCompressor.NUM_BINS
            )

            hist_time = time.time() - hist_start

            # 阶段2&3: CPU阈值搜索
            search_start = time.time()

            numel = tensor.numel()
            k = max(int(numel * ratio), 1)
            tolerance = max(int(k * HGGTopKCompressor.TOLERANCE), 1)

            # 获取该层的迭代计数
            if name not in HGGTopKCompressor.iteration_count:
                HGGTopKCompressor.iteration_count[name] = 0
            iteration = HGGTopKCompressor.iteration_count[name]

            # 混合阈值搜索策略
            if iteration == 0 or name not in HGGTopKCompressor.prev_thresholds:
                # 冷启动: 完整二分搜索
                bin_idx = HGGTopKCompressor._binary_search_threshold(
                    suffix_sum, k, tolerance
                )
            else:
                # 历史引导的倍增搜索
                prev_threshold = HGGTopKCompressor.prev_thresholds[name]

                if max_val > 1e-10:
                    import math
                    numerator = math.log(1.0 + HGGTopKCompressor.GAMMA * prev_threshold)
                    denominator = math.log(1.0 + HGGTopKCompressor.GAMMA * max_val)
                    prev_idx = int(HGGTopKCompressor.NUM_BINS * numerator / denominator)
                    prev_idx = max(0, min(HGGTopKCompressor.NUM_BINS - 1, prev_idx))
                else:
                    prev_idx = 0

                bin_idx = HGGTopKCompressor._galloping_search(
                    suffix_sum, k, prev_idx, tolerance
                )

            search_time = time.time() - search_start

            # 计算最终阈值（带插值）
            final_threshold = HGGTopKCompressor._compute_final_threshold(
                bin_idx, abs_values, bin_indices, max_val, k,
                HGGTopKCompressor.NUM_BINS, HGGTopKCompressor.GAMMA,
                HGGTopKCompressor.BETA
            )

            # 阶段4: 压缩内核 (GPU, 访存密集)
            mask_start = time.time()

            mask = abs_values >= final_threshold
            indexes = mask.nonzero().squeeze().view(-1)

            # 确保不超过k
            if indexes.numel() > k:
                selected_abs_values = abs_values[indexes]
                _, topk_indices = torch.topk(selected_abs_values, k)
                indexes = indexes[topk_indices]

            values = tensor.data[indexes]

            mask_time = time.time() - mask_start

            # 更新压缩统计
            task.indexes = indexes
            task.values = values
            task.compressed = True

            # 更新迭代计数和阈值历史
            HGGTopKCompressor.prev_thresholds[name] = final_threshold
            HGGTopKCompressor.iteration_count[name] += 1

            # 记录性能数据
            if name not in self.compression_times:
                self.compression_times[name] = []
                self.histogram_times[name] = []
                self.search_times[name] = []
                self.mask_times[name] = []

            total_time = time.time() - start_time
            self.compression_times[name].append(total_time)
            self.histogram_times[name].append(hist_time)
            self.search_times[name].append(search_time)
            self.mask_times[name].append(mask_time)

            # 记录事件用于同步
            if self.comm_stream and task.event:
                task.event.record(self.comm_stream)

            # 恢复默认流
            if self.comm_stream:
                cuda.set_stream(self.main_stream)

    def compress_async(self, tensor, name, ratio=0.05):
        """
        异步压缩梯度张量

        在通信流中启动压缩任务后立即返回，允许主流继续BP

        Args:
            tensor: 要压缩的梯度张量
            name: 层名称
            ratio: 压缩率

        Returns:
            task: AsyncCompressionTask句柄，用于后续同步
        """
        if not self.enable_pipeline or not cuda.is_available():
            # 回退到同步压缩
            _, indexes, values = HGGTopKCompressor.compress(tensor, name, ratio=ratio)
            task = AsyncCompressionTask(tensor, name, ratio, tensor.device)
            task.indexes = indexes
            task.values = values
            task.compressed = True
            return task

        # 创建异步任务
        task = AsyncCompressionTask(tensor, name, ratio, tensor.device)

        # 在通信流中启动异步压缩
        self._async_compress_worker(task)

        # 存储活跃任务
        self.active_tasks[name] = task

        return task

    def synchronize(self, task):
        """
        同步并等待压缩任务完成

        在通信前调用以确保压缩完成

        Args:
            task: 要同步的AsyncCompressionTask

        Returns:
            indexes: 选中梯度的索引
            values: 选中梯度的值
        """
        if not task.compressed:
            # 等待任务完成
            if self.comm_stream and task.event:
                task.event.synchronize()

        # 从活跃任务中移除
        if task.name in self.active_tasks:
            del self.active_tasks[task.name]

        return task.indexes, task.values

    def update_residuals(self, tensor, indexes, name):
        """
        压缩后更新残差

        Args:
            tensor: 原始梯度张量
            indexes: 传输梯度的索引
            name: 层名称
        """
        with torch.no_grad():
            if name not in HGGTopKCompressor.residuals:
                HGGTopKCompressor.residuals[name] = torch.zeros_like(tensor.data)

            # 更新残差: 保留未传输的梯度
            HGGTopKCompressor.residuals[name].data = tensor.data.clone()
            HGGTopKCompressor.residuals[name].data[indexes] = 0.0

    def print_profiling(self, rank=0):
        """
        打印压缩流水线的性能统计

        Args:
            rank: 进程rank (用于分布式训练)
        """
        if len(self.compression_times) == 0:
            return

        print('='*80)
        print(f'HGG Pipeline Profiling Statistics (Rank {rank})')
        print('='*80)

        for name in sorted(self.compression_times.keys()):
            if len(self.compression_times[name]) == 0:
                continue

            total_avg = np.mean(self.compression_times[name]) * 1000  # ms
            hist_avg = np.mean(self.histogram_times[name]) * 1000
            search_avg = np.mean(self.search_times[name]) * 1000
            mask_avg = np.mean(self.mask_times[name]) * 1000

            print(f'Layer: {name}')
            print(f'  Total compression time: {total_avg:.3f} ms')
            print(f'  - Histogram construction: {hist_avg:.3f} ms ({100*hist_avg/total_avg:.1f}%)')
            print(f'  - Threshold search (CPU): {search_avg:.3f} ms ({100*search_avg/total_avg:.1f}%)')
            print(f'  - Mask generation: {mask_avg:.3f} ms ({100*mask_avg/total_avg:.1f}%)')

        # 总体统计
        all_times = []
        for times in self.compression_times.values():
            all_times.extend(times)

        if len(all_times) > 0:
            print('-'*80)
            print(f'Overall average compression time: {np.mean(all_times)*1000:.3f} ms')
            print(f'Overall total compression time: {np.sum(all_times)*1000:.3f} ms')
            print('='*80)

    def clear_statistics(self):
        """清除所有性能统计"""
        self.compression_times.clear()
        self.histogram_times.clear()
        self.search_times.clear()
        self.mask_times.clear()


# 全局流水线实例
_global_pipeline = None


def get_pipeline(enable_pipeline=True):
    """
    获取或创建全局流水线实例

    Args:
        enable_pipeline: 是否启用异步流水线

    Returns:
        HGGPipelineCompressor实例
    """
    global _global_pipeline
    if _global_pipeline is None:
        _global_pipeline = HGGPipelineCompressor(enable_pipeline=enable_pipeline)
    return _global_pipeline


def reset_pipeline():
    """重置全局流水线实例"""
    global _global_pipeline
    _global_pipeline = None
