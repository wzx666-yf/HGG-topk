# -*- coding: utf-8 -*-
"""
梯度压缩算法实现
包含 TopK, Gaussian, HGG-TopK 等多种压缩方法
"""
from __future__ import print_function
import torch
import numpy as np
import time
import math


class NoneCompressor():
    """无压缩"""
    @staticmethod
    def compress(tensor, name=None, ratio=0.05):
        return tensor, None, tensor.data.view(-1)

    @staticmethod
    def decompress(tensor, ctc, name=None):
        return tensor


class TopKCompressor():
    """标准 TopK 压缩"""
    residuals = {}
    values = {}
    indexes = {}
    name = 'topk'

    @staticmethod
    def clear():
        TopKCompressor.residuals = {}
        TopKCompressor.values = {}
        TopKCompressor.indexes = {}

    @staticmethod
    def compress(tensor, name=None, ratio=0.05):
        with torch.no_grad():
            if name not in TopKCompressor.residuals:
                TopKCompressor.residuals[name] = torch.zeros_like(tensor.data)

            numel = tensor.numel()
            k = max(int(numel * ratio), 1)

            # 添加残差
            tensor.add_(TopKCompressor.residuals[name].data)

            # TopK 选择
            values, indexes = torch.topk(torch.abs(tensor.data), k=k)
            values = tensor.data[indexes]

            # 更新残差
            TopKCompressor.residuals[name].data = tensor.data.clone()
            TopKCompressor.residuals[name].data[indexes] = 0.

            TopKCompressor.values[name] = values
            TopKCompressor.indexes[name] = indexes

            return tensor, indexes, values

    @staticmethod
    def decompress(tensor, ctc, name=None):
        return tensor


class GaussianCompressor():
    """高斯压缩"""
    residuals = {}
    values = {}
    indexes = {}
    name = 'gaussian'

    @staticmethod
    def clear():
        GaussianCompressor.residuals = {}
        GaussianCompressor.values = {}
        GaussianCompressor.indexes = {}

    @staticmethod
    def compress(tensor, name=None, ratio=0.05):
        with torch.no_grad():
            if name not in GaussianCompressor.residuals:
                GaussianCompressor.residuals[name] = torch.zeros_like(tensor.data)

            numel = tensor.numel()
            k = max(int(numel * ratio), 1)

            tensor.add_(GaussianCompressor.residuals[name].data)

            # 使用标准差阈值
            std = torch.std(tensor)
            mean = torch.mean(tensor)
            threshold = float(std) * 2.5

            abs_tensor = torch.abs(tensor)
            mask = abs_tensor > threshold
            indexes = mask.nonzero().squeeze().view(-1)

            # 调整到接近k
            if indexes.numel() > k * 1.5:
                values = abs_tensor[indexes]
                _, topk_idx = torch.topk(values, k)
                indexes = indexes[topk_idx]

            values = tensor.data[indexes]

            GaussianCompressor.residuals[name].data = tensor.data.clone()
            GaussianCompressor.residuals[name].data[indexes] = 0.0

            GaussianCompressor.values[name] = values
            GaussianCompressor.indexes[name] = indexes

            return tensor, indexes, values

    @staticmethod
    def decompress(tensor, ctc, name=None):
        return tensor


class RedSyncCompressor():
    """
    RedSync: Reducing Synchronization Overhead via Adaptive Threshold Selection

    使用二分搜索在均值和最大值之间找到合适的阈值
    目标是选择接近k个梯度，误差控制在[k/2, 2k]
    """
    residuals = {}
    values = {}
    indexes = {}
    name = 'redsync'

    @staticmethod
    def clear():
        RedSyncCompressor.residuals = {}
        RedSyncCompressor.values = {}
        RedSyncCompressor.indexes = {}

    @staticmethod
    def compress(tensor, name=None, ratio=0.05):
        """使用自适应阈值搜索压缩梯度"""
        with torch.no_grad():
            if name not in RedSyncCompressor.residuals:
                RedSyncCompressor.residuals[name] = torch.zeros_like(tensor.data)

            numel = tensor.numel()
            k = max(int(numel * ratio), 1)

            # 添加残差
            tensor.add_(RedSyncCompressor.residuals[name].data)

            abs_tensor = torch.abs(tensor)
            mean_val = torch.mean(abs_tensor)
            max_val = torch.max(abs_tensor)

            # 二分搜索阈值
            l = 0.0
            r = 1.0
            eps = 0.2  # 搜索精度
            thres = 0.0

            while r - l > eps:
                tmp_ratio = l + (r - l) / 2
                thres = mean_val + tmp_ratio * (max_val - mean_val)
                mask = abs_tensor > thres
                indexes = mask.nonzero().squeeze().view(-1)
                nnz = indexes.numel()

                # 如果在 [k, 2k] 范围内则接受
                if nnz > k and 2 * k > nnz:
                    break
                elif nnz < k / 2:
                    r = tmp_ratio
                else:
                    l = tmp_ratio

            # 最终选择
            indexes = indexes
            values = tensor.data[indexes]

            # 更新残差
            RedSyncCompressor.residuals[name].data = tensor.data.clone()
            RedSyncCompressor.residuals[name].data[indexes] = 0.0

            RedSyncCompressor.values[name] = values
            RedSyncCompressor.indexes[name] = indexes

            return tensor, indexes, values

    @staticmethod
    def decompress(tensor, ctc, name=None):
        return tensor


class HGGTopKCompressor():
    """
    HGG-TopK: History-Guided Gradient Top-K 压缩算法

    核心特性:
    1. 对数域指数分桶 - 处理长尾分布
    2. 混合阈值搜索 - 冷启动二分 + 历史引导倍增
    3. 误差补偿机制 - 保证收敛
    4. 自适应阈值插值
    """
    residuals = {}
    values = {}
    indexes = {}
    prev_thresholds = {}
    iteration_count = {}
    name = 'hggtopk'

    # 可调超参数
    NUM_BINS = 1024      # 桶数量
    GAMMA = 1000.0       # 对数缩放因子
    TOLERANCE = 0.01     # 搜索容忍度
    BETA = 0.98         # 保守插值系数

    @staticmethod
    def clear():
        HGGTopKCompressor.residuals = {}
        HGGTopKCompressor.values = {}
        HGGTopKCompressor.indexes = {}
        HGGTopKCompressor.prev_thresholds = {}
        HGGTopKCompressor.iteration_count = {}

    @staticmethod
    def _log_bin_mapping(abs_values, max_val, num_bins, gamma):
        """对数域分桶映射"""
        if max_val < 1e-10:
            return torch.zeros_like(abs_values, dtype=torch.long)

        numerator = torch.log(1.0 + gamma * abs_values)
        denominator = math.log(1.0 + gamma * max_val)
        bin_indices = torch.floor(num_bins * numerator / denominator)
        bin_indices = torch.clamp(bin_indices, 0, num_bins - 1).long()

        return bin_indices

    @staticmethod
    def _build_histogram(bin_indices, num_bins):
        """构建直方图和后缀和"""
        # torch.bincount 仅接受 1D 非负整数输入，确保展平且类型正确
        flat_indices = bin_indices.reshape(-1).long()
        histogram = torch.bincount(flat_indices, minlength=num_bins)
        suffix_sum = torch.flip(torch.cumsum(torch.flip(histogram, [0]), dim=0), [0])
        return histogram, suffix_sum

    @staticmethod
    def _binary_search_threshold(suffix_sum, k, tolerance):
        """二分搜索阈值桶"""
        left, right = 0, len(suffix_sum) - 1
        best_idx = right

        while left <= right:
            mid = (left + right) // 2
            count = suffix_sum[mid].item()

            if abs(count - k) <= tolerance:
                return mid

            if count > k:
                left = mid + 1
                best_idx = mid
            else:
                right = mid - 1

        return best_idx

    @staticmethod
    def _galloping_search(suffix_sum, k, prev_idx, tolerance):
        """历史引导的倍增搜索"""
        if 0 <= prev_idx < len(suffix_sum):
            count = suffix_sum[prev_idx].item()
            if abs(count - k) <= tolerance:
                return prev_idx

        step = 1
        direction = 1 if count > k else -1
        start_idx = prev_idx

        while True:
            new_idx = prev_idx + direction * step

            if new_idx < 0 or new_idx >= len(suffix_sum):
                new_idx = max(0, min(len(suffix_sum) - 1, new_idx))
                break

            new_count = suffix_sum[new_idx].item()

            if direction == 1 and new_count < k:
                break
            elif direction == -1 and new_count > k:
                break

            if abs(new_count - k) <= tolerance:
                return new_idx

            prev_idx = new_idx
            step *= 2

        left = min(start_idx, new_idx)
        right = max(start_idx, new_idx)

        return HGGTopKCompressor._binary_search_threshold(
            suffix_sum[left:right+1], k, tolerance
        ) + left

    @staticmethod
    def _compute_final_threshold(bin_idx, abs_values, bin_indices, max_val, k, num_bins, gamma, beta):
        """计算最终阈值（带插值）"""
        denominator = math.log(1.0 + gamma * max_val)

        bin_lower_ratio = bin_idx / num_bins
        bin_lower_val = (math.exp(bin_lower_ratio * denominator) - 1.0) / gamma

        bin_upper_ratio = (bin_idx + 1) / num_bins
        bin_upper_val = (math.exp(bin_upper_ratio * denominator) - 1.0) / gamma

        mask = bin_indices >= bin_idx
        count_above = torch.sum(mask).item()

        if count_above >= k:
            return bin_lower_val

        mask_in_bin = bin_indices == bin_idx
        count_in_bin = torch.sum(mask_in_bin).item()

        if count_in_bin == 0:
            return bin_lower_val

        k_remain = k - (count_above - count_in_bin)
        bin_width = bin_upper_val - bin_lower_val
        interpolation = 1.0 - (k_remain / count_in_bin)
        final_threshold = bin_lower_val + bin_width * interpolation * beta

        return final_threshold

    @staticmethod
    def compress(tensor, name=None, ratio=0.05):
        """压缩梯度"""
        with torch.no_grad():
            if name not in HGGTopKCompressor.residuals:
                HGGTopKCompressor.residuals[name] = torch.zeros_like(tensor.data)
                HGGTopKCompressor.iteration_count[name] = 0

            numel = tensor.numel()
            k = max(int(numel * ratio), 1)
            tolerance = max(int(k * HGGTopKCompressor.TOLERANCE), 1)

            # 误差补偿
            tensor.add_(HGGTopKCompressor.residuals[name].data)

            abs_values = torch.abs(tensor.data)
            max_val = torch.max(abs_values).item()

            if max_val < 1e-10:
                HGGTopKCompressor.residuals[name].data = tensor.data.clone()
                empty_indexes = torch.tensor([], dtype=torch.long, device=tensor.device)
                empty_values = torch.tensor([], dtype=tensor.dtype, device=tensor.device)
                return tensor, empty_indexes, empty_values

            # 对数域分桶
            bin_indices = HGGTopKCompressor._log_bin_mapping(
                abs_values, max_val, HGGTopKCompressor.NUM_BINS, HGGTopKCompressor.GAMMA
            )

            histogram, suffix_sum = HGGTopKCompressor._build_histogram(
                bin_indices, HGGTopKCompressor.NUM_BINS
            )

            # 混合搜索
            iteration = HGGTopKCompressor.iteration_count[name]

            if iteration == 0 or name not in HGGTopKCompressor.prev_thresholds:
                bin_idx = HGGTopKCompressor._binary_search_threshold(
                    suffix_sum, k, tolerance
                )
            else:
                prev_threshold = HGGTopKCompressor.prev_thresholds[name]

                if max_val > 1e-10:
                    numerator = math.log(1.0 + HGGTopKCompressor.GAMMA * prev_threshold)
                    denominator = math.log(1.0 + HGGTopKCompressor.GAMMA * max_val)
                    prev_idx = int(HGGTopKCompressor.NUM_BINS * numerator / denominator)
                    prev_idx = max(0, min(HGGTopKCompressor.NUM_BINS - 1, prev_idx))
                else:
                    prev_idx = 0

                bin_idx = HGGTopKCompressor._galloping_search(
                    suffix_sum, k, prev_idx, tolerance
                )

            # 计算最终阈值
            final_threshold = HGGTopKCompressor._compute_final_threshold(
                bin_idx, abs_values, bin_indices, max_val, k,
                HGGTopKCompressor.NUM_BINS, HGGTopKCompressor.GAMMA,
                HGGTopKCompressor.BETA
            )

            # 选择梯度
            mask = abs_values >= final_threshold
            # nonzero 输出可能是非连续内存，使用 reshape 保证展平
            indexes = mask.nonzero(as_tuple=False).reshape(-1)

            if indexes.numel() > k:
                selected_abs_values = abs_values[indexes]
                _, topk_indices = torch.topk(selected_abs_values, k)
                indexes = indexes[topk_indices]

            values = tensor.data[indexes]

            # 更新残差
            HGGTopKCompressor.residuals[name].data = tensor.data.clone()
            HGGTopKCompressor.residuals[name].data[indexes] = 0.0

            HGGTopKCompressor.values[name] = values
            HGGTopKCompressor.indexes[name] = indexes
            HGGTopKCompressor.prev_thresholds[name] = final_threshold
            HGGTopKCompressor.iteration_count[name] += 1

            return tensor, indexes, values

    @staticmethod
    def decompress(tensor, ctc, name=None):
        return tensor


# 压缩器字典
compressors = {
    'none': NoneCompressor,
    'topk': TopKCompressor,
    'gaussian': GaussianCompressor,
    'redsync': RedSyncCompressor,
    'hggtopk': HGGTopKCompressor,
}
