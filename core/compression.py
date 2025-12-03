# -*- coding: utf-8 -*-
"""
梯度压缩算法实现 - 统一版本
包含 TopK, Gaussian, RedSync, DGC, RandomK, HGG-TopK 等多种压缩方法

特性:
- 修复多维张量索引问题
- 统一接口使用 ratio 参数
- 移除外部依赖 (settings, utils)
- 包含完整的 HGG-TopK 实现
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
    """
    标准 TopK 压缩
    Sparse Communication for Distributed Gradient Descent, Alham Fikri Aji et al., 2017
    """
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

            # 展平张量进行 TopK 选择
            tensor_flat = tensor.data.view(-1)
            values, indexes = torch.topk(torch.abs(tensor_flat), k=k)
            values = tensor_flat[indexes]

            # 更新残差
            TopKCompressor.residuals[name].data = tensor.data.clone()
            TopKCompressor.residuals[name].data.view(-1)[indexes] = 0.

            TopKCompressor.values[name] = values
            TopKCompressor.indexes[name] = indexes

            return tensor, indexes, values

    @staticmethod
    def get_residuals(name, like_tensor):
        if name not in TopKCompressor.residuals:
            TopKCompressor.residuals[name] = torch.zeros_like(like_tensor.data)
        return TopKCompressor.residuals[name]

    @staticmethod
    def add_residuals(included_indexes, name):
        with torch.no_grad():
            residuals = TopKCompressor.residuals[name]
            if type(included_indexes) is np.ndarray:
                indexes_t = torch.from_numpy(included_indexes).to(device=residuals.device).long()
            else:
                indexes_t = included_indexes
            values = TopKCompressor.values[name]
            values.data[indexes_t] = 0.0
            residuals.data.view(-1)[TopKCompressor.indexes[name]] += values.data

    @staticmethod
    def decompress(tensor, ctc, name=None):
        return tensor


class TopKCompressor2(TopKCompressor):
    """TopK 压缩（无残差）"""
    name = 'topk2'

    @staticmethod
    def compress(tensor, name=None, ratio=0.05):
        with torch.no_grad():
            if name not in TopKCompressor.residuals:
                TopKCompressor.residuals[name] = torch.zeros_like(tensor.data)

            numel = tensor.numel()
            k = max(int(numel * ratio), 1)

            tensor_flat = tensor.data.view(-1)
            values, indexes = torch.topk(torch.abs(tensor_flat), k=k)
            values = tensor_flat[indexes]

            TopKCompressor.residuals[name].data.view(-1)[indexes] = 0.
            TopKCompressor.values[name] = values
            TopKCompressor.indexes[name] = indexes

            return tensor, indexes, values


class GaussianCompressor():
    """高斯分布压缩"""
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
            # 根据稀疏度估计阈值倍数
            sigma_scale = 2.5

            # 展平张量进行操作
            tensor_flat = tensor.data.view(-1)
            abs_tensor_flat = torch.abs(tensor_flat)
            threshold = float(std) * sigma_scale

            mask = abs_tensor_flat > threshold
            indexes = mask.nonzero(as_tuple=False).view(-1)

            # 自适应调整阈值
            loops = 0
            while loops < 3:
                if indexes.numel() < 2 * k / 3:
                    threshold *= 0.5
                elif indexes.numel() > 4 * k / 3:
                    threshold *= 1.5
                else:
                    break
                mask = abs_tensor_flat > threshold
                indexes = mask.nonzero(as_tuple=False).view(-1)
                loops += 1

            values = tensor_flat[indexes]

            GaussianCompressor.residuals[name].data = tensor.data.clone()
            GaussianCompressor.residuals[name].data.view(-1)[indexes] = 0.0

            GaussianCompressor.values[name] = values
            GaussianCompressor.indexes[name] = indexes

            return tensor, indexes, values

    @staticmethod
    def add_residuals(included_indexes, name):
        with torch.no_grad():
            residuals = GaussianCompressor.residuals[name]
            indexes_t = torch.from_numpy(included_indexes).to(device=residuals.device).long()
            values = GaussianCompressor.values[name]
            values[indexes_t] = 0.0
            residuals.data.view(-1)[GaussianCompressor.indexes[name]] += values.data

    @staticmethod
    def decompress(tensor, ctc, name=None):
        return tensor


class GaussianCompressor2(GaussianCompressor):
    """高斯压缩（无残差）"""
    name = 'gaussian2'

    @staticmethod
    def compress(tensor, name=None, ratio=0.05):
        with torch.no_grad():
            if name not in GaussianCompressor.residuals:
                GaussianCompressor.residuals[name] = torch.zeros_like(tensor.data)

            numel = tensor.numel()
            k = max(int(numel * ratio), 1)

            std = torch.std(tensor)
            mean = torch.mean(tensor)
            sigma_scale = 2.5

            tensor_flat = tensor.data.view(-1)
            abs_tensor_flat = torch.abs(tensor_flat)
            threshold = float(std) * sigma_scale

            mask = abs_tensor_flat > threshold
            indexes = mask.nonzero(as_tuple=False).view(-1)

            loops = 0
            while loops < 5:
                if indexes.numel() < 2 * k / 3:
                    threshold *= 0.5
                elif indexes.numel() > 4 * k / 3:
                    threshold *= 1.5
                else:
                    break
                mask = abs_tensor_flat > threshold
                indexes = mask.nonzero(as_tuple=False).view(-1)
                loops += 1

            values = tensor_flat[indexes]
            GaussianCompressor.residuals[name].data = tensor.data.clone()
            GaussianCompressor.residuals[name].data.view(-1)[indexes] = 0.0

            return tensor, indexes, values


class RandomKCompressor():
    """随机K压缩"""
    residuals = {}
    values = {}
    indexes = {}
    name = 'randomk'
    counter = 0

    @staticmethod
    def clear():
        RandomKCompressor.residuals = {}
        RandomKCompressor.values = {}
        RandomKCompressor.indexes = {}

    @staticmethod
    def compress(tensor, name=None, ratio=0.05):
        with torch.no_grad():
            if name not in RandomKCompressor.residuals:
                RandomKCompressor.residuals[name] = torch.zeros_like(tensor.data)
            numel = tensor.numel()
            k = max(int(numel * ratio), 1)

            perm = torch.randperm(numel, device=tensor.device)
            RandomKCompressor.counter += 1
            indexes = perm[:k]

            tensor_flat = tensor.data.view(-1)
            values = tensor_flat[indexes]

            RandomKCompressor.residuals[name].data = tensor.data.clone()
            RandomKCompressor.residuals[name].data.view(-1)[indexes] = 0.0

            return tensor, indexes, values

    @staticmethod
    def add_residuals(included_indexes, name):
        with torch.no_grad():
            residuals = RandomKCompressor.residuals[name]
            indexes_t = torch.from_numpy(included_indexes).to(device=residuals.device).long()
            values = RandomKCompressor.values[name]
            values[indexes_t] = 0.0
            residuals.data.view(-1)[RandomKCompressor.indexes[name]] += values.data

    @staticmethod
    def decompress(tensor, ctc, name=None):
        return tensor


class RandomKECCompressor(RandomKCompressor):
    """随机K压缩（带误差补偿）"""
    name = 'randomkec'

    @staticmethod
    def compress(tensor, name=None, ratio=0.05):
        with torch.no_grad():
            if name not in RandomKCompressor.residuals:
                RandomKCompressor.residuals[name] = torch.zeros_like(tensor.data)
            numel = tensor.numel()
            k = max(int(numel * ratio), 1)

            tensor.add_(RandomKCompressor.residuals[name].data)
            perm = torch.randperm(numel, device=tensor.device)
            indexes = perm[:k]

            tensor_flat = tensor.data.view(-1)
            values = tensor_flat[indexes]

            RandomKCompressor.residuals[name].data = tensor.data.clone()
            RandomKCompressor.residuals[name].data.view(-1)[indexes] = 0.0

            return tensor, indexes, values


class DGCSamplingCompressor():
    """DGC采样压缩"""
    residuals = {}
    values = {}
    indexes = {}
    name = 'dgcsampling'

    @staticmethod
    def clear():
        DGCSamplingCompressor.residuals = {}
        DGCSamplingCompressor.values = {}
        DGCSamplingCompressor.indexes = {}

    @staticmethod
    def compress(tensor, name=None, ratio=0.05):
        with torch.no_grad():
            if name not in DGCSamplingCompressor.residuals:
                DGCSamplingCompressor.residuals[name] = torch.zeros_like(tensor.data)
            numel = tensor.numel()
            k = max(int(numel * ratio), 1)

            tensor.add_(DGCSamplingCompressor.residuals[name].data)

            tensor_flat = tensor.data.view(-1)
            abs_tensor_flat = torch.abs(tensor_flat)

            # 采样估计阈值
            perm = torch.randperm(numel, device=tensor.device)
            fk = max(int(numel * 0.01), k)
            sampled_indexes = perm[0:fk]
            sampled_values = abs_tensor_flat[sampled_indexes]
            tmpvalues, tmpindexes = torch.topk(sampled_values, k=min(k, sampled_values.numel()))

            thres = tmpvalues[-1] if tmpvalues.numel() > 0 else 0
            mask = abs_tensor_flat > thres
            indexes = mask.nonzero(as_tuple=False).view(-1)

            # 如果选中太多，再做TopK
            if indexes.numel() > 4 * k / 3:
                tmpvalues = abs_tensor_flat[indexes]
                topk_k = min(k, tmpvalues.numel())
                _, tmpindexes = torch.topk(tmpvalues, k=topk_k)
                indexes = indexes[tmpindexes]

            values = tensor_flat[indexes]
            DGCSamplingCompressor.residuals[name].data = tensor.data.clone()
            DGCSamplingCompressor.residuals[name].data.view(-1)[indexes] = 0.0

            return tensor, indexes, values

    @staticmethod
    def add_residuals(included_indexes, name):
        with torch.no_grad():
            residuals = DGCSamplingCompressor.residuals[name]
            indexes_t = torch.from_numpy(included_indexes).to(device=residuals.device).long()
            values = DGCSamplingCompressor.values[name]
            values[indexes_t] = 0.0
            residuals.data.view(-1)[DGCSamplingCompressor.indexes[name]] += values.data

    @staticmethod
    def decompress(tensor, ctc, name=None):
        return tensor


class RedSyncCompressor():
    """
    RedSync: Reducing Synchronization Overhead via Adaptive Threshold Selection
    使用二分搜索在均值和最大值之间找到合适的阈值
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
        with torch.no_grad():
            if name not in RedSyncCompressor.residuals:
                RedSyncCompressor.residuals[name] = torch.zeros_like(tensor.data)

            numel = tensor.numel()
            k = max(int(numel * ratio), 1)

            # 添加残差
            tensor.add_(RedSyncCompressor.residuals[name].data)

            # 展平张量
            tensor_flat = tensor.data.view(-1)
            abs_tensor_flat = torch.abs(tensor_flat)
            mean_val = torch.mean(abs_tensor_flat)
            max_val = torch.max(abs_tensor_flat)

            # 二分搜索阈值
            l = 0.0
            r = 1.0
            eps = 0.2
            thres = 0.0

            while r - l > eps:
                tmp_ratio = l + (r - l) / 2
                thres = mean_val + tmp_ratio * (max_val - mean_val)
                mask = abs_tensor_flat > thres
                indexes = mask.nonzero(as_tuple=False).view(-1)
                nnz = indexes.numel()

                # 如果在 [k, 2k] 范围内则接受
                if nnz > k and 2 * k > nnz:
                    break
                elif nnz < k / 2:
                    r = tmp_ratio
                else:
                    l = tmp_ratio

            values = tensor_flat[indexes]

            # 更新残差
            RedSyncCompressor.residuals[name].data = tensor.data.clone()
            RedSyncCompressor.residuals[name].data.view(-1)[indexes] = 0.0

            RedSyncCompressor.values[name] = values
            RedSyncCompressor.indexes[name] = indexes

            return tensor, indexes, values

    @staticmethod
    def add_residuals(included_indexes, name):
        with torch.no_grad():
            residuals = RedSyncCompressor.residuals[name]
            indexes_t = torch.from_numpy(included_indexes).to(device=residuals.device).long()
            values = RedSyncCompressor.values[name]
            values[indexes_t] = 0.0
            residuals.data.view(-1)[RedSyncCompressor.indexes[name]] += values.data

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
    BETA = 0.98          # 保守插值系数

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
        """构建直方图和后缀和（优化GPU版本）"""
        flat_indices = bin_indices.reshape(-1).long()
        # 使用histc替代bincount，在GPU上更快
        if flat_indices.is_cuda:
            histogram = torch.histc(
                flat_indices.float(),
                bins=num_bins,
                min=0,
                max=num_bins-1
            ).long()
        else:
            histogram = torch.bincount(flat_indices, minlength=num_bins)
        suffix_sum = torch.flip(torch.cumsum(torch.flip(histogram, [0]), dim=0), [0])
        return histogram, suffix_sum

    @staticmethod
    def _binary_search_threshold(suffix_sum, k, tolerance):
        """二分搜索阈值桶（优化版：减少GPU-CPU同步）"""
        # 在GPU上找到最接近k的索引
        diff = torch.abs(suffix_sum - k)
        within_tolerance = diff <= tolerance

        if within_tolerance.any():
            # 找到满足容忍度的第一个索引
            valid_indices = within_tolerance.nonzero(as_tuple=False).view(-1)
            return valid_indices[0].item()

        # 找到最接近的索引
        best_idx = torch.argmin(diff).item()
        return best_idx

    @staticmethod
    def _galloping_search(suffix_sum, k, prev_idx, tolerance):
        """历史引导的倍增搜索（优化版：减少GPU-CPU同步）"""
        # 检查prev_idx是否仍然有效
        if 0 <= prev_idx < len(suffix_sum):
            diff_at_prev = torch.abs(suffix_sum[prev_idx] - k)
            if diff_at_prev.item() <= tolerance:
                return prev_idx

        # 使用向量化搜索代替迭代
        # 在prev_idx附近的一个窗口内搜索
        window_size = min(128, len(suffix_sum))
        start = max(0, prev_idx - window_size // 2)
        end = min(len(suffix_sum), start + window_size)

        window_suffix = suffix_sum[start:end]
        diff = torch.abs(window_suffix - k)

        # 找到窗口内最佳索引
        within_tolerance = diff <= tolerance
        if within_tolerance.any():
            valid_indices = within_tolerance.nonzero(as_tuple=False).view(-1)
            return (start + valid_indices[0]).item()

        # 如果窗口内没有找到，使用全局搜索
        return HGGTopKCompressor._binary_search_threshold(suffix_sum, k, tolerance)

    @staticmethod
    def _compute_final_threshold(bin_idx, abs_values, bin_indices, max_val, k, num_bins, gamma, beta):
        """计算最终阈值（带插值，优化版）"""
        denominator = math.log(1.0 + gamma * max_val)

        bin_lower_ratio = bin_idx / num_bins
        bin_lower_val = (math.exp(bin_lower_ratio * denominator) - 1.0) / gamma

        bin_upper_ratio = (bin_idx + 1) / num_bins
        bin_upper_val = (math.exp(bin_upper_ratio * denominator) - 1.0) / gamma

        # 批量计算mask，减少GPU操作
        mask_above = bin_indices > bin_idx
        mask_in_bin = bin_indices == bin_idx

        # 一次性同步到CPU
        counts = torch.stack([mask_above.sum(), mask_in_bin.sum()])
        count_above, count_in_bin = counts.tolist()

        if count_above >= k:
            return bin_lower_val

        if count_in_bin == 0:
            return bin_lower_val

        k_remain = k - count_above
        bin_width = bin_upper_val - bin_lower_val
        interpolation = 1.0 - (k_remain / count_in_bin)
        final_threshold = bin_lower_val + bin_width * interpolation * beta

        return final_threshold

    @staticmethod
    def compress(tensor, name=None, ratio=0.05):
        """压缩梯度（优化版：减少GPU-CPU同步，使用缓存）"""
        with torch.no_grad():
            if name not in HGGTopKCompressor.residuals:
                HGGTopKCompressor.residuals[name] = torch.zeros_like(tensor.data)
                HGGTopKCompressor.iteration_count[name] = 0

            numel = tensor.numel()
            k = max(int(numel * ratio), 1)
            tolerance = max(int(k * HGGTopKCompressor.TOLERANCE), 1)

            # 误差补偿
            tensor.add_(HGGTopKCompressor.residuals[name].data)

            # 计算绝对值和最大值（减少一次.item()调用）
            abs_values = torch.abs(tensor.data)
            max_val_tensor = torch.max(abs_values)

            # 提前检查是否需要压缩
            if max_val_tensor < 1e-10:
                HGGTopKCompressor.residuals[name].data = tensor.data.clone()
                empty_indexes = torch.tensor([], dtype=torch.long, device=tensor.device)
                empty_values = torch.tensor([], dtype=tensor.dtype, device=tensor.device)
                return tensor, empty_indexes, empty_values

            max_val = max_val_tensor.item()

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
                numerator = math.log(1.0 + HGGTopKCompressor.GAMMA * prev_threshold)
                denominator = math.log(1.0 + HGGTopKCompressor.GAMMA * max_val)
                prev_idx = int(HGGTopKCompressor.NUM_BINS * numerator / denominator)
                prev_idx = max(0, min(HGGTopKCompressor.NUM_BINS - 1, prev_idx))

                bin_idx = HGGTopKCompressor._galloping_search(
                    suffix_sum, k, prev_idx, tolerance
                )

            # 计算最终阈值
            final_threshold = HGGTopKCompressor._compute_final_threshold(
                bin_idx, abs_values, bin_indices, max_val, k,
                HGGTopKCompressor.NUM_BINS, HGGTopKCompressor.GAMMA,
                HGGTopKCompressor.BETA
            )

            # 选择梯度（优化：展平后统一处理）
            flat_abs = abs_values.view(-1)
            flat_tensor = tensor.data.view(-1)

            # 使用阈值过滤
            mask = flat_abs >= final_threshold
            indexes = mask.nonzero(as_tuple=False).view(-1)

            # 如果超过k，使用topk精确选择
            if indexes.numel() > k:
                selected_abs_values = flat_abs[indexes]
                topk_k = min(k, selected_abs_values.numel())
                _, topk_indices = torch.topk(selected_abs_values, topk_k)
                indexes = indexes[topk_indices]

            values = flat_tensor[indexes]

            # 更新残差（优化：直接在展平视图上操作）
            residual_flat = tensor.data.clone().view(-1)
            residual_flat[indexes] = 0.0
            HGGTopKCompressor.residuals[name].data = residual_flat.view_as(tensor.data)

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
    'topk2': TopKCompressor2,
    'gaussian': GaussianCompressor,
    'gaussian2': GaussianCompressor2,
    'randomk': RandomKCompressor,
    'randomkec': RandomKECCompressor,
    'dgcsampling': DGCSamplingCompressor,
    'redsync': RedSyncCompressor,
    'hggtopk': HGGTopKCompressor,
}
