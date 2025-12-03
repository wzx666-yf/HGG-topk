# -*- coding: utf-8 -*-
"""
统一训练器 - 支持多种模型和详细性能分析

支持：
- 视觉模型: ResNet18/50, VGG11/16 (CIFAR-10/100)
- 语言模型: LSTM (PTB), GPT2-small/medium
- 所有压缩算法: TopK, Gaussian, RedSync, HGG-TopK (含流水线)
- 详细性能测量
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import time
import json
import numpy as np
from typing import Dict, Tuple

from core.compression import compressors
from core.hgg_pipeline import HGGPipelineCompressor
from core.models import LSTMModel, repackage_hidden
from data_utils.ptb_reader import ptb_raw_data, PTBDataset


class Trainer:
    """统一训练器"""

    def __init__(self, rank, world_size, **kwargs):
        self.rank = rank
        self.world_size = world_size
        self.model_name = kwargs.get('model_name', 'resnet18')
        self.dataset = kwargs.get('dataset', 'cifar10')
        self.batch_size = kwargs.get('batch_size', 128)
        self.lr = kwargs.get('lr', 0.1)
        self.epochs = kwargs.get('epochs', 100)
        self.compressor_name = kwargs.get('compressor', None)
        self.density = kwargs.get('density', 1.0)
        self.use_pipeline = kwargs.get('use_pipeline', False)
        self.data_dir = kwargs.get('data_dir', './data')
        self.log_dir = kwargs.get('log_dir', './logs')

        # 判断模型类型
        self.is_vision = self.model_name in ['resnet18', 'resnet50', 'vgg11', 'vgg16']
        self.is_lstm = self.model_name == 'lstm'

        # 设置设备
        self.device = torch.device(f'cuda:{rank}')
        torch.cuda.set_device(self.device)

        # 初始化记录
        self.train_times = []
        self.train_losses = []
        self.train_accs = []
        self.test_accs = []
        self.compression_ratios = []

        # 详细时间记录
        self.forward_times = []
        self.backward_times = []
        self.sparsification_times = []
        self.communication_times = []
        self.threshold_accuracies = []

        # 构建模型和数据
        self.model = self._build_model().to(self.device)
        if not self.is_lstm:
            self.model = DDP(self.model, device_ids=[rank])

        self.train_loader, self.test_loader = self._build_dataloaders()

        # 优化器和损失
        self.criterion = nn.CrossEntropyLoss()
        if self.is_lstm:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=1e-6)
            self.scheduler = None
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=[self.epochs//2, self.epochs*3//4], gamma=0.1
            )

        # 压缩器
        self.compressor = None
        self.pipeline = None
        if self.compressor_name and self.density < 1.0:
            if self.use_pipeline and self.compressor_name == 'hggtopk':
                self.pipeline = HGGPipelineCompressor(enable_pipeline=True)
                if rank == 0:
                    print(f"✓ Using HGG-TopK Pipeline (density={self.density})")
            else:
                self.compressor = compressors[self.compressor_name]
                if rank == 0:
                    print(f"✓ Using {self.compressor_name} (density={self.density})")

    def _build_model(self):
        """构建模型"""
        if self.is_vision:
            num_classes = 10 if self.dataset == 'cifar10' else 100
            if self.model_name == 'resnet18':
                return models.resnet18(num_classes=num_classes)
            elif self.model_name == 'resnet50':
                return models.resnet50(num_classes=num_classes)
            elif self.model_name == 'vgg11':
                return models.vgg11(num_classes=num_classes)
            elif self.model_name == 'vgg16':
                return models.vgg16(num_classes=num_classes)
        elif self.is_lstm:
            train_data, _, _, word_to_id, _ = ptb_raw_data(self.data_dir, prefix="ptb")
            return LSTMModel(
                vocab_size=len(word_to_id),
                embedding_dim=1500,
                num_steps=35,
                batch_size=self.batch_size,
                num_layers=2
            )
        raise ValueError(f"Unsupported model: {self.model_name}")

    def _build_dataloaders(self):
        """构建数据加载器"""
        if self.is_vision:
            return self._build_vision_loaders()
        elif self.is_lstm:
            return self._build_lstm_loaders()

    def _build_vision_loaders(self):
        """构建视觉数据加载器"""
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        if self.dataset == 'cifar10':
            train_dataset = torchvision.datasets.CIFAR10(
                root=self.data_dir, train=True, download=True, transform=transform_train
            )
            test_dataset = torchvision.datasets.CIFAR10(
                root=self.data_dir, train=False, download=True, transform=transform_test
            )
        else:
            train_dataset = torchvision.datasets.CIFAR100(
                root=self.data_dir, train=True, download=True, transform=transform_train
            )
            test_dataset = torchvision.datasets.CIFAR100(
                root=self.data_dir, train=False, download=True, transform=transform_test
            )

        train_sampler = DistributedSampler(train_dataset, num_replicas=self.world_size, rank=self.rank)
        test_sampler = DistributedSampler(test_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler, num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, sampler=test_sampler, num_workers=2, pin_memory=True)

        return train_loader, test_loader

    def _build_lstm_loaders(self):
        """构建LSTM数据加载器"""
        train_data, valid_data, test_data, word_to_id, _ = ptb_raw_data(self.data_dir, prefix="ptb")
        train_dataset = PTBDataset(train_data, self.batch_size, num_steps=35)
        test_dataset = PTBDataset(test_data, self.batch_size, num_steps=35)

        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        return train_loader, test_loader

    def _compute_threshold_accuracy(self, grad_tensor, selected_threshold, k):
        """计算阈值精度"""
        with torch.no_grad():
            abs_grad = torch.abs(grad_tensor)
            if k >= abs_grad.numel():
                return 0.0
            topk_values, _ = torch.topk(abs_grad.view(-1), k)
            true_threshold = topk_values[-1].item()
            if true_threshold < 1e-10:
                return 0.0
            return abs(selected_threshold - true_threshold) / true_threshold

    def _compress_gradients(self):
        """压缩梯度并记录时间"""
        if not self.compressor and not self.pipeline:
            return 0.0, 0.0

        start_time = time.time()
        total_elements = 0
        compressed_elements = 0
        threshold_errors = []

        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue

            grad = param.grad.data
            total_elements += grad.numel()

            # 压缩
            if self.pipeline:
                task = self.pipeline.compress_async(grad, name=name, ratio=self.density)
                indexes, values = self.pipeline.synchronize(task)
            else:
                _, indexes, values = self.compressor.compress(grad, name=name, ratio=self.density)

            if indexes is not None:
                compressed_elements += indexes.numel()

                # 计算阈值精度
                if values.numel() > 0:
                    selected_threshold = torch.min(torch.abs(values)).item()
                    threshold_error = self._compute_threshold_accuracy(grad, selected_threshold, indexes.numel())
                    threshold_errors.append(threshold_error)

                # 应用稀疏梯度
                sparse_grad = torch.zeros_like(grad)
                sparse_grad.view(-1)[indexes] = values
                param.grad.data = sparse_grad

        sparsification_time = time.time() - start_time

        if total_elements > 0:
            self.compression_ratios.append(compressed_elements / total_elements)

        avg_threshold_acc = np.mean(threshold_errors) if threshold_errors else 0.0
        return sparsification_time, avg_threshold_acc

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        if hasattr(self.train_loader, 'sampler'):
            self.train_loader.sampler.set_epoch(epoch)

        running_loss = 0.0
        correct = 0
        total = 0

        epoch_forward = 0.0
        epoch_backward = 0.0
        epoch_sparse = 0.0
        epoch_comm = 0.0
        epoch_threshold_accs = []

        epoch_start = time.time()

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()

            # Forward
            fwd_start = time.time()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            fwd_time = time.time() - fwd_start

            # Backward
            bwd_start = time.time()
            loss.backward()
            bwd_time = time.time() - bwd_start

            # Sparsification
            sparse_time, threshold_acc = self._compress_gradients()

            # Communication
            comm_start = time.time()
            self.optimizer.step()
            comm_time = time.time() - comm_start

            # 统计
            epoch_forward += fwd_time
            epoch_backward += bwd_time
            epoch_sparse += sparse_time
            epoch_comm += comm_time
            epoch_threshold_accs.append(threshold_acc)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if self.rank == 0 and batch_idx % 50 == 0:
                print(f'Epoch {epoch} [{batch_idx}/{len(self.train_loader)}] '
                      f'Loss: {running_loss/(batch_idx+1):.3f} Acc: {100.*correct/total:.2f}%')

        epoch_time = time.time() - epoch_start
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total

        timing_stats = {
            'forward': epoch_forward,
            'backward': epoch_backward,
            'sparsification': epoch_sparse,
            'communication': epoch_comm,
            'threshold_accuracy': np.mean(epoch_threshold_accs)
        }

        return epoch_loss, epoch_acc, epoch_time, timing_stats

    @torch.no_grad()
    def test(self):
        """测试"""
        self.model.eval()
        correct = 0
        total = 0

        for inputs, targets in self.test_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        # 同步
        correct_tensor = torch.tensor(correct).to(self.device)
        total_tensor = torch.tensor(total).to(self.device)
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)

        acc = 100. * correct_tensor.item() / total_tensor.item() if total_tensor.item() > 0 else 0.0
        return acc

    def run(self):
        """运行训练"""
        if self.rank == 0:
            print(f"\n{'='*80}")
            print(f"Training: {self.model_name} on {self.dataset}")
            print(f"Compressor: {self.compressor_name or 'None'} (density={self.density})")
            print(f"Pipeline: {self.use_pipeline}, Epochs: {self.epochs}")
            print(f"{'='*80}\n")

        best_acc = 0.0

        for epoch in range(self.epochs):
            train_loss, train_acc, train_time, timing_stats = self.train_epoch(epoch)
            test_acc = self.test()

            if self.scheduler:
                self.scheduler.step()

            # 记录
            self.train_times.append(train_time)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            self.test_accs.append(test_acc)
            self.forward_times.append(timing_stats['forward'])
            self.backward_times.append(timing_stats['backward'])
            self.sparsification_times.append(timing_stats['sparsification'])
            self.communication_times.append(timing_stats['communication'])
            self.threshold_accuracies.append(timing_stats['threshold_accuracy'])

            if test_acc > best_acc:
                best_acc = test_acc

            if self.rank == 0:
                print(f'\nEpoch {epoch}: Loss={train_loss:.3f}, Train Acc={train_acc:.2f}%, '
                      f'Test Acc={test_acc:.2f}% (Best={best_acc:.2f}%)')
                print(f'  Time: {train_time:.1f}s (Fwd:{timing_stats["forward"]:.1f}s, '
                      f'Bwd:{timing_stats["backward"]:.1f}s, Sparse:{timing_stats["sparsification"]:.1f}s)')
                if self.compressor or self.pipeline:
                    sparse_overhead = timing_stats["sparsification"] / train_time * 100
                    print(f'  Sparsification Overhead: {sparse_overhead:.2f}%')
                    print(f'  Threshold Accuracy: {timing_stats["threshold_accuracy"]:.4f}')

        # 保存结果
        if self.rank == 0:
            self._save_results(best_acc)

        return best_acc

    def _save_results(self, best_acc):
        """保存结果"""
        os.makedirs(self.log_dir, exist_ok=True)

        exp_name = f"{self.model_name}_{self.dataset}"
        if self.compressor_name:
            exp_name += f"_{self.compressor_name}_d{self.density}"
            if self.use_pipeline:
                exp_name += "_pipeline"
        else:
            exp_name += "_baseline"

        results = {
            'model': self.model_name,
            'dataset': self.dataset,
            'compressor': self.compressor_name,
            'density': self.density,
            'use_pipeline': self.use_pipeline,
            'epochs': self.epochs,
            'best_acc': best_acc,
            'final_acc': self.test_accs[-1] if self.test_accs else 0.0,
            'train_times': self.train_times,
            'train_losses': self.train_losses,
            'train_accs': self.train_accs,
            'test_accs': self.test_accs,
            'compression_ratios': self.compression_ratios,
            'avg_epoch_time': np.mean(self.train_times) if self.train_times else 0.0,
            'forward_times': self.forward_times,
            'backward_times': self.backward_times,
            'sparsification_times': self.sparsification_times,
            'communication_times': self.communication_times,
            'threshold_accuracies': self.threshold_accuracies,
            'avg_sparsification_time': np.mean(self.sparsification_times) if self.sparsification_times else 0.0,
            'avg_threshold_accuracy': np.mean(self.threshold_accuracies) if self.threshold_accuracies else 0.0,
        }

        result_file = os.path.join(self.log_dir, f'{exp_name}.json')
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results saved: {result_file}")


def setup_distributed(rank, world_size):
    """初始化分布式"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)


def cleanup():
    """清理"""
    dist.destroy_process_group()


def train_worker(rank, world_size, **kwargs):
    """训练进程"""
    setup_distributed(rank, world_size)
    trainer = Trainer(rank, world_size, **kwargs)
    best_acc = trainer.run()
    cleanup()
    return best_acc


def main(**kwargs):
    """主函数"""
    gpus = kwargs.pop('gpus', None)
    if gpus is None:
        gpus = torch.cuda.device_count()

    if gpus == 0:
        raise RuntimeError("No GPU available!")

    print(f"Using {gpus} GPUs")
    mp.spawn(train_worker, args=(gpus,), kwargs=kwargs, nprocs=gpus, join=True)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='HGG-TopK Training')
    parser.add_argument('--model', type=str, default='resnet18',
                       choices=['resnet18', 'resnet50', 'vgg11', 'vgg16', 'lstm'])
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'cifar100', 'ptb'])
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--compressor', type=str, default=None,
                       choices=['topk', 'gaussian', 'redsync', 'hggtopk'])
    parser.add_argument('--density', type=float, default=1.0)
    parser.add_argument('--use-pipeline', action='store_true')
    parser.add_argument('--gpus', type=int, default=None)
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--log-dir', type=str, default='./logs')

    args = parser.parse_args()

    main(
        model_name=args.model,
        dataset=args.dataset,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        compressor=args.compressor,
        density=args.density,
        use_pipeline=args.use_pipeline,
        gpus=args.gpus,
        data_dir=args.data_dir,
        log_dir=args.log_dir
    )
