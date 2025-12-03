# -*- coding: utf-8 -*-
"""
GPT-2数据加载器
支持WikiText-2和OpenWebText数据集
"""

import torch
from torch.utils.data import Dataset, DataLoader
import os

try:
    from transformers import GPT2Tokenizer
    from datasets import load_dataset
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False
    print("Warning: transformers or datasets not installed. GPT-2 data loading will not be available.")


class GPT2TextDataset(Dataset):
    """GPT-2文本数据集"""

    def __init__(self, texts, tokenizer, seq_length=512):
        """
        Args:
            texts: 文本列表
            tokenizer: GPT2Tokenizer
            seq_length: 序列长度
        """
        self.tokenizer = tokenizer
        self.seq_length = seq_length

        # Tokenize所有文本
        self.examples = []
        for text in texts:
            if not text.strip():
                continue

            tokens = tokenizer.encode(text, add_special_tokens=True)

            # 分割成固定长度的块
            for i in range(0, len(tokens) - seq_length, seq_length):
                chunk = tokens[i:i + seq_length + 1]
                if len(chunk) == seq_length + 1:
                    self.examples.append(chunk)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        tokens = self.examples[idx]
        # input_ids是前seq_length个token
        # labels是后seq_length个token（向右shift 1）
        return {
            'input_ids': torch.tensor(tokens[:-1], dtype=torch.long),
            'labels': torch.tensor(tokens[1:], dtype=torch.long)
        }


def load_wikitext2(data_dir='./data', seq_length=512, tokenizer_name='gpt2'):
    """
    加载WikiText-2数据集

    Args:
        data_dir: 数据目录
        seq_length: 序列长度
        tokenizer_name: tokenizer名称

    Returns:
        train_dataset, valid_dataset, test_dataset
    """
    if not HAS_DEPS:
        raise ImportError("transformers and datasets required. Install with: pip install transformers datasets")

    # 加载tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    # 加载WikiText-2数据集
    print("Loading WikiText-2 dataset...")
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', cache_dir=data_dir)

    # 创建数据集
    train_dataset = GPT2TextDataset(
        dataset['train']['text'],
        tokenizer,
        seq_length
    )

    valid_dataset = GPT2TextDataset(
        dataset['validation']['text'],
        tokenizer,
        seq_length
    )

    test_dataset = GPT2TextDataset(
        dataset['test']['text'],
        tokenizer,
        seq_length
    )

    print(f"Train examples: {len(train_dataset)}")
    print(f"Valid examples: {len(valid_dataset)}")
    print(f"Test examples: {len(test_dataset)}")

    return train_dataset, valid_dataset, test_dataset, tokenizer


def load_openwebtext(data_dir='./data', seq_length=512, tokenizer_name='gpt2', num_samples=100000):
    """
    加载OpenWebText数据集（子集）

    Args:
        data_dir: 数据目录
        seq_length: 序列长度
        tokenizer_name: tokenizer名称
        num_samples: 使用的样本数量

    Returns:
        train_dataset, valid_dataset, test_dataset
    """
    if not HAS_DEPS:
        raise ImportError("transformers and datasets required. Install with: pip install transformers datasets")

    # 加载tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    # 加载OpenWebText数据集
    print(f"Loading OpenWebText dataset (first {num_samples} samples)...")
    dataset = load_dataset('openwebtext', split=f'train[:{num_samples}]', cache_dir=data_dir)

    # 分割数据集
    texts = dataset['text']
    train_size = int(0.9 * len(texts))
    valid_size = int(0.05 * len(texts))

    train_texts = texts[:train_size]
    valid_texts = texts[train_size:train_size + valid_size]
    test_texts = texts[train_size + valid_size:]

    # 创建数据集
    train_dataset = GPT2TextDataset(train_texts, tokenizer, seq_length)
    valid_dataset = GPT2TextDataset(valid_texts, tokenizer, seq_length)
    test_dataset = GPT2TextDataset(test_texts, tokenizer, seq_length)

    print(f"Train examples: {len(train_dataset)}")
    print(f"Valid examples: {len(valid_dataset)}")
    print(f"Test examples: {len(test_dataset)}")

    return train_dataset, valid_dataset, test_dataset, tokenizer


def collate_fn(batch):
    """数据collate函数"""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])

    return {
        'input_ids': input_ids,
        'labels': labels
    }


def create_gpt2_dataloaders(dataset_name='wikitext2', data_dir='./data',
                            seq_length=512, batch_size=4, num_workers=2):
    """
    创建GPT-2数据加载器

    Args:
        dataset_name: 数据集名称 ('wikitext2' 或 'openwebtext')
        data_dir: 数据目录
        seq_length: 序列长度
        batch_size: 批大小
        num_workers: worker数量

    Returns:
        train_loader, valid_loader, test_loader, tokenizer
    """
    if dataset_name == 'wikitext2':
        train_ds, valid_ds, test_ds, tokenizer = load_wikitext2(data_dir, seq_length)
    elif dataset_name == 'openwebtext':
        train_ds, valid_ds, test_ds, tokenizer = load_openwebtext(data_dir, seq_length)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    return train_loader, valid_loader, test_loader, tokenizer
