# -*- coding: utf-8 -*-
"""
PTB数据集读取工具
用于LSTM语言模型训练
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from torch.utils.data import Dataset
import collections
import os
import numpy as np


def _read_words(filename):
    """读取文件并分词"""
    with open(filename, "r", encoding='utf-8') as f:
        return f.read().replace("\n", "<eos>").split()


def _build_vocab(filename):
    """构建词汇表"""
    data = _read_words(filename)
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    id_to_word = dict((v, k) for k, v in word_to_id.items())
    return word_to_id, id_to_word


def _file_to_word_ids(filename, word_to_id):
    """将文件转换为词ID序列"""
    data = _read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]


def ptb_raw_data(data_path=None, prefix="ptb"):
    """
    加载PTB原始数据

    Args:
        data_path: 数据目录路径
        prefix: 文件前缀 (ptb 或 wikitext)

    Returns:
        train_data, valid_data, test_data, word_to_id, id_to_word
    """
    train_path = os.path.join(data_path, prefix + ".train.txt")
    valid_path = os.path.join(data_path, prefix + ".valid.txt")
    test_path = os.path.join(data_path, prefix + ".test.txt")

    word_to_id, id_to_word = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)

    return train_data, valid_data, test_data, word_to_id, id_to_word


class PTBDataset(Dataset):
    """PTB数据集"""

    def __init__(self, raw_data, batch_size, num_steps):
        self.raw_data = np.array(raw_data, dtype=np.int64)
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.data_len = len(self.raw_data)
        self.sample_len = self.data_len // self.num_steps

    def __getitem__(self, idx):
        num_steps_begin_index = self.num_steps * idx
        num_steps_end_index = self.num_steps * (idx + 1)

        x = self.raw_data[num_steps_begin_index:num_steps_end_index]
        y = self.raw_data[num_steps_begin_index + 1:num_steps_end_index + 1]

        return (x, y)

    def __len__(self):
        return self.sample_len - self.sample_len % self.batch_size
