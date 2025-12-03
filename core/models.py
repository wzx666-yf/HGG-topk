# -*- coding: utf-8 -*-
"""
语言模型定义
包含LSTM和GPT-2模型
"""
import torch
import torch.nn as nn
from torch.autograd import Variable

try:
    from transformers import GPT2LMHeadModel, GPT2Config
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers not installed. GPT-2 models will not be available.")


class LSTMModel(nn.Module):
    """LSTM语言模型"""

    def __init__(self, vocab_size, embedding_dim=1500, num_steps=35,
                 batch_size=20, num_layers=2, dp_keep_prob=0.35):
        super(LSTMModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.dp_keep_prob = dp_keep_prob
        self.num_layers = num_layers

        self.dropout = nn.Dropout(1 - dp_keep_prob)
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            num_layers=num_layers,
            dropout=1 - dp_keep_prob
        )
        self.sm_fc = nn.Linear(in_features=embedding_dim, out_features=vocab_size)
        self.init_weights()

    def init_weights(self):
        """初始化权重"""
        init_range = 0.1
        self.word_embeddings.weight.data.uniform_(-init_range, init_range)
        self.sm_fc.bias.data.fill_(0.0)
        self.sm_fc.weight.data.uniform_(-init_range, init_range)

    def init_hidden(self, batch_size=None):
        """初始化隐藏状态"""
        if batch_size is None:
            batch_size = self.batch_size
        weight = next(self.parameters()).data
        return (
            Variable(weight.new(self.num_layers, batch_size, self.embedding_dim).zero_()),
            Variable(weight.new(self.num_layers, batch_size, self.embedding_dim).zero_())
        )

    def forward(self, inputs, hidden):
        """前向传播"""
        embeds = self.dropout(self.word_embeddings(inputs))
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = self.dropout(lstm_out)
        logits = self.sm_fc(lstm_out.view(-1, self.embedding_dim))
        return logits.view(self.num_steps, self.batch_size, self.vocab_size), hidden


def repackage_hidden(h):
    """分离隐藏状态的梯度历史"""
    if isinstance(h, Variable):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


class GPT2Medium(nn.Module):
    """GPT-2 Medium模型 (345M参数)"""

    def __init__(self, vocab_size=50257, seq_length=512):
        super(GPT2Medium, self).__init__()
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers library required for GPT-2. Install with: pip install transformers")

        # GPT-2 Medium配置
        config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=seq_length,
            n_embd=1024,
            n_layer=24,
            n_head=16,
            n_inner=4096,
            activation_function="gelu_new",
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
        )

        self.model = GPT2LMHeadModel(config)
        self.seq_length = seq_length

    def forward(self, input_ids, labels=None):
        """
        前向传播
        Args:
            input_ids: (batch_size, seq_length)
            labels: (batch_size, seq_length) 用于计算loss
        Returns:
            如果labels提供，返回(loss, logits)，否则只返回logits
        """
        outputs = self.model(input_ids=input_ids, labels=labels)

        if labels is not None:
            # 返回loss和logits
            return outputs.loss, outputs.logits
        else:
            # 只返回logits
            return outputs.logits


class GPT2Small(nn.Module):
    """GPT-2 Small模型 (117M参数) - 用于快速测试"""

    def __init__(self, vocab_size=50257, seq_length=512):
        super(GPT2Small, self).__init__()
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers library required for GPT-2. Install with: pip install transformers")

        # GPT-2 Small配置
        config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=seq_length,
            n_embd=768,
            n_layer=12,
            n_head=12,
            n_inner=3072,
            activation_function="gelu_new",
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
        )

        self.model = GPT2LMHeadModel(config)
        self.seq_length = seq_length

    def forward(self, input_ids, labels=None):
        """前向传播"""
        outputs = self.model(input_ids=input_ids, labels=labels)

        if labels is not None:
            return outputs.loss, outputs.logits
        else:
            return outputs.logits
