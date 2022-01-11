import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.modules import (
    LayerNorm,
    SamePad,
)
from fairseq.modules.transformer_sentence_encoder import init_bert_params

from fairseq.models.wav2vec.wav2vec2 import TransformerSentenceEncoderLayer

class TransformerEncoder(nn.Module):
    """
    Args:
        encoder_embed_dim: 卷积特征图的通道数，每个pixel作为一个样本。
        encoder_ffn_embed_dim: feedforward模块的中间层的维度。
        encoder_attention_heads: self-attention 的头数
        layer_norm_first: "apply layernorm first in the transformer"
        encoder_layers: num encoder layers in the transformer
        seq_len: input sequence length
    """
    def __init__(self, dropout=0.1, encoder_embed_dim=128, encoder_ffn_embed_dim=256, encoder_attention_heads=1,
                 layer_norm_first=False, encoder_layers=2, seq_len=3, learnable_pos=False):
        super().__init__()

        self.dropout = dropout
        self.embedding_dim = encoder_embed_dim  # 特征图通道数
        self.learnable_pos = learnable_pos

        # fixed positional embeddings
        self.sinusoid_table = self.get_sinusoid_table(seq_len + 1, encoder_embed_dim)
        self.pos_embedding = nn.Embedding.from_pretrained(self.sinusoid_table, freeze=True)

        # trainable positional embeddings
        self.pos_conv = nn.Conv1d(
            self.embedding_dim,
            self.embedding_dim,
            kernel_size=1)
        std = math.sqrt(4 / (seq_len * self.embedding_dim))
        nn.init.normal_(self.pos_conv.weight, mean=0, std=std)
        nn.init.constant_(self.pos_conv.bias, 0)
        self.pos_conv = nn.utils.weight_norm(self.pos_conv, name="weight", dim=2)
        self.pos_conv = nn.Sequential(self.pos_conv, nn.GELU())

        self.layers = nn.ModuleList(
            [
                TransformerSentenceEncoderLayer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=encoder_ffn_embed_dim,
                    num_attention_heads=encoder_attention_heads,
                    dropout=self.dropout,
                    attention_dropout=0.1,
                    activation_dropout=0.0,
                    activation_fn="gelu",
                    layer_norm_first=layer_norm_first,
                )
                for _ in range(encoder_layers)
            ]
        )
        self.layer_norm_first = layer_norm_first
        self.layer_norm = LayerNorm(self.embedding_dim)

        self.apply(init_bert_params)

    def forward(self, x):
        x = self.extract_features(x)

        if self.layer_norm_first:
            x = self.layer_norm(x)

        return x

    def extract_features(self, x):
        if self.learnable_pos:
            # 位置编码
            x_conv = self.pos_conv(x.transpose(1, 2))
            x_conv = x_conv.transpose(1, 2)
            x = x + x_conv
        else:
            positions = torch.arange(x.size(1), device=x.device, dtype=torch.long).repeat(x.size(0), 1)
            x = x + self.pos_embedding(positions)


        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        for i, layer in enumerate(self.layers):
            x, z = layer(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x

    def get_sinusoid_table(self, seq_len, d_model):
        def get_angle(pos, i, d_model):
            return pos / np.power(10000, (2 * (i // 2)) / d_model)

        sinusoid_table = np.zeros((seq_len, d_model))
        for pos in range(seq_len):
            for i in range(d_model):
                if i % 2 == 0:
                    sinusoid_table[pos, i] = np.sin(get_angle(pos, i, d_model))
                else:
                    sinusoid_table[pos, i] = np.cos(get_angle(pos, i, d_model))

        return torch.FloatTensor(sinusoid_table)

