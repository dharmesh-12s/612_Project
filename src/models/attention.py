"""Transformer encoder written from scratch (Vaswani et al., 2017).

Contents:
    MultiHeadAttention   scaled dot-product attention with h heads
    PositionalEncoding   fixed sinusoidal position encoding
    EncoderLayer         attention + feed-forward, residuals, LayerNorm (post-LN or pre-LN)
    Encoder              stack of EncoderLayers

Everything is trained from random initialization. MultiHeadAttention is checked
against torch.nn.MultiheadAttention in tests/test_attention.py.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model, self.h, self.d_k = d_model, n_heads, d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split(self, x):
        # (B, L, d_model) -> (B, h, L, d_k)
        B, L, _ = x.shape
        return x.view(B, L, self.h, self.d_k).transpose(1, 2)

    def forward(self, q, k, v, attn_bias=None, key_padding_mask=None, need_weights=False):
        """
        q: (B, Lq, d)   k, v: (B, Lk, d)
        attn_bias:        (Lq, Lk) or (B, h, Lq, Lk) float, added to scores
                          (e.g. -inf to block, or -alpha * distance for spatial attention)
        key_padding_mask: (B, Lk) bool, True = ignore this key
        """
        Q, K, V = self._split(self.w_q(q)), self._split(self.w_k(k)), self._split(self.w_v(v))
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)          # (B, h, Lq, Lk)
        if attn_bias is not None:
            scores = scores + attn_bias
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = self.dropout(attn) @ V                                     # (B, h, Lq, d_k)
        out = out.transpose(1, 2).reshape(q.shape[0], q.shape[1], self.d_model)
        out = self.w_o(out)
        return (out, attn) if need_weights else out


class PositionalEncoding(nn.Module):
    """PE(pos, 2i) = sin(pos / 10000^(2i/d)),  PE(pos, 2i+1) = cos(...)."""

    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):                                                # x: (B, L, d)
        return x + self.pe[: x.shape[1]]


class EncoderLayer(nn.Module):
    """One encoder block. pre_ln=False is the original Vaswani layout;
    pre_ln=True normalizes before each sublayer (Xiong et al., 2020), which trains more stably."""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, pre_ln=False):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model))
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.drop1, self.drop2 = nn.Dropout(dropout), nn.Dropout(dropout)
        self.pre_ln = pre_ln

    def forward(self, x, attn_bias=None, key_padding_mask=None):
        if self.pre_ln:
            h = self.norm1(x)
            x = x + self.drop1(self.attn(h, h, h, attn_bias, key_padding_mask))
            x = x + self.drop2(self.ff(self.norm2(x)))
        else:
            x = self.norm1(x + self.drop1(self.attn(x, x, x, attn_bias, key_padding_mask)))
            x = self.norm2(x + self.drop2(self.ff(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, d_ff, dropout=0.1, pre_ln=False):
        super().__init__()
        self.layers = nn.ModuleList(
            EncoderLayer(d_model, n_heads, d_ff, dropout, pre_ln) for _ in range(n_layers))
        self.final_norm = nn.LayerNorm(d_model) if pre_ln else nn.Identity()

    def forward(self, x, attn_bias=None, key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, attn_bias, key_padding_mask)
        return self.final_norm(x)
