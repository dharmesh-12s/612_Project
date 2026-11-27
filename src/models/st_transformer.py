"""Spatiotemporal transformer for station demand (our model).

Built on our from-scratch encoder (src/models/attention.py). Every change from the
plain Vaswani encoder is a config flag, so the ablation ladder is just five configs:

    step 1  plain       hourly tokens, global per-station scaling, post-LN
    step 2  + revin, patching (and Pre-LN)
    step 3  + time and station embeddings (with 2 future calendar tokens)
    step 4  + spatial attention across nearby stations
    step 5  + quantile head (10/50/90th percentiles, pinball loss)

Input batch (see src/data/dataset.py):
    x    (B, N, 168, 6)   departures, arrivals, temperature, precipitation, wind, humidity
    cal  (B, 192, 4)      hour, weekday, month(0-11), holiday for 168 past + 24 future hours
Output: (B, N, 24, 2, Q) in trips per hour, Q = number of quantiles (1 = median only).
"""
import math

import torch
import torch.nn as nn

from src.models.attention import EncoderLayer


def sinusoid(n, d):
    pos = torch.arange(n).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
    pe = torch.zeros(n, d)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def station_distances_km(coords):
    """coords (N, 2) as lat, lng in degrees -> (N, N) approximate distance in km."""
    lat, lng = torch.as_tensor(coords, dtype=torch.float32).T
    dy = (lat[:, None] - lat[None, :]) * 111.0
    dx = (lng[:, None] - lng[None, :]) * 111.0 * math.cos(math.radians(38.9))
    return torch.sqrt(dx ** 2 + dy ** 2)


class RevIN(nn.Module):
    """Reversible instance normalization (Kim et al., 2022), per station and channel."""

    def __init__(self, n_channels, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(n_channels))
        self.beta = nn.Parameter(torch.zeros(n_channels))

    def norm(self, x):                      # x (B, N, L, C)
        self.mu = x.mean(2, keepdim=True)
        self.sd = x.std(2, keepdim=True) + self.eps
        return (x - self.mu) / self.sd * self.gamma + self.beta

    def denorm(self, y):                    # y (B, N, H, C, Q)
        g, b = self.gamma[:, None], self.beta[:, None]
        return (y - b) / (g + self.eps) * self.sd.unsqueeze(-1) + self.mu.unsqueeze(-1)


class STBlock(nn.Module):
    """Temporal attention within each station, then (optionally) spatial attention across stations."""

    def __init__(self, d, heads, d_ff, dropout, pre_ln, spatial):
        super().__init__()
        self.temporal = EncoderLayer(d, heads, d_ff, dropout, pre_ln)
        self.spatial = EncoderLayer(d, heads, d_ff, dropout, pre_ln) if spatial else None
        if spatial:
            # learnable distance penalty per head: score = QK^T/sqrt(d) - alpha * distance
            self.log_alpha = nn.Parameter(torch.zeros(heads))

    def forward(self, h, spatial_bias=None):               # h (B, N, T, d)
        B, N, T, d = h.shape
        h = self.temporal(h.reshape(B * N, T, d)).view(B, N, T, d)
        if self.spatial is not None:
            dist, far = spatial_bias                       # (N, N) km, (N, N) bool
            alpha = torch.nn.functional.softplus(self.log_alpha)[:, None, None]
            bias = (-alpha * dist).masked_fill(far, float("-inf")).to(h.dtype)   # (heads, N, N)
            hs = h.transpose(1, 2).reshape(B * T, N, d)
            hs = self.spatial(hs, attn_bias=bias)
            h = hs.view(B, T, N, d).transpose(1, 2)
        return h


class STTransformer(nn.Module):
    def __init__(self, n_stations, coords, demand_mean, demand_std, in_len=168, out_len=24,
                 d_model=128, n_heads=8, n_layers=3, d_ff=256, dropout=0.1,
                 revin=True, patch_len=12, time_emb=True, station_emb=True,
                 spatial=True, k_neighbors=16, pre_ln=True, quantiles=(0.1, 0.5, 0.9)):
        super().__init__()
        self.in_len, self.out_len = in_len, out_len
        self.patch = patch_len if patch_len else 1
        self.n_past = in_len // self.patch
        self.n_future = out_len // self.patch if time_emb else 0
        self.Q = len(quantiles)
        self.use_revin, self.time_emb, self.spatial = revin, time_emb, spatial

        self.register_buffer("d_mean", torch.as_tensor(demand_mean, dtype=torch.float32))
        self.register_buffer("d_std", torch.as_tensor(demand_std, dtype=torch.float32))
        if revin:
            self.revin = RevIN(2)

        self.value_proj = nn.Linear(self.patch * 6, d_model)          # change 1: numeric inputs
        T = self.n_past + self.n_future
        self.register_buffer("pe", sinusoid(T, d_model))
        if time_emb:
            self.hour_emb = nn.Embedding(24, d_model)
            self.wday_emb = nn.Embedding(7, d_model)
            self.month_emb = nn.Embedding(12, d_model)
            self.hol_emb = nn.Embedding(2, d_model)
            self.future_token = nn.Parameter(torch.zeros(d_model))
        self.station = nn.Embedding(n_stations, d_model) if station_emb else None

        if spatial:
            dist = station_distances_km(coords)
            k = min(k_neighbors, n_stations)
            nearest = dist.topk(k, largest=False).indices
            far = torch.ones_like(dist, dtype=torch.bool)
            far.scatter_(1, nearest, False)                           # keep k nearest (incl. self)
            self.register_buffer("dist", dist)
            self.register_buffer("far", far)

        self.blocks = nn.ModuleList(
            STBlock(d_model, n_heads, d_ff, dropout, pre_ln, spatial) for _ in range(n_layers))
        self.final_norm = nn.LayerNorm(d_model) if pre_ln else nn.Identity()
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(T * d_model, out_len * 2 * self.Q)      # change 6: direct head

    def _calendar(self, cal):
        """cal (B, 192, 4) -> (B, T, d): calendar of the first hour of each token, holiday = any."""
        B = cal.shape[0]
        n_tok = (self.in_len + self.n_future * self.patch) // self.patch
        c = cal[:, : n_tok * self.patch].view(B, n_tok, self.patch, 4)
        first = c[:, :, 0]
        hol = c[..., 3].amax(-1)
        return (self.hour_emb(first[..., 0]) + self.wday_emb(first[..., 1]) +
                self.month_emb(first[..., 2]) + self.hol_emb(hol))

    def forward(self, x, cal):
        B, N, L, _ = x.shape
        demand, weather = x[..., :2], x[..., 2:]
        if self.use_revin:
            demand = self.revin.norm(demand)
        else:
            demand = (demand - self.d_mean[None, :, None]) / self.d_std[None, :, None]
        feats = torch.cat([demand, weather], -1)                          # (B, N, L, 6)

        tok = self.value_proj(feats.reshape(B, N, self.n_past, self.patch * 6))   # (B, N, P, d)
        if self.n_future:
            fut = self.future_token.expand(B, N, self.n_future, -1)
            tok = torch.cat([tok, fut], 2)
        tok = tok + self.pe
        if self.time_emb:
            tok = tok + self._calendar(cal)[:, None]
        if self.station is not None:
            tok = tok + self.station.weight[None, :, None]
        h = self.drop(tok)

        sb = (self.dist, self.far) if self.spatial else None
        for blk in self.blocks:
            h = blk(h, sb)
        h = self.final_norm(h)

        out = self.head(h.reshape(B, N, -1)).view(B, N, self.out_len, 2, self.Q)
        if self.use_revin:
            out = self.revin.denorm(out)
        else:
            out = out * self.d_std[None, :, None, :, None] + self.d_mean[None, :, None, :, None]
        return out
