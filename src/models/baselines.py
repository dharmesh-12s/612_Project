"""Neural baselines, re-implemented compactly from the papers and trained from scratch on our data.

    DLinear       Zeng et al. (2023)  trend/seasonal decomposition + one linear layer per part
    PatchTST      Nie et al. (2023)   channel-independent patches + transformer encoder + RevIN
    iTransformer  Liu et al. (2024)   each whole series is one token; attention across series

All take the station batch x (B, N, 168, 6) and return (B, N, 24, 2, 1) (median forecast).
Following the papers, DLinear and PatchTST are univariate per series (departures and
arrivals of each station modeled separately, no weather); iTransformer sees every series
(all stations' departures and arrivals plus the 4 weather series).
These baselines use PyTorch's built-in nn.TransformerEncoder; our own model uses our
from-scratch attention.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _series(x):
    """(B, N, L, 6) -> demand series (B, N*2, L)."""
    B, N, L, _ = x.shape
    return x[..., :2].permute(0, 1, 3, 2).reshape(B, N * 2, L)


def _unseries(y, B, N):
    """(B, N*2, H) -> (B, N, H, 2, 1)."""
    H = y.shape[-1]
    return y.view(B, N, 2, H).permute(0, 1, 3, 2).unsqueeze(-1)


def _instance_norm(s, eps=1e-5):
    mu = s.mean(-1, keepdim=True)
    sd = s.std(-1, keepdim=True) + eps
    return (s - mu) / sd, mu, sd


class DLinear(nn.Module):
    def __init__(self, in_len=168, out_len=24, kernel=25, **_):
        super().__init__()
        self.kernel = kernel
        self.seasonal = nn.Linear(in_len, out_len)
        self.trend = nn.Linear(in_len, out_len)

    def forward(self, x, cal=None):
        B, N = x.shape[:2]
        s = _series(x)                                                   # (B, S, L)
        pad = (self.kernel - 1) // 2
        padded = torch.cat([s[..., :1].expand(-1, -1, pad), s, s[..., -1:].expand(-1, -1, pad)], -1)
        trend = F.avg_pool1d(padded, self.kernel, stride=1)              # moving average
        y = self.seasonal(s - trend) + self.trend(trend)
        return _unseries(y, B, N)


class PatchTST(nn.Module):
    def __init__(self, in_len=168, out_len=24, patch_len=16, stride=8, d_model=128, n_heads=8,
                 n_layers=3, d_ff=256, dropout=0.2, **_):
        super().__init__()
        self.patch_len, self.stride = patch_len, stride
        n_patch = (in_len - patch_len) // stride + 2                       # +1 from end padding
        self.embed = nn.Linear(patch_len, d_model)
        self.pos = nn.Parameter(torch.randn(n_patch, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(n_patch * d_model, out_len)

    def forward(self, x, cal=None):
        B, N = x.shape[:2]
        s, mu, sd = _instance_norm(_series(x))                             # RevIN
        S = s.shape[1]
        s = torch.cat([s, s[..., -1:].expand(-1, -1, self.stride)], -1)    # pad end
        p = s.unfold(-1, self.patch_len, self.stride)                      # (B, S, P, patch)
        h = self.embed(p).reshape(B * S, -1, self.pos.shape[1]) + self.pos
        h = self.encoder(h)
        y = self.head(h.reshape(B, S, -1)) * sd + mu
        return _unseries(y, B, N)


class ITransformer(nn.Module):
    def __init__(self, in_len=168, out_len=24, d_model=128, n_heads=8, n_layers=3, d_ff=256,
                 dropout=0.1, **_):
        super().__init__()
        self.embed = nn.Linear(in_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True,
                                           norm_first=False)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d_model, out_len)

    def forward(self, x, cal=None):
        B, N = x.shape[:2]
        demand = _series(x)                                                # (B, 2N, L)
        weather = x[:, 0, :, 2:].transpose(1, 2)                           # (B, 4, L), same for all stations
        s, mu, sd = _instance_norm(torch.cat([demand, weather], 1))
        h = self.encoder(self.embed(s))                                    # one token per series
        y = self.head(h) * sd + mu
        return _unseries(y[:, : 2 * N], B, N)
