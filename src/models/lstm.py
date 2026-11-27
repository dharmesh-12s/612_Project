"""LSTM baseline: reads the input window, predicts the whole horizon at once from the last hidden state.

LSTMForecaster:  (B, L, F) -> (B, horizon, n_targets)       used by the UCI pilot
StationLSTM:     station batches (B, N, 168, 6) -> (B, N, 24, 2, 1)
                 stations are folded into the batch, so one LSTM is shared by all stations.
"""
import torch.nn as nn


class LSTMForecaster(nn.Module):
    def __init__(self, in_features, horizon=24, n_targets=1, hidden=64, layers=2, dropout=0.1):
        super().__init__()
        self.horizon, self.n_targets = horizon, n_targets
        self.lstm = nn.LSTM(in_features, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, horizon * n_targets)

    def forward(self, x):
        _, (h, _) = self.lstm(x)              # h: (layers, B, hidden)
        return self.head(h[-1]).view(-1, self.horizon, self.n_targets)


class StationLSTM(nn.Module):
    def __init__(self, demand_mean, demand_std, hidden=128, layers=2, dropout=0.1, out_len=24, **_):
        super().__init__()
        import torch
        self.register_buffer("d_mean", torch.as_tensor(demand_mean, dtype=torch.float32))
        self.register_buffer("d_std", torch.as_tensor(demand_std, dtype=torch.float32))
        self.out_len = out_len
        self.net = LSTMForecaster(6, horizon=out_len, n_targets=2, hidden=hidden,
                                  layers=layers, dropout=dropout)

    def forward(self, x, cal=None):
        B, N, L, F = x.shape
        m, s = self.d_mean[None, :, None], self.d_std[None, :, None]
        x = x.clone()
        x[..., :2] = (x[..., :2] - m) / s                  # per-station scaling (train stats)
        out = self.net(x.reshape(B * N, L, F)).view(B, N, self.out_len, 2)
        return (out * s + m).unsqueeze(-1)
