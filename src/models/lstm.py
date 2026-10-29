"""LSTM baseline: reads the input window, predicts the whole horizon at once from the last hidden state.

Input  (B, L, F)  ->  output (B, horizon, n_targets)
For station data, fold stations into the batch: (B, N, L, F) -> (B*N, L, F).
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
