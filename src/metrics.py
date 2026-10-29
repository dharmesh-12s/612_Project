"""Forecast metrics: MAE, RMSE, CRPS (from quantiles), interval coverage, pinball loss.

All metrics ignore masked targets (mask == 0).
Shapes:  y (B, N, 24, 2)   mask (B, N, 24)
         point forecast    pred (B, N, 24, 2)
         quantile forecast pred (B, N, 24, 2, 3)  for QUANTILES = 0.1, 0.5, 0.9
"""
import numpy as np
import pandas as pd
import torch

QUANTILES = (0.1, 0.5, 0.9)
HORIZONS = (1, 6, 12, 24)


def pinball_loss(pred_q, y, mask, quantiles=QUANTILES):
    """Quantile (pinball) loss, averaged over unmasked targets. Use this to train."""
    q = torch.tensor(quantiles, device=pred_q.device, dtype=pred_q.dtype)
    err = y.unsqueeze(-1) - pred_q                                  # (B, N, 24, 2, Q)
    loss = torch.maximum(q * err, (q - 1) * err).mean(-1)            # (B, N, 24, 2)
    m = mask.unsqueeze(-1).expand_as(loss)
    return (loss * m).sum() / m.sum().clamp(min=1)


class MetricAccumulator:
    """Collects error sums over many batches, then reports metrics per horizon and station group.

    groups: dict name -> bool array of length N (e.g. {"all": ..., "busy": ..., "quiet": ...})
    """

    def __init__(self, groups, horizon=24, quantiles=QUANTILES):
        self.groups = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in groups.items()}
        self.q = quantiles
        self.H = horizon
        z = lambda: {g: torch.zeros(horizon, dtype=torch.float64) for g in groups}
        self.n, self.abs, self.sq, self.crps, self.cover = z(), z(), z(), z(), z()
        self.has_quantiles = False

    @torch.no_grad()
    def update(self, pred, y, mask):
        pred, y, mask = pred.float().cpu(), y.float().cpu(), mask.float().cpu()
        if pred.dim() == y.dim() + 1:                                   # quantile forecast
            self.has_quantiles = True
            q = torch.tensor(self.q)
            err = y.unsqueeze(-1) - pred
            pin = torch.maximum(q * err, (q - 1) * err).mean(-1)        # (B, N, 24, 2)
            crps = 2 * pin                                              # quantile-score approx of CRPS
            inside = ((y >= pred[..., 0]) & (y <= pred[..., -1])).float()
            median = pred[..., len(self.q) // 2]
        else:
            crps, inside, median = None, None, pred

        abs_err = (median - y).abs().sum(-1)                            # sum over dep/arr -> (B, N, 24)
        sq_err = ((median - y) ** 2).sum(-1)
        for g, sel in self.groups.items():
            m = mask * sel[None, :, None]                               # (B, N, 24)
            self.n[g] += (2 * m).sum((0, 1)).double()                   # 2 targets per station-hour
            self.abs[g] += (abs_err * m).sum((0, 1)).double()
            self.sq[g] += (sq_err * m).sum((0, 1)).double()
            if crps is not None:
                self.crps[g] += (crps.sum(-1) * m).sum((0, 1)).double()
                self.cover[g] += (inside.sum(-1) * m).sum((0, 1)).double()

    def compute(self):
        rows = []
        for g in self.groups:
            n = self.n[g].clamp(min=1)
            for h in list(HORIZONS) + ["all"]:
                idx = slice(None) if h == "all" else h - 1
                N = n[idx].sum()
                row = {"group": g, "horizon": h,
                       "MAE": (self.abs[g][idx].sum() / N).item(),
                       "RMSE": np.sqrt((self.sq[g][idx].sum() / N).item())}
                if self.has_quantiles:
                    row["CRPS"] = (self.crps[g][idx].sum() / N).item()
                    row["coverage_10_90"] = (self.cover[g][idx].sum() / N).item()
                rows.append(row)
        return pd.DataFrame(rows)
