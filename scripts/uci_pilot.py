"""UCI pilot: train a plain transformer encoder and an LSTM from scratch on the UCI
Bike Sharing hourly data (2011-2012, system-wide counts) and compare with seasonal baselines.

Task: previous 168 hours -> next 24 hours of total rentals (cnt).

Usage (from the repo root):
    python scripts/uci_pilot.py                      # both models, seed 0
    python scripts/uci_pilot.py --model transformer --epochs 30 --seed 1
    python scripts/uci_pilot.py --overfit            # sanity check: memorize one small batch

Data: Fanaee-T (2013), https://doi.org/10.24432/C5W894  (downloaded to data/raw/uci/)
Outputs: results/uci_pilot_metrics.csv, results/uci_pilot_loss.png, results/uci_pilot_history.csv
"""
import argparse
import io
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.attention import Encoder, PositionalEncoding  # noqa: E402
from src.models.lstm import LSTMForecaster  # noqa: E402

URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
RAW = Path("data/raw/uci")
RESULTS = Path("results")
IN_LEN, OUT_LEN = 168, 24
# casual + registered = cnt, so they are excluded (they would leak the target)
FEATURES = ["cnt", "temp", "atemp", "hum", "windspeed", "weathersit",
            "hr", "weekday", "mnth", "holiday", "workingday"]


# ------------------------------------------------------------------ data
def load_uci():
    path = RAW / "hour.csv"
    if not path.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {URL}")
        with urllib.request.urlopen(URL) as r:
            zipfile.ZipFile(io.BytesIO(r.read())).extract("hour.csv", RAW)
    df = pd.read_csv(path, parse_dates=["dteday"])
    df["time"] = df["dteday"] + pd.to_timedelta(df["hr"], unit="h")
    # ~165 hours are absent from the file (no rentals recorded) -> complete hourly grid
    full = pd.date_range(df["time"].min(), df["time"].max(), freq="h")
    df = df.set_index("time").reindex(full)
    n_missing = int(df["cnt"].isna().sum())
    df["cnt"] = df["cnt"].fillna(0)
    df[FEATURES] = df[FEATURES].ffill()
    df["hr"], df["weekday"], df["mnth"] = full.hour, (full.weekday + 1) % 7, full.month
    return df[FEATURES].astype(np.float32), n_missing


def make_windows(values, starts):
    x = np.stack([values[s:s + IN_LEN] for s in starts])
    y = np.stack([values[s + IN_LEN:s + IN_LEN + OUT_LEN, 0] for s in starts])
    return torch.from_numpy(x), torch.from_numpy(y)


def build_splits(df):
    """Chronological 70 / 15 / 15 split by target time. Scaler fit on train only."""
    T = len(df)
    t_train, t_val = int(0.70 * T), int(0.85 * T)
    raw = df.to_numpy()
    mean, std = raw[:t_train].mean(0), raw[:t_train].std(0) + 1e-6
    scaled = (raw - mean) / std
    starts = np.arange(0, T - IN_LEN - OUT_LEN + 1)
    end_of_target = starts + IN_LEN + OUT_LEN
    split_idx = {
        "train": starts[end_of_target <= t_train],
        "val": starts[(starts + IN_LEN >= t_train) & (end_of_target <= t_val)],
        "test": starts[starts + IN_LEN >= t_val],
    }
    data = {k: make_windows(scaled, v) for k, v in split_idx.items()}
    raw_y = {k: make_windows(raw, v)[1] for k, v in split_idx.items()}   # unscaled targets
    raw_x = {k: make_windows(raw, v)[0] for k, v in split_idx.items()}
    return data, raw_x, raw_y, (mean[0], std[0])


# ------------------------------------------------------------------ models
class PlainTransformer(nn.Module):
    """Ablation step 1: plain Vaswani encoder on hourly tokens.
    Each hour's features -> linear projection -> + sinusoidal PE -> encoder -> flatten -> 24 outputs."""

    def __init__(self, in_features, d_model=64, n_heads=4, n_layers=2, d_ff=128, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(in_features, d_model)
        self.pe = PositionalEncoding(d_model, max_len=IN_LEN)
        self.encoder = Encoder(n_layers, d_model, n_heads, d_ff, dropout, pre_ln=False)
        self.head = nn.Linear(IN_LEN * d_model, OUT_LEN)

    def forward(self, x):                               # (B, 168, F)
        h = self.encoder(self.pe(self.proj(x)))         # (B, 168, d)
        return self.head(h.flatten(1))                  # (B, 24)


def build_model(name, n_features):
    if name == "transformer":
        return PlainTransformer(n_features)
    return LSTMWrapper(n_features)


class LSTMWrapper(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = LSTMForecaster(n_features, horizon=OUT_LEN, n_targets=1, hidden=64, layers=2)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ------------------------------------------------------------------ training
def run_epoch(model, loader, opt, device, train):
    model.train(train)
    total, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = nn.functional.mse_loss(model(x), y)
            if train:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            total += loss.item() * len(x)
            n += len(x)
    return total / n


@torch.no_grad()
def predict(model, x, device, scale, bs=512):
    model.eval()
    out = torch.cat([model(x[i:i + bs].to(device)).cpu() for i in range(0, len(x), bs)])
    return out * scale[1] + scale[0]                    # back to rentals per hour


def mae_rmse(pred, y):
    err = pred - y
    return err.abs().mean().item(), err.pow(2).mean().sqrt().item()


def train_model(name, data, raw_y, scale, args, device):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    n_feat = data["train"][0].shape[-1]
    model = build_model(name, n_feat).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # linear warmup then cosine decay
    total_steps = args.epochs * int(np.ceil(len(data["train"][0]) / args.batch_size))
    warm = max(1, int(0.05 * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm,
                                              0.5 * (1 + np.cos(np.pi * min(1.0, s / total_steps)))))

    train_loader = DataLoader(TensorDataset(*data["train"]), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(*data["val"]), batch_size=512)

    history, best, best_state, bad = [], float("inf"), None, 0
    print(f"\n[{name}] {n_params:,} parameters, device={device}")
    for epoch in range(1, args.epochs + 1):
        t = time.time()
        model.train()
        tr_loss, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = nn.functional.mse_loss(model(x), y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tr_loss += loss.item() * len(x)
            n += len(x)
        tr_loss /= n
        va_loss = run_epoch(model, val_loader, None, device, train=False)
        va_mae, _ = mae_rmse(predict(model, data["val"][0], device, scale), raw_y["val"])
        history.append({"model": name, "epoch": epoch, "train_loss": tr_loss,
                        "val_loss": va_loss, "val_MAE": va_mae})
        print(f"  epoch {epoch:3d}  train {tr_loss:.4f}  val {va_loss:.4f}  "
              f"val MAE {va_mae:7.2f}  ({time.time() - t:.0f}s)")
        if va_mae < best:                                 # early stopping on validation MAE
            best, bad = va_mae, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print(f"  early stop (no val improvement for {args.patience} epochs)")
                break
    model.load_state_dict(best_state)
    return model, history, n_params


def overfit_check(data, device, args):
    """A correct model + training loop must drive loss near zero on 32 fixed windows."""
    x, y = data["train"][0][:32].to(device), data["train"][1][:32].to(device)
    for name in ["transformer", "lstm"]:
        torch.manual_seed(0)
        model = build_model(name, x.shape[-1]).to(device)
        if name == "transformer":                       # no dropout for memorization
            for m in model.modules():
                if isinstance(m, nn.Dropout):
                    m.p = 0.0
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for step in range(1, 1001):
            loss = nn.functional.mse_loss(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step in (1, 100, 500, 1000):
                print(f"[overfit {name}] step {step:4d}  loss {loss.item():.5f}")
        print(f"[overfit {name}] {'PASS' if loss.item() < 0.01 else 'FAIL'} (final loss {loss.item():.5f})\n")


def save_plot(history):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping loss plot")
        return
    h = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, g in h.groupby("model"):
        axes[0].plot(g["epoch"], g["train_loss"], label=f"{name} train")
        axes[0].plot(g["epoch"], g["val_loss"], "--", label=f"{name} val")
        axes[1].plot(g["epoch"], g["val_MAE"], label=name)
    axes[0].set(title="Loss (MSE, scaled)", xlabel="epoch")
    axes[1].set(title="Validation MAE (rentals/hour)", xlabel="epoch")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "uci_pilot_loss.png", dpi=120)
    print(f"Saved {RESULTS / 'uci_pilot_loss.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["transformer", "lstm", "both"], default="both")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overfit", action="store_true", help="only run the tiny-batch overfit check")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    df, n_missing = load_uci()
    data, raw_x, raw_y, scale = build_splits(df)
    print(f"UCI hourly: {len(df):,} hours ({n_missing} missing hours filled with 0 rentals)")
    print("windows: " + ", ".join(f"{k} {len(v[0]):,}" for k, v in data.items()))

    if args.overfit:
        overfit_check(data, device, args)
        return

    RESULTS.mkdir(exist_ok=True)
    rows, history = [], []
    # seasonal baselines (copy yesterday / last week), in raw rentals
    cnt = raw_x["test"][..., 0]
    for name, src in [("seasonal_day", cnt[:, IN_LEN - 24:]), ("seasonal_week", cnt[:, :OUT_LEN])]:
        mae, rmse = mae_rmse(src, raw_y["test"])
        rows.append({"model": name, "seed": "-", "params": 0, "test_MAE": mae, "test_RMSE": rmse})

    names = ["transformer", "lstm"] if args.model == "both" else [args.model]
    for name in names:
        model, hist, n_params = train_model(name, data, raw_y, scale, args, device)
        history += hist
        mae, rmse = mae_rmse(predict(model, data["test"][0], device, scale), raw_y["test"])
        rows.append({"model": name, "seed": args.seed, "params": n_params,
                     "test_MAE": mae, "test_RMSE": rmse})

    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "uci_pilot_metrics.csv", index=False)
    pd.DataFrame(history).to_csv(RESULTS / "uci_pilot_history.csv", index=False)
    print("\n=== TEST RESULTS (rentals per hour, 24 h ahead) ===")
    print(res.round(2).to_string(index=False))
    save_plot(history)


if __name__ == "__main__":
    main()
