"""Evaluate a forecaster on a split, the same way for every model.

Usage:
    python -m src.evaluate --model seasonal_day  --split val
    python -m src.evaluate --model seasonal_week --split val

Writes results/<model>_<split>.csv with MAE / RMSE (and CRPS / coverage for
quantile models) at horizons 1, 6, 12, 24 and overall, for all / busy / quiet stations.
Busy = stations above the median training-period departure rate.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from src.data.dataset import IN_LEN, OUT_LEN, make_loader
from src.metrics import MetricAccumulator


# ---- seasonal naive baselines: copy the past, nothing learned ----
def seasonal_day(batch):
    """Same hour, previous day: target hour t uses t - 24."""
    return batch["x"][:, :, IN_LEN - 24:IN_LEN - 24 + OUT_LEN, :2]


def seasonal_week(batch):
    """Same hour, previous week: target hour t uses t - 168."""
    return batch["x"][:, :, IN_LEN - 168:IN_LEN - 168 + OUT_LEN, :2]


MODELS = {"seasonal_day": seasonal_day, "seasonal_week": seasonal_week}


def station_groups(ds):
    rate = ds.demand_mean[:, 0]                       # mean training departures per hour
    busy = rate > np.median(rate)
    return {"all": np.ones_like(busy), "busy": busy, "quiet": ~busy}


@torch.no_grad()
def evaluate(model_fn, split="val", batch_size=32, stride=1):
    loader = make_loader(split, batch_size=batch_size, stride=stride, shuffle=False)
    acc = MetricAccumulator(station_groups(loader.dataset))
    for batch in loader:
        acc.update(model_fn(batch), batch["y"], batch["mask"])
    return acc.compute()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--stride", type=int, default=1, help="use every k-th window (1 = all)")
    args = ap.parse_args()

    t = time.time()
    df = evaluate(MODELS[args.model], args.split, stride=args.stride)
    out = Path("results") / f"{args.model}_{args.split}.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(df.round(3).to_string(index=False))
    print(f"\nSaved {out}  ({time.time() - t:.0f}s)")


if __name__ == "__main__":
    main()
