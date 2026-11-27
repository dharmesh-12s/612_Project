"""Train and evaluate any model on the station data from one YAML config.

Usage:
    python -m src.train --config configs/st5_quantile.yaml --seed 0
    python -m src.train --config configs/st5_quantile.yaml --seed 0 --smoke   # 2 tiny epochs, CPU ok

Per run it writes results/runs/<name>/seed<k>/:
    history.csv        train loss, val MAE (and coverage) per epoch  -> loss curves
    metrics_val.csv    final metrics on 2024 (all / busy / quiet, horizons 1, 6, 12, 24)
    metrics_test.csv   final metrics on 2025
    run_info.json      config, parameters, best epoch, training time
    best.pt            best checkpoint by validation MAE (gitignored)

Training details: AdamW, linear warmup + cosine decay, gradient clipping 1.0, bf16 autocast on
GPU, early stopping on validation MAE. Loss = pinball loss over the model's quantiles
(for point models the 0.5 quantile, i.e. MAE). Masked targets (outages, stations not yet
open) are ignored. Windows are sliced directly on the GPU for speed.
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.data.dataset import IN_LEN, OUT_LEN, BikeDataset
from src.evaluate import station_groups
from src.metrics import MetricAccumulator, pinball_loss


# ----------------------------------------------------------------------------- data on GPU
class Windows:
    """Holds the full arrays on the device and cuts (x, cal, y, mask) batches by origin index."""

    def __init__(self, ds, device):
        self.demand = torch.from_numpy(ds.demand.astype(np.float32)).to(device)   # (N, T, 2)
        self.weather = torch.from_numpy(ds.weather).to(device)                      # (T, 4)
        self.cal = torch.from_numpy(ds.cal).to(device)                              # (T, 4)
        self.valid = torch.from_numpy(ds.valid).to(device)                          # (N, T)
        self.N = self.demand.shape[0]
        self.device = device
        self.past = torch.arange(-IN_LEN, 0, device=device)
        self.fut = torch.arange(0, OUT_LEN, device=device)

    def batch(self, origins):
        o = torch.as_tensor(origins, device=self.device)
        tp = o[:, None] + self.past                                    # (B, 168)
        tf = o[:, None] + self.fut                                     # (B, 24)
        B = len(o)
        d_past = self.demand[:, tp].permute(1, 0, 2, 3)                # (B, N, 168, 2)
        wx = self.weather[tp][:, None].expand(B, self.N, IN_LEN, 4)
        x = torch.cat([d_past, wx], -1)
        cal = self.cal[torch.cat([tp, tf], 1)]                         # (B, 192, 4)
        y = self.demand[:, tf].permute(1, 0, 2, 3)                     # (B, N, 24, 2)
        mask = self.valid[:, tf].permute(1, 0, 2)                      # (B, N, 24)
        return x, cal, y, mask


# ----------------------------------------------------------------------------- models
def build_model(cfg, ds, quantiles):
    m = dict(cfg["model"])
    kind = m.pop("type")
    stats = dict(demand_mean=ds.demand_mean, demand_std=ds.demand_std)
    if kind == "st_transformer":
        from src.models.st_transformer import STTransformer
        st = pd.read_csv(Path(ds.data_dir) / "stations.csv")
        return STTransformer(n_stations=ds.N, coords=st[["lat", "lng"]].to_numpy(),
                             quantiles=quantiles, **stats, **m)
    if kind == "lstm":
        from src.models.lstm import StationLSTM
        return StationLSTM(**stats, **m)
    from src.models.baselines import DLinear, ITransformer, PatchTST
    return {"dlinear": DLinear, "patchtst": PatchTST, "itransformer": ITransformer}[kind](**m)


# ----------------------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(model, win, origins, groups, batch_size, amp):
    model.eval()
    acc = MetricAccumulator(groups)
    for i in range(0, len(origins), batch_size):
        x, cal, y, mask = win.batch(origins[i:i + batch_size])
        with torch.autocast(win.device.type, dtype=torch.bfloat16, enabled=amp):
            pred = model(x, cal)
        pred = pred.float().sort(-1).values                            # no quantile crossing
        if pred.shape[-1] == 1:
            pred = pred[..., 0]
        acc.update(pred, y, mask)
    return acc.compute()


def headline(df):
    row = df[(df.group == "all") & (df.horizon == "all")].iloc[0]
    return row["MAE"], row.get("coverage_10_90", float("nan"))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-hours", type=float, default=10.0, help="stop training after this long")
    ap.add_argument("--out", default="results/runs")
    ap.add_argument("--smoke", action="store_true", help="tiny run to check the code end to end")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    tr = cfg["train"]
    if args.smoke:
        tr.update(epochs=2, steps_per_epoch=3, batch_size=1, eval_batch=1,
                  val_stride=24 * 60, eval_stride=24 * 60)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"

    train_ds, val_ds, test_ds = (BikeDataset(s) for s in ("train", "val", "test"))
    win = Windows(train_ds, device)            # same arrays for every split; only origins differ
    groups = station_groups(train_ds)
    quantiles = tuple(tr.get("quantiles", [0.5]))

    model = build_model(cfg, train_ds, quantiles).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"], weight_decay=tr.get("weight_decay", 1e-4))
    total = tr["epochs"] * tr["steps_per_epoch"]
    warm = max(1, int(tr.get("warmup_frac", 0.05) * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, 0.5 * (1 + math.cos(math.pi * min(1.0, s / total)))))

    run_dir = Path(args.out) / cfg["name"] / f"seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    val_sel = val_ds.origins[:: tr.get("val_stride", 24)]
    print(f"[{cfg['name']} seed {args.seed}] {n_params:,} params | device {device} | "
          f"{len(train_ds.origins):,} train windows | val check on {len(val_sel)} windows", flush=True)

    history, best, best_ep, bad, t0 = [], float("inf"), 0, 0, time.time()
    for epoch in range(1, tr["epochs"] + 1):
        model.train()
        te, losses = time.time(), []
        for _ in range(tr["steps_per_epoch"]):
            o = np.random.choice(train_ds.origins, tr["batch_size"], replace=False)
            x, cal, y, mask = win.batch(o)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                pred = model(x, cal)
            loss = pinball_loss(pred.float(), y, mask, quantiles)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
        val_mae, val_cov = headline(evaluate(model, win, val_sel, groups, tr.get("eval_batch", 8), amp))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_MAE": val_mae,
                        "val_coverage": val_cov, "lr": sched.get_last_lr()[0],
                        "seconds": time.time() - te})
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        cov = "" if math.isnan(val_cov) else f"  cov {val_cov:.3f}"
        print(f"  epoch {epoch:3d}  loss {np.mean(losses):.4f}  val MAE {val_mae:.4f}{cov}  "
              f"({time.time() - te:.0f}s)", flush=True)
        if val_mae < best:
            best, best_ep, bad = val_mae, epoch, 0
            torch.save(model.state_dict(), run_dir / "best.pt")
        else:
            bad += 1
            if bad >= tr.get("patience", 6):
                print(f"  early stop: no improvement for {bad} epochs", flush=True)
                break
        if (time.time() - t0) / 3600 > args.max_hours:
            print("  stopping: time limit reached", flush=True)
            break
    train_hours = (time.time() - t0) / 3600

    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    es = tr.get("eval_stride", 1)
    for name, ds in [("val", val_ds), ("test", test_ds)]:
        df = evaluate(model, win, ds.origins[::es], groups, tr.get("eval_batch", 8), amp)
        df.to_csv(run_dir / f"metrics_{name}.csv", index=False)
        mae, cov = headline(df)
        cov = "" if math.isnan(cov) else f", coverage {cov:.3f}"
        print(f"  final {name}: MAE {mae:.4f}{cov}", flush=True)

    info = {"name": cfg["name"], "seed": args.seed, "params": n_params, "best_epoch": best_ep,
            "epochs_run": len(history), "best_val_MAE_check": best, "train_hours": round(train_hours, 3),
            "device": str(device), "config": cfg}
    json.dump(info, open(run_dir / "run_info.json", "w"), indent=2)
    print(f"Saved {run_dir}", flush=True)


if __name__ == "__main__":
    main()
