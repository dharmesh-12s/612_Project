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
    state.pt           latest training state for resuming; deleted when the run finishes (gitignored)

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


def bench(model, opt, win, train_ds, val_ds, tr, quantiles, amp, device, name, n_params):
    """Seconds per training step / eval batch and peak GPU memory, to plan the compute budget."""
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model.train()
    for i in range(23):
        if i == 3:
            sync()
            t = time.time()
        x, cal, y, mask = win.batch(np.random.choice(train_ds.origins, tr["batch_size"], replace=False))
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            loss = pinball_loss(model(x, cal).float(), y, mask, quantiles)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    sync()
    step = (time.time() - t) / 20
    model.eval()
    eb = tr.get("eval_batch", 8)
    with torch.no_grad():
        t = time.time()
        for i in range(5):
            x, cal, _, _ = win.batch(val_ds.origins[i * eb:(i + 1) * eb])
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                model(x, cal)
        sync()
    ev = (time.time() - t) / 5
    mem = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else float("nan")
    epoch_s = step * tr["steps_per_epoch"] + ev * len(val_ds.origins[::tr.get("val_stride", 24)]) / eb
    final_s = ev * 2 * len(val_ds.origins) / eb / tr.get("eval_stride", 1)          # val + test
    print(f"BENCH {name:16s} params {n_params:>9,}  step {step:6.3f}s  epoch ~{epoch_s / 60:5.1f} min  "
          f"final eval ~{final_s / 60:5.1f} min  peak mem {mem:5.2f} GB", flush=True)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-hours", type=float, default=10.0, help="total training-time budget (across resumes)")
    ap.add_argument("--out", default="results/runs")
    ap.add_argument("--smoke", action="store_true", help="tiny run to check the code end to end")
    ap.add_argument("--bench", action="store_true", help="time 20 training steps + 5 eval batches, report memory")
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

    if args.bench:
        bench(model, opt, win, train_ds, val_ds, tr, quantiles, amp, device, cfg["name"], n_params)
        return

    run_dir = Path(args.out) / cfg["name"] / f"seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "run_info.json").exists() and not args.smoke:
        print(f"{run_dir} is already finished; delete it to retrain.", flush=True)
        return
    val_sel = val_ds.origins[:: tr.get("val_stride", 24)]
    print(f"[{cfg['name']} seed {args.seed}] {n_params:,} params | device {device} | "
          f"{len(train_ds.origins):,} train windows | val check on {len(val_sel)} windows", flush=True)

    # resume after an interruption (scavenger jobs can be preempted and requeued)
    st = {"epoch": 0, "history": [], "best": float("inf"), "best_ep": 0, "bad": 0,
          "train_seconds": 0.0, "done": False}
    state_path = run_dir / "state.pt"
    if state_path.exists():
        ck = torch.load(state_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        np.random.set_state(ck["np_rng"])
        torch.set_rng_state(ck["torch_rng"])
        st = ck["st"]
        print(f"  resumed after epoch {st['epoch']} (best val MAE {st['best']:.4f})", flush=True)

    def save_state():
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "np_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(), "st": st},
                   state_path.with_suffix(".tmp"))
        state_path.with_suffix(".tmp").replace(state_path)          # atomic: never a half-written file

    t0 = time.time()
    for epoch in range(st["epoch"] + 1, tr["epochs"] + 1):
        if st["done"]:
            break
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
        st["train_seconds"] += time.time() - te
        st["history"].append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_MAE": val_mae,
                              "val_coverage": val_cov, "lr": sched.get_last_lr()[0],
                              "seconds": time.time() - te})
        pd.DataFrame(st["history"]).to_csv(run_dir / "history.csv", index=False)
        cov = "" if math.isnan(val_cov) else f"  cov {val_cov:.3f}"
        print(f"  epoch {epoch:3d}  loss {np.mean(losses):.4f}  val MAE {val_mae:.4f}{cov}  "
              f"({time.time() - te:.0f}s)", flush=True)
        st["epoch"] = epoch
        if val_mae < st["best"]:
            st["best"], st["best_ep"], st["bad"] = val_mae, epoch, 0
            torch.save(model.state_dict(), run_dir / "best.pt")
        else:
            st["bad"] += 1
            if st["bad"] >= tr.get("patience", 6):
                print(f"  early stop: no improvement for {st['bad']} epochs", flush=True)
                st["done"] = True
        if st["train_seconds"] / 3600 > args.max_hours:
            print("  stopping: training-time limit reached", flush=True)
            st["done"] = True
        save_state()
    history, best, best_ep = st["history"], st["best"], st["best_ep"]
    train_hours = st["train_seconds"] / 3600

    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device, weights_only=True))
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
    state_path.unlink(missing_ok=True)                                # optimizer state no longer needed
    print(f"Saved {run_dir}", flush=True)


if __name__ == "__main__":
    main()
