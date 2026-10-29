"""Windowed station-demand dataset (splits, scaling, windows, masks).

Each item follows the interface contract in PROJECT.md:
    x    (N, 168, 6)  float  departures, arrivals, temperature, precipitation, wind, humidity
    cal  (192, 4)     long   hour_of_day, weekday, month (0-11), holiday  for 168 past + 24 future hours
    y    (N, 24, 2)   float  future departures, arrivals (raw trip counts)
    mask (N, 24)      float  0 where the target is in an outage or the station wasn't open yet
A DataLoader adds the batch dimension in front.

Leakage rules:
    - x only contains past hours (demand and observed weather).
    - Future hours appear only in cal (calendar features).
    - Weather scaler is fit on the training years only.
    - Splits are by the hour being predicted: train 2021-2023, val 2024, test 2025.

Usage:
    from src.data.dataset import make_loader
    train_loader = make_loader("train", batch_size=16)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

DATA = Path("data/processed")
IN_LEN, OUT_LEN = 168, 24
WEATHER = ["temperature", "precipitation", "wind", "humidity"]
CAL = ["hour_of_day", "weekday", "month", "holiday"]
SPLITS = {  # by the target (predicted) hours
    "train": ("2021-01-01", "2023-12-31 23:00"),
    "val": ("2024-01-01", "2024-12-31 23:00"),
    "test": ("2025-01-01", "2025-12-31 23:00"),
}


class BikeDataset(Dataset):
    def __init__(self, split, stride=1, data_dir=DATA):
        data_dir = Path(data_dir)
        self.demand = np.load(data_dir / "demand.npy")          # (N, T, 2) uint16
        hours = pd.read_csv(data_dir / "hours.csv", parse_dates=["hour"])
        stations = pd.read_csv(data_dir / "stations.csv", dtype={"station_id": str},
                               parse_dates=["first_seen", "last_seen"])
        times = hours["hour"]
        N, T, _ = self.demand.shape

        # weather: z-score with training-period stats only
        wx = hours[WEATHER].to_numpy(np.float32)
        train_rows = (times >= SPLITS["train"][0]) & (times <= SPLITS["train"][1])
        self.wx_mean = wx[train_rows].mean(0)
        self.wx_std = wx[train_rows].std(0) + 1e-6
        self.weather = (wx - self.wx_mean) / self.wx_std          # (T, 4)

        cal = hours[CAL].to_numpy(np.int64)
        cal[:, 2] -= 1                                            # month 1-12 -> 0-11
        self.cal = cal                                            # (T, 4)

        # valid target hours: not an outage, and station open (first..last trip day)
        t = times.to_numpy()
        is_open = ((t[None, :] >= stations["first_seen"].to_numpy()[:, None]) &
                   (t[None, :] < (stations["last_seen"] + pd.Timedelta("1D")).to_numpy()[:, None]))
        self.valid = (is_open & (hours["outage"].to_numpy() == 0)[None, :]).astype(np.float32)

        # per-station demand stats from training years (for models without RevIN, e.g. LSTM)
        train_demand = self.demand[:, train_rows.to_numpy(), :].astype(np.float32)
        self.demand_mean = train_demand.mean(1)                   # (N, 2)
        self.demand_std = train_demand.std(1) + 1e-6

        # forecast origins: index of the first predicted hour
        lo, hi = SPLITS[split]
        first = np.searchsorted(t, np.datetime64(pd.Timestamp(lo)))
        last = np.searchsorted(t, np.datetime64(pd.Timestamp(hi)), side="right") - OUT_LEN
        first = max(first, IN_LEN)
        self.origins = np.arange(first, last + 1, stride)

        self.split, self.N, self.times = split, N, times

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, i):
        o = self.origins[i]
        past, fut = slice(o - IN_LEN, o), slice(o, o + OUT_LEN)

        demand_past = self.demand[:, past, :].astype(np.float32)              # (N, 168, 2)
        wx_past = np.broadcast_to(self.weather[past], (self.N, IN_LEN, 4))    # (N, 168, 4)
        x = np.concatenate([demand_past, wx_past], axis=-1)                   # (N, 168, 6)

        return {
            "x": torch.from_numpy(x),
            "cal": torch.from_numpy(self.cal[o - IN_LEN:o + OUT_LEN]),        # (192, 4)
            "y": torch.from_numpy(self.demand[:, fut, :].astype(np.float32)),  # (N, 24, 2)
            "mask": torch.from_numpy(self.valid[:, fut]),                     # (N, 24)
        }


def make_loader(split, batch_size=16, stride=1, shuffle=None, num_workers=0, data_dir=DATA):
    ds = BikeDataset(split, stride=stride, data_dir=data_dir)
    if shuffle is None:
        shuffle = split == "train"
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        ds = BikeDataset(split)
        b = next(iter(DataLoader(ds, batch_size=4)))
        o = ds.origins
        print(f"{split:5s}: {len(ds):,} windows, first target {ds.times[o[0]]}, "
              f"last target {ds.times[o[-1] + OUT_LEN - 1]}")
        for k, v in b.items():
            print(f"       {k:4s} {tuple(v.shape)} {v.dtype}")
        print(f"       masked targets in split: {1 - ds.valid[:, o[0]:o[-1] + OUT_LEN].mean():.1%}")
