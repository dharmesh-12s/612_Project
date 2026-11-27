"""Shape and leakage checks for the station dataset (needs data/processed/; skipped otherwise).

Run:  python -m pytest tests/test_dataset.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.dataset import IN_LEN, OUT_LEN, SPLITS, BikeDataset  # noqa: E402

DATA = ROOT / "data" / "processed"
pytestmark = pytest.mark.skipif(not (DATA / "demand.npy").exists(), reason="data/processed missing")


@pytest.fixture(scope="module")
def splits():
    return {s: BikeDataset(s, data_dir=DATA) for s in ("train", "val", "test")}


def test_item_shapes(splits):
    ds = splits["val"]
    item = ds[0]
    assert item["x"].shape == (ds.N, IN_LEN, 6)
    assert item["cal"].shape == (IN_LEN + OUT_LEN, 4)
    assert item["y"].shape == (ds.N, OUT_LEN, 2)
    assert item["mask"].shape == (ds.N, OUT_LEN)


def test_targets_stay_inside_split(splits):
    for name, ds in splits.items():
        lo, hi = (pd.Timestamp(t) for t in SPLITS[name])
        first = ds.times[ds.origins[0]]
        last = ds.times[ds.origins[-1] + OUT_LEN - 1]
        assert first >= lo and last <= hi, name


def test_no_target_overlap_between_splits(splits):
    def target_hours(ds):
        return set(range(ds.origins[0], ds.origins[-1] + OUT_LEN))
    tr, va, te = (target_hours(splits[s]) for s in ("train", "val", "test"))
    assert not (tr & va) and not (va & te) and not (tr & te)


@pytest.mark.parametrize("i", [0, 1000, -1])
def test_inputs_are_strictly_past(splits, i):
    """x must equal the 168 hours before the origin and y the 24 hours from it."""
    ds = splits["test"]
    o = ds.origins[i]
    item = ds[i if i >= 0 else len(ds) - 1]
    assert np.array_equal(item["x"][..., :2].numpy(), ds.demand[:, o - IN_LEN:o])
    assert np.array_equal(item["y"].numpy(), ds.demand[:, o:o + OUT_LEN])
    assert np.allclose(item["x"][0, :, 2:].numpy(), ds.weather[o - IN_LEN:o])   # weather: past only


def test_weather_scaler_uses_train_years_only(splits):
    ds = splits["test"]
    t = ds.times
    rows = ((t >= SPLITS["train"][0]) & (t <= SPLITS["train"][1])).to_numpy()
    assert np.allclose(ds.weather[rows].mean(0), 0, atol=1e-3)
    assert np.allclose(ds.weather[rows].std(0), 1, atol=1e-2)


def test_mask_hides_stations_before_opening(splits):
    ds = splits["train"]
    st = pd.read_csv(DATA / "stations.csv", parse_dates=["first_seen"])
    late = st.index[st["first_seen"] > "2022-01-01"][0]
    opened = int(np.searchsorted(ds.times.to_numpy(), st.loc[late, "first_seen"].to_datetime64()))
    assert ds.valid[late, :opened].sum() == 0
    assert ds.valid[late, opened:opened + 24].sum() > 0
