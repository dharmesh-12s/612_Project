"""Shape and sanity checks for every model on small random batches (no data files needed).

Run:  python -m pytest tests/test_models.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.baselines import DLinear, ITransformer, PatchTST  # noqa: E402
from src.models.lstm import StationLSTM  # noqa: E402
from src.models.st_transformer import STTransformer  # noqa: E402

B, N = 2, 20


def batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(B, N, 168, 6, generator=g) * 5
    cal = torch.stack([torch.randint(0, k, (B, 192), generator=g) for k in (24, 7, 12, 2)], -1)
    return x, cal


def st(**kw):
    rng = np.random.default_rng(0)
    coords = np.c_[38.9 + rng.normal(0, .02, N), -77.03 + rng.normal(0, .02, N)]
    base = dict(n_stations=N, coords=coords, demand_mean=np.ones((N, 2)), demand_std=np.ones((N, 2)),
                d_model=32, n_heads=4, n_layers=2, d_ff=64, k_neighbors=5)
    base.update(kw)
    return STTransformer(**base)


LADDER = {
    "st1_plain": dict(revin=False, patch_len=0, time_emb=False, station_emb=False, spatial=False,
                      pre_ln=False, quantiles=(0.5,)),
    "st2_revin_patch": dict(time_emb=False, station_emb=False, spatial=False, quantiles=(0.5,)),
    "st3_embeddings": dict(spatial=False, quantiles=(0.5,)),
    "st4_spatial": dict(quantiles=(0.5,)),
    "st5_quantile": dict(quantiles=(0.1, 0.5, 0.9)),
}


@pytest.mark.parametrize("name", LADDER)
def test_ladder_output_shape(name):
    model = st(**LADDER[name]).eval()
    x, cal = batch()
    out = model(x, cal)
    Q = len(LADDER[name]["quantiles"])
    assert out.shape == (B, N, 24, 2, Q)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("cls", [DLinear, PatchTST, ITransformer])
def test_baseline_output_shape(cls):
    out = cls().eval()(*batch())
    assert out.shape == (B, N, 24, 2, 1)


def test_lstm_output_shape():
    out = StationLSTM(np.ones((N, 2)), np.ones((N, 2)), hidden=16).eval()(*batch())
    assert out.shape == (B, N, 24, 2, 1)


def test_spatial_attention_only_sees_neighbors():
    """Changing a station outside every other station's 5-nearest set must not change
    forecasts of stations that don't have it as a neighbor."""
    torch.manual_seed(0)
    model = st(quantiles=(0.5,), n_layers=1).eval()
    x, cal = batch()
    far = model.far                                   # (N, N) True = not a neighbor
    target = 0
    unaffected = [i for i in range(N) if far[i, target] and i != target]
    x2 = x.clone()
    x2[:, target] += 10.0
    with torch.no_grad():
        a, b = model(x, cal), model(x2, cal)
    assert torch.allclose(a[:, unaffected], b[:, unaffected], atol=1e-5)
    assert not torch.allclose(a[:, target], b[:, target])


def test_future_calendar_used_but_not_future_values():
    """The model input has no future demand at all; only future calendar enters via cal[:, 168:]."""
    model = st(quantiles=(0.5,)).eval()
    x, cal = batch()
    cal[:, 168:, 3] = 0
    cal2 = cal.clone()
    cal2[:, 168:, 3] = 1                              # make the forecast day a holiday
    with torch.no_grad():
        assert not torch.allclose(model(x, cal), model(x, cal2))


def test_can_overfit_tiny_batch():
    torch.manual_seed(0)
    model = st(quantiles=(0.5,), dropout=0.0)
    x, cal = batch()
    y = torch.rand(B, N, 24, 2) * 5
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(300):
        loss = (model(x, cal)[..., 0] - y).abs().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.3, loss.item()
