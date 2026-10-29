"""Check our from-scratch attention against PyTorch's built-in version with shared weights.

Run:  python -m pytest tests/test_attention.py -v
  or: python tests/test_attention.py
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.attention import EncoderLayer, MultiHeadAttention  # noqa: E402

B, L, D, H = 4, 20, 64, 8
ATOL = 1e-5


def copy_mha(ours, theirs):
    """Copy torch.nn.MultiheadAttention weights into our module."""
    Wq, Wk, Wv = theirs.in_proj_weight.chunk(3)
    bq, bk, bv = theirs.in_proj_bias.chunk(3)
    with torch.no_grad():
        for lin, w, b in [(ours.w_q, Wq, bq), (ours.w_k, Wk, bk), (ours.w_v, Wv, bv),
                          (ours.w_o, theirs.out_proj.weight, theirs.out_proj.bias)]:
            lin.weight.copy_(w)
            lin.bias.copy_(b)


def make_pair():
    torch.manual_seed(0)
    theirs = nn.MultiheadAttention(D, H, dropout=0.0, batch_first=True).eval()
    ours = MultiHeadAttention(D, H).eval()
    copy_mha(ours, theirs)
    return ours, theirs


def test_self_attention_matches():
    ours, theirs = make_pair()
    x = torch.randn(B, L, D)
    out_t, w_t = theirs(x, x, x, average_attn_weights=False)
    out_o, w_o = ours(x, x, x, need_weights=True)
    assert torch.allclose(out_o, out_t, atol=ATOL)
    assert torch.allclose(w_o, w_t, atol=ATOL)


def test_cross_attention_matches():
    ours, theirs = make_pair()
    q, kv = torch.randn(B, 7, D), torch.randn(B, L, D)
    assert torch.allclose(ours(q, kv, kv), theirs(q, kv, kv)[0], atol=ATOL)


def test_causal_mask_matches():
    ours, theirs = make_pair()
    x = torch.randn(B, L, D)
    causal = torch.triu(torch.full((L, L), float("-inf")), diagonal=1)
    assert torch.allclose(ours(x, x, x, attn_bias=causal),
                          theirs(x, x, x, attn_mask=causal)[0], atol=ATOL)


def test_padding_mask_matches():
    ours, theirs = make_pair()
    x = torch.randn(B, L, D)
    pad = torch.zeros(B, L, dtype=torch.bool)
    pad[:, -5:] = True
    assert torch.allclose(ours(x, x, x, key_padding_mask=pad),
                          theirs(x, x, x, key_padding_mask=pad)[0], atol=ATOL)


def test_encoder_layer_matches_pytorch():
    """Whole post-LN block vs nn.TransformerEncoderLayer (original Vaswani layout)."""
    torch.manual_seed(0)
    theirs = nn.TransformerEncoderLayer(D, H, dim_feedforward=128, dropout=0.0,
                                        batch_first=True, norm_first=False).eval()
    ours = EncoderLayer(D, H, d_ff=128, dropout=0.0, pre_ln=False).eval()
    copy_mha(ours.attn, theirs.self_attn)
    with torch.no_grad():
        ours.ff[0].load_state_dict(theirs.linear1.state_dict())
        ours.ff[3].load_state_dict(theirs.linear2.state_dict())
        ours.norm1.load_state_dict(theirs.norm1.state_dict())
        ours.norm2.load_state_dict(theirs.norm2.state_dict())
    x = torch.randn(B, L, D)
    assert torch.allclose(ours(x), theirs(x), atol=ATOL)


def test_gradients_flow():
    ours, _ = make_pair()
    x = torch.randn(B, L, D, requires_grad=True)
    ours(x, x, x).sum().backward()
    assert x.grad is not None and all(p.grad is not None for p in ours.parameters())


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
