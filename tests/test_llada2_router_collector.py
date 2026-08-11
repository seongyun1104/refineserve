from __future__ import annotations

import torch

from hardware.collect_llada2_router_trace import block_attention_mask


def test_block_attention_mask_matches_block_causal_semantics() -> None:
    mask = block_attention_mask(prefix_length=4, block_width=2, device="cpu")

    assert mask.shape == (1, 1, 6, 6)
    allowed = torch.isfinite(mask[0, 0])
    assert allowed[0].tolist() == [True, True, False, False, False, False]
    assert allowed[2].tolist() == [True, True, True, True, False, False]
    assert allowed[5].tolist() == [True, True, True, True, True, True]


def test_block_attention_mask_requires_aligned_prefix() -> None:
    try:
        block_attention_mask(prefix_length=3, block_width=2, device="cpu")
    except ValueError as error:
        assert "divisible" in str(error)
    else:
        raise AssertionError("unaligned prefix must be rejected")
