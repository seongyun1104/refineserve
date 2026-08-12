from __future__ import annotations

import torch

from hardware.collect_llada2_router_trace import (
    block_attention_mask,
    extract_routes,
    prompts,
)


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


def test_native_prompt_classes_have_full_fixed_pool() -> None:
    assert len(prompts("reasoning")) == 32
    assert len(prompts("code")) == 32
    assert len(prompts("general")) >= 32
    assert set(prompts("reasoning")).isdisjoint(prompts("code"))


def test_extract_routes_reconstructs_selected_weights() -> None:
    logits = torch.tensor([[[0.0, 1.0, -1.0]]])
    topk = torch.tensor([[[1, 0]]])

    ids, weights = extract_routes((logits, topk), routed_scaling_factor=2.5)

    assert torch.equal(ids, topk)
    assert torch.allclose(weights.sum(dim=-1), torch.tensor([[2.5]]))
    assert weights[0, 0, 0] > weights[0, 0, 1]
