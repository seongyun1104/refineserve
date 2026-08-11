# Accounting changelog

## 2026-08-09 — Explicit SwiGLU and native-denominator audit

- Expert MLP FLOPs are now declared as three matmuls (`gate_proj`, `up_proj`,
  `down_proj`) at two FLOPs per multiply-accumulate: `6 * tokens * H * N`.
- The generated v1 shape artifact already used factor 6 for both toy and LLaDA2-mini,
  but its metadata did not declare the matmul count. Earlier reviewer recalculations
  using factor 4 therefore understated toy compute by exactly 1.5x. This was a
  provenance omission, not a toy-only formula change.
- Native LLaDA2 sensitivity is no longer represented by one assignment-granular,
  EP-only row. It now reports the cross product of assignment-granular versus
  destination-coalesced communication and EP-only versus full-iteration denominator.
- Native scheduler authorization is preregistered against
  `destination_coalesced × full_iteration`. The coalesced EP-only row is a mechanism
  diagnostic; assignment-granular rows are correctness-baseline sensitivity only.
- Full-iteration sensitivity uses the official immutable config values: hidden 2048,
  20 attention layers, one dense SwiGLU layer with intermediate 5120, 19 sparse
  layers with routed/shared intermediate 512, 16 query heads, four KV heads, head
  dimension 128, and vocabulary 157184. It includes attention projection/matmul,
  dense MLP, router projection, and LM-head FLOPs, while explicitly omitting
  non-matmul kernels.
- Stock LLaDA2 recomputes the full prefix plus current block at every denoising step
  (`use_cache=False`). Native trace accounting therefore separates controlled block
  width, model-forward positions, and remaining masked positions.
