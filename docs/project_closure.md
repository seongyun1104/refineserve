# Project closure

Date: 2026-08-13

Status: **COMPLETED NEGATIVE-RESULT STUDY — PRIMARY HYPOTHESIS NOT SUPPORTED**

## Decision

RefineServe is closed at M2.1. No further GPU rental, native-shape timing replay,
full-model Expert Parallel adapter, or serving-engine upstream work is authorized for
the current hypothesis.

The tested causal chain was:

```text
native position-parallel refinement
  -> request-specific MoE routing structure
  -> batch composition changes EP-rank load
  -> critical-path reduction
  -> wall-clock improvement
```

The measured result was:

```text
temporal route persistence                 high
inter-request route differentiation        weak
best-found composition headroom            < 0.5% in every measured cell
existing hardware measurement target       2%
proxy-to-time transfer                      unresolved
```

Route prediction was not the limiting problem. The measured workloads exposed too
little request-level composition freedom to justify paid timing or adapter work.

## Evidence chain

1. M0 established a deterministic native position-parallel simulator.
2. M1 found simulated critical-path improvements and identified online scheduler
   overhead as a boundary condition.
3. M2 validated real four-rank dispatch, rank-owned expert compute, and combine on
   H100x4, but classified the toy timing path as `PASS-UNPOWERED`.
4. Gate 2B classified the proxy-to-time relation as `PROXY_TIME_UNRESOLVED`.
5. M2.1 collected stock LLaDA2.0-mini denoising routes across reasoning, code, and
   general workloads and found less than 0.5% best-found request-composition headroom
   in every measured cell.

The detailed numerical result is in
[`m2_1_20260813_results.md`](m2_1_20260813_results.md). Hardware limitations and the
measurement provenance remain in [`m2_hardware_status.md`](m2_hardware_status.md) and
[`m2_gate2_20260812_results.md`](m2_gate2_20260812_results.md).

## Final milestone disposition

```text
M0    synthetic feasibility                 COMPLETE
M1    critical-path scheduler               COMPLETE
M2    H100 characterization                 COMPLETE
M2.1  native route opportunity              COMPLETE — NEGATIVE
M2.2  native-shape timing replay            DEFERRED INDEFINITELY
M3    full LLaDA2 true-EP adapter           DEFERRED INDEFINITELY
M4    native scheduler speedup              NOT PURSUED
M5    width/policy ablation                 NOT PURSUED
```

## Claim boundary

Supported:

> For the measured LLaDA2.0-mini denoising workloads, request-level route-aware batch
> composition exposed negligible MoE EP critical-path headroom relative to the
> project's hardware measurement target.

Not supported:

- request-level scheduling can never help any diffusion MoE model;
- expert placement, active-width control, or position-level scheduling are ineffective;
- a native-model latency effect was measured;
- the project produced a native-model speedup.

Placement, active-width, and position-level policies are distinct hypotheses. Pursuing
them requires a new project statement, evidence contract, and milestone sequence; they
must not be described as a successful continuation of RefineServe.

## Repository disposition

The repository, tests, compact artifacts, raw-artifact manifests, negative results,
and historical plans remain public as a reproducible research record. Conditional
upstream plans are retained for provenance but marked inactive. No result is removed
or retuned to manufacture a positive conclusion.
