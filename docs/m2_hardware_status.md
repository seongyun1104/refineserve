# M2 Controlled Hardware Status

Date: **2026-08-12**

## Gate 2 follow-up

The paid three-mode identifiability run is complete and the instance has been
destroyed. The result is `PASS-UNPOWERED`; the preregistered Gate 2B result is
`PROXY_TIME_UNRESOLVED`. Gate 3 was not run. Clock locking was denied by the provider,
so this run is hardware characterization rather than a percent-level performance
claim. See `m2_gate2_20260812_results.md` for the complete interpretation and artifact
location.

## Outcome

The controlled native K-position EP4 matrix is complete on one four-H100 SXM host.
This is a real hardware result with synthetic routing and synthetic model semantics;
it is not yet a native LLaDA2.0-mini result.

```text
8 MoE layers
16 experts, top-2
EP=4, TP=1, PP=1
hidden=2048, intermediate=8192, BF16
K=1,2,4,8,16,32,64
7 routing modes, 6 schedulers, 3 seeds
```

- 2,646 measured runs after warmup
- 42,336 rank-critical aggregated layer records
- all finite, route-range, assignment-count, and collective checks pass
- four rank-local CPU/CUDA Chrome traces retained for `K=32`

## Primary observed result

The current online critical-path scheduler is not beneficial in this hardware regime.

```text
Three-seed mean P50 end-to-end change vs FIFO: +3.506%
Three-seed-mean cells improving:               0 / 49
Mean GPU-only path change vs FIFO:             +1.175%
Mean scheduler fraction:                       2.277%
```

The result is intentionally preserved as negative evidence. The online selection cost
explains most, but not all, of the slowdown. Even the rank-local actual-route replay
averages +3.112% end-to-end and +0.924% GPU-only versus FIFO, so the current small
candidate pool and batch-composition freedom do not recover enough EP critical-path
time. It sees actual routes for one source rank only and is not an oracle or a
coordinated four-rank bound.

## K sweep under FIFO

| K | P50 latency mean | Useful positions/s | Assignments/active expert | Active experts | Communication fraction |
|---:|---:|---:|---:|---:|---:|
| 1 | 31.211 ms | 1,026 | 2.68 | 12.36 | 55.30% |
| 2 | 31.096 ms | 2,060 | 4.36 | 14.85 | 54.38% |
| 4 | 31.335 ms | 4,087 | 8.23 | 15.61 | 54.42% |
| 8 | 31.295 ms | 8,184 | 16.08 | 15.93 | 53.73% |
| 16 | 31.342 ms | 16,349 | 32.05 | 15.98 | 52.32% |
| 32 | 31.273 ms | 32,766 | 64.00 | 16.00 | 51.75% |
| 64 | 31.230 ms | 65,634 | 128.00 | 16.00 | 48.15% |

The latency, active-expert count, and communication fraction are measured. Useful
positions/s is derived algebraically from `32 * K / latency`, and tokens/active expert
is derived from assignment count and active-expert count; neither is independent
evidence.

The near-flat latency must not currently be interpreted as K-scaling speedup. The
measured interval contains two request batches times eight layers, and the v1 loop
performed per-layer count exchange, CUDA synchronization, correctness checks, and
metric extraction inside that interval. Those fixed control and validation costs can
hide the variable K-dependent data path. The decreasing absolute communication time at
larger K is therefore a diagnostic sign that the present timing cannot identify the
payload-scaling effect; it is not evidence that moving more bytes is intrinsically
faster.

What remains supported is narrower:

- the real EP=4 dispatch, rank-owned expert compute, and combine path passed functional
  checks across K=1..64;
- all 16 experts become active by K=32 in this E=16 prototype;
- the v1 K-scaling magnitude and synthetic useful-position throughput are uncalibrated;
- no native-model quality-safe speedup has been established.

## Limitations

- Scheduler arms were run in a fixed order; small GPU-only differences may include
  order drift.
- Each source rank schedules independently, so the current policy cannot optimize the
  combined receive load produced by all four sources.
- Three measured repetitions per seed make P95/P99 descriptive rather than strong tail
  estimates.
- Routing is synthetic, not LLaDA2.0-mini routing.
- The path isolates router/EP dispatch/expert/EP combine and is not a full native model.
- vLLM used its default H100 fused-MoE configuration because no exact E=4, N=8192 tuning
  file was present.
- The v1 timed interval included measurement and validation work that must be moved out
  of the isolated GPU data-path interval before another scaling claim is made.

## Next gate

1. Collect a stock LLaDA2.0-mini initial masked-block router trace and apply the
   summed batch/layer critical-load screen under the planned EP=4 placement.
2. Measure native-shape accessibility separately; do not reuse the toy denominator.
3. Run scheduler timing only if both the native-shape timing gate and native workload
   opportunity gate pass. Do not combine toy timing and native route fractions as if
   their shapes were identical.
4. Bring up true EP correctness with `inclusionAI/LLaDA-MoE-7B-A1B-Instruct`.
5. Replace the LLaDA2.0-mini stock tensor-parallel MoE path with rank-owned EP=4.
6. Hold model router, remasking, order policy, and finalization fixed.
7. Record quality, useful progress, and wall-clock together.

The post-review implementation audit found and fixed a source-specific receive-split
error in the constructed low-dose Gate 2B arm before rental. Gate 2B now measures its
own FIFO local/minimal/real accessibility, uses precomputed global send/receive splits,
and requires confidence-bound transport identification before reporting transmission.
See `docs/gate2_internal_double_check.md`. No new paid run has started.

The primary native model is `inclusionAI/LLaDA2.0-mini`; DiffusionGemma is secondary
cross-model validation.
