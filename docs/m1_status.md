# M1 status: online policy selected

Date: 2026-08-04

## Outcome

M1 is closed for the initial eight-layer simulator contract. The selected runtime
policy is a communication-aware, one-shot proxy scheduler:

1. Reuse the previous iteration's observed per-layer route profile.
2. Keep the oldest request as the batch anchor and admit expired requests first.
3. Score the remaining candidates once against the anchor histogram.
4. Fill the batch from that ranking without repeated full-cost replay.
5. Enable the proxy only on a communication-bound fabric profile; otherwise use the
   exact FIFO fallback.

This policy changes neither router decisions nor workload finalization semantics.

## Implemented foundation

- Eight logical attention/MoE layers and four EP ranks by default.
- Synthetic stable, unstable, skewed, and request-correlated routing.
- Expert token/weight roofline and rank-local communication cost.
- FIFO, locality, load, critical-path, joint, and information-oracle schedulers.
- Per-rank, per-batch, and per-request-iteration traces.
- Modeled scheduler overhead and measured Python selection/profile-update wall time.
- Exact global makespan oracle for small validation workloads.
- Reference layer replay and an exactly equivalent histogram estimator.
- Incremental batch histograms, prewarmed previous-route profiles, vectorized proxy
  scoring, and an adaptive FIFO safety fallback.

## Selected-policy result

Three paired seeds, eight logical layers:

| Scenario | Makespan vs FIFO | P95 vs FIFO | Selection P95 | Scheduler wall / makespan | Gate |
|---|---:|---:|---:|---:|---:|
| Compute-bound | 0.00% | 0.00% | 0.009 ms | 0.38% | PASS |
| Communication-bound | -5.19% | -5.09% | 0.312 ms | 3.44% | PASS |
| Deadline-bound | 0.00% | 0.00% | 0.009 ms | 0.74% | PASS |

The compute- and deadline-bound rows are exact FIFO fallbacks. The useful synthetic
regime remains communication-bound. These values are simulator observations, not
hardware performance claims.

The one-shot proxy can outperform the greedy full-joint policy because both are local
batch-construction heuristics and create different future ready queues. Consequently,
the greedy full-joint policy is a diagnostic baseline, not a global upper bound.

## Layer sensitivity

Across 4, 8, 16, and 32 logical layers, every seed is non-regressive. Communication-
bound mean makespan changes are -4.03%, -5.19%, -2.98%, and -3.85%, respectively.
Compute- and deadline-bound cases use FIFO and remain unchanged.

| Layers | Communication gain | Selection P95 | Scheduler wall / makespan | Online gate |
|---:|---:|---:|---:|---:|
| 4 | -4.03% | 0.265 ms | 5.98% | FAIL |
| 8 | -5.19% | 0.316 ms | 3.53% | PASS |
| 16 | -2.98% | 0.471 ms | 2.22% | PASS |
| 32 | -3.85% | 0.013 ms | 0.67% | PASS |

The four-layer miss is an overhead boundary: selection latency is below 1 ms, but the
fixed Python policy work is large relative to the shorter modeled execution. A small
profile-construction optimization reduced a follow-up measurement from 5.98% to 5.54%
without changing modeled makespan; it still does not pass the 5% gate. The project
does not hide this boundary or tune away the result. The original M1 contract is eight
layers.

The applicability statement is therefore conditional: critical-path-aware scheduling
is beneficial only when avoided communication and straggler time exceeds online
selection cost. M2 will add a calibrated gain-versus-overhead bypass rather than
forcing the scheduler onto short execution paths.

## Gates

```text
Mechanism gate:
  named regime across >= 3 paired seeds                  PASS

Correctness gate:
  histogram estimator == full layer replay              PASS
  work/finalization/routing conservation                 PASS

Quality gate:
  non-regression outside the useful regime               PASS
  communication-bound improvement across 3 seeds         PASS

Online gate for the 8-layer contract:
  selection P95 <= 1 ms                                  PASS
  scheduler wall <= 5% of modeled makespan               PASS

Sensitivity boundary:
  4-layer communication scheduler wall <= 5%             FAIL (5.98%)

Calibration gate:
  measured kernel/NCCL traces                            NOT STARTED
```

## Next milestone

M2 replaces synthetic assumptions with versioned native-route and hardware timing
traces. Its preparation work is:

1. Freeze the trace schema and unit/provenance rules.
2. Validate exact route cardinality, model dimensions, and workload key coverage.
3. Replay measured routing without silent synthetic fallback.
4. Fit expert-kernel and communication latency curves while preserving raw samples.
5. Re-run FIFO and the selected M1 policy with synthetic versus calibrated costs.
6. Report prediction error before making any hardware-level performance claim.
