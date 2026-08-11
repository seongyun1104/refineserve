# H100 hardware collectors and native K-position EP4 runner

The v1 matrix completed functional EP4 bring-up, but its K-scaling timing interval mixed
data-plane work with per-layer synchronization and validation. Do not rerun the full
matrix first. Follow `docs/m2_followup_measurement_plan.md` and begin with clock
preflight plus the timing-identifiability gate:

```bash
torchrun --standalone --nproc-per-node=4 hardware/benchmark_timing_gate_ep4.py \
  --output results/hardware/contract-followup/timing-gate \
  --active-positions 1 16 64 --warmup 3 --repetitions 10

python hardware/analyze_timing_gate_ep4.py \
  results/hardware/contract-followup/timing-gate \
  --target-mde-percent 2.0 \
  --screening-profile \
    results/hardware/contract-followup/cpu-screening/scheduler_screening_by_cell.csv
```

The companion Gate 2B constructed proxy validation runs FIFO under local-copy,
minimal-payload, and full-payload transport before applying 8.333% and 33.333%
objective-reduction doses. Its transmission estimate uses the constructed FIFO
accessibility; the balanced-route Gate 2 fraction is only a cross-check.

The timing gate is independently useful, but a paid scheduler matrix also requires a
native router-opportunity screen:

```bash
python hardware/collect_llada2_router_trace.py \
  --revision dad945cac317da394b390f82c7b40691d8a881ed \
  --output results/hardware/contract-followup/llada2-router
python hardware/screen_measured_router_trace.py \
  results/hardware/contract-followup/llada2-router \
  --trace-phase native_denoising --all-iterations \
  --output results/hardware/contract-followup/llada2-router-screen
```

The collector captures initial width ablations and a stock-semantics block-32 native
denoising trajectory with the checkpoint's block-diagonal causal mask. It is route-only
supporting calibration, not EP timing, quality, or finalization-correctness evidence.
Its native 256-expert/top-8 opportunity fraction must not be multiplied by the
controlled toy 16-expert/top-2 timing fraction.

After both gates pass, the corrected scheduler runner supports these diagnostic arms:

```bash
--schedulers fifo fifo_selection_control random_permutation critical_path \
  local_route_replay coordinated_route_replay
```

`local_route_replay` sees actual routes for one source only. `coordinated_route_replay`
is an offline best-found coordinate-descent plan with non-vacuity diagnostics. Neither
is called a global oracle or upper bound.

`K` means `active_position_count`: the positions that execute the real EP path. It
does not mean model block width or arbitrary-order freedom. See
`docs/decoding_semantics_boundary.md`.

The 2026-08-05 matrix is under
`results/hardware/h100_ep4_20260805/contract-full-v2`; its status is summarized in
`docs/m2_hardware_status.md`. Its derived useful-position throughput is uncalibrated.

## Supporting TP=1 collectors

These scripts collect the hardware evidence available from a single H100 without
claiming multi-GPU Expert Parallel calibration.

```bash
python hardware/collect_expert_kernel.py --output results/hardware/kernel
python hardware/collect_vllm_routes.py --output results/hardware/routes --backend fa3
```

`collect_expert_kernel.py` measures one active expert because the current M2
calibration model consumes a one-dimensional per-expert latency curve. It records
the fixed active-expert count in metadata and must not be mixed with grouped
multi-expert samples.

`collect_vllm_routes.py` captures actual model routes from ordinary autoregressive
decode. Its output is explicitly marked `ar_decode_observational` and is useful for
routing skew and stability calibration only. It is not a replacement for the M2
`native_position_parallel` trace gate.

TP=1 cannot produce `ep_dispatch_combine` NCCL all-to-all samples. That measurement
requires the multi-GPU EP runner above.
