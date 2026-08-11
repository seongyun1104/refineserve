# Gate 2/2B internal double-check

Date: **2026-08-12**

## Decision

```text
Gate 2 timing characterization       GO after the fixes below
Gate 2B proxy validation             GO after the fixes below
Gate 3 scheduler matrix              NO-GO
active GPU rental                    none
```

No performance claim is authorized by this audit. It verifies that the next paid run
will measure the preregistered mechanism rather than a harness artifact.

## Defects found before rental

### 1. Gate 2B receive splits assumed identical source plans

The 8.333% dose changes rank 0's composition while ranks 1--3 retain FIFO. The old
implementation constructed each destination's receive splits by repeating the local
source's send count. That is valid only when every source has the same plan. At the
low dose, ranks 1--3 could therefore pass incorrect `output_split_sizes` to
`all_to_all_single`.

The benchmark now derives this immutable contract before timing:

```text
[arm][source rank][batch][destination rank] -> assignment count
```

Each receiving rank uses one count from every source. A pure CPU unit test proves that
the low-dose receive vector is source-specific and would fail under the old repeated
count assumption.

### 2. Dynamic count extraction occurred inside the measured path

The previous Gate 2B implementation converted a CUDA `bincount` to a Python list in
the layer loop. That host-visible conversion synchronizes the device and contaminates
the interval. Send/receive splits are now computed and validated once before timing;
there is no `.tolist()` or dynamic count exchange in the CUDA interval.

### 3. Local/minimal controls were not shape-correct for an imbalanced receive rank

For an imbalanced FIFO composition, a rank's receive token count need not equal its
send token count. Gate 2B now preallocates rank/batch-specific receive, combine,
restoration, expert-weight, and expert-ID buffers from the global split contract. The
local-copy and minimal-payload controls execute the same receive shape and local expert
ID distribution as the full FIFO path.

### 4. Gate 2B imported accessibility from a different route shape

The original analyzer divided by Gate 2's balanced-route accessibility. A constructed
imbalanced FIFO route can have a different fixed/payload ratio, so this could bias the
transmission coefficient. Gate 2B now measures its own FIFO controls:

```text
local copy -> minimal-payload NCCL -> full-payload NCCL
```

The constructed accessibility is the transmission denominator. Gate 2's value remains
only a cross-check. If bootstrap lower confidence bounds do not identify the launch
floor, accessible payload, and total premium, the proxy status is unresolved.

### 5. Host setup remained between the outer CUDA events

CUDA event creation, scratch allocation, and expert-weight allocation could leave GPU
idle gaps after the outer start event. Both Gate 2 and Gate 2B now preallocate stage
events and reusable scratch buffers before recording the measured run. Packing,
shape-matched control copies, fused expert work, transport, and restoration remain
inside the interval by design.

### 6. Timing-gate distinguishability used only median sign

The former analyzer treated a positive median NCCL premium as identified. It now
requires positive bootstrap-median lower confidence bounds for both the launch floor
and total NCCL premium at every K. Unattributed time is tested by absolute magnitude so
a large negative accounting residual cannot silently pass.

## Frozen Gate 2B arms

```text
fifo_local_copy_control       FIFO composition, local-copy transport
fifo_nccl_minimal_control     FIFO composition, three minimal collectives/layer
fifo_constructed              FIFO composition, three full collectives/layer
dose_083_constructed          full collectives, objective reduction exactly 1/12
balanced_constructed          full collectives, objective reduction exactly 1/3
```

All arms preserve request count, top-2 assignment count, expert-ID element count, and
global destination assignment totals. Only their per-batch destination alignment and
the explicitly declared transport control differ.

## Remaining hardware-only checks

These cannot be closed on the local non-H100 machine:

- installed vLLM `fused_experts` signature and selected kernel/config hash;
- exact four-rank NCCL lowering from `NCCL_DEBUG=INFO` and profiler evidence;
- H100 clock-lock permission and achieved clock/thermal stability;
- real local/minimal/full-payload confidence intervals;
- profiler confirmation that no unexpected host gap dominates the corrected interval.

Any failure remains fail-closed: preserve artifacts, stop performance work, and return
the instance without running Gate 3.
