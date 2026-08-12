# M2 LLaDA2 router-trace single-GPU runbook

Status: **PREPARED; PAID RUN NOT STARTED**

## Purpose

Collect stock native denoising router trajectories before spending money on Gate 3
scheduler timing. This is supporting measured-route evidence, not EP timing or a
native-model speedup result.

## Frozen model and software

```text
model: inclusionAI/LLaDA2.0-mini
revision: dad945cac317da394b390f82c7b40691d8a881ed
checkpoint dtype: BF16
transformers: 4.57.1 (checkpoint-declared compatibility version)
GPU: one CUDA GPU with at least 48 GiB; H100 80 GB preferred for short runtime
prefix length: 64, aligned to every tested block width
workloads: reasoning, code, general
segments: seeds 17,29,41 per workload; 32 requests each
native block width: 32
native generation length: 128
```

The checkpoint is approximately 32.5 GB. Confirm at least 45 GB free persistent disk
before purchase so download, cache metadata, and trace output do not fail mid-rental.

## Admission gates

1. Print offer ID, GPU, VRAM, disk, image digest, price, hard stop, and max spend.
2. Confirm the model snapshot is accessible before starting the paid clock when the
   provider supports volume preloading; otherwise make download time explicit.
3. Create an isolated environment. Do not downgrade the H100x4 Gate 2 environment.
4. Run collector `--help`, import checks, and a one-request shape smoke before the full
   five-segment collection.
5. Copy trace and hashes off-host, then destroy the instance.

## Environment

```bash
python -m venv /workspace/llada-route-venv
source /workspace/llada-route-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  'torch>=2.6' 'transformers==4.57.1' accelerate safetensors numpy pandas
```

## Collection and screening

```bash
python hardware/collect_llada2_router_trace.py \
  --model inclusionAI/LLaDA2.0-mini \
  --revision dad945cac317da394b390f82c7b40691d8a881ed \
  --workloads reasoning code general --seeds 17 29 41 \
  --generation-length 128 \
  --output results/hardware/contract-followup/llada2-router

python hardware/analyze_native_route_opportunity.py \
  results/hardware/contract-followup/llada2-router \
  --output results/hardware/contract-followup/llada2-native-opportunity
```

The collector validates that every prompt fills the aligned clean prefix rather than
treating padding as content. It uses the checkpoint's block-diagonal causal mask,
captures all 19 sparse layers, selected routing weights, exact pre-forward position
roles, and the following distinct quantities:

```text
native block width             = fixed 32
model-forward routed positions = clean prefix + prior blocks + current block
remaining masked positions     = changes with confidence
finalized positions this step  = useful progress, not model compute width
```

Confidence changes which current-block positions remain masked; it does not remove
prefix or finalized positions from the stock forward. Dense IDs, weights, and position
roles are stored in separate compressed NPZ arrays with a request/block/step manifest.
The opportunity analysis evaluates both contiguous ownership and `expert_id % 4`
round-robin ownership so router grouping is not confused with a universal scheduler
effect.

## Evidence boundary and decision

```text
near-zero realized objective reduction:
  stop scheduler work; continue native EP correctness, active-width, and placement

non-zero reproducible reduction:
  scheduling axis survives workload screening; next measure accessibility on the
  exact native shape before any scheduler speedup claim
```

Do not multiply this route reduction by the controlled toy timing fraction. The trace
does cover native denoising steps, but it still does not establish quality,
finalization correctness, EP timing, or native-model speedup.
