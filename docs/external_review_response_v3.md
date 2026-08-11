# External Review Response v3

추가 검토 감사합니다. 신규 blocker인 mode 비대칭은 실제 구현에 존재했고,
**렌탈 전 필수 수정으로 수용했습니다.** 그 외 Q1과 Q2도 경험적 screening과
within-cell dose-response 설계로 교체했습니다. 현재 Vast.ai 활성 인스턴스는
없습니다.

## 1. Three-mode symmetry blocker

지적대로 기존 구현은 `local_copy`와 `nccl_minimal`에서 shape-matched clone을
수행했지만 `nccl_real`에서는 collective가 이를 대체했습니다. 따라서 기존
`real - minimal`은 local-copy HBM 비용을 차감하는 비대칭 차분이었습니다.

수정 후 세 mode 모두 다음 비-collective 작업을 동일하게 수행합니다.

```text
packing
shape-matched hidden/ID local copies
expert compute
shape-matched expert-output local copy
unpacking
```

`nccl_real`은 위 local copies를 dead control work로 유지한 채 full-payload
collective를 추가합니다. `nccl_minimal`은 같은 위치에 rank-pair당 1 element인
collective를 추가합니다. 따라서:

```text
minimal - local = collective launch/sync floor
real - minimal = full-payload collective increment
```

각 run은 `packing_ms`, `local_copy_memory_ms`, `unpacking_ms`를 별도 기록합니다.
metadata에는 `non_collective_work_symmetric_across_modes=true`를 남깁니다.

## 2. Compute imbalance

v1의 기존 `expert_compute_mean_ms`는 rank 평균도 rank 최댓값도 아니었습니다.
layer 전체 시간이 가장 긴 critical rank의 compute 구간을 선택한 뒤 여러 layer
record에서 평균한 값이었습니다. 따라서 “expert compute가 거의 변하지 않았다”는
문장은 rank imbalance 메커니즘을 판별할 수 없습니다.

v2 분석은 repetition마다 다음을 모두 출력합니다.

```text
expert_compute_rank_max_ms
expert_compute_rank_mean_ms
expert_compute_rank_imbalance_ms = max - mean
```

`max - mean`은 payload/floor와 구분되는 scheduler-addressable 항으로 보존합니다.
phase attribution은 unattributed <=5%와 mode/arm gap <1%p를 통과할 때만 주장합니다.

## 3. Expert-ID collective boundary

현재 측정 경로의 data plane은 여전히 세 collective입니다.

```text
hidden dispatch
int32 expert-ID dispatch
hidden combine
```

이를 primary path에서 지금 제거하지는 않았습니다. 대신 회계 artifact에
expert-major 정렬과 expert-level counts로 ID를 복원하는 명시적 `3 -> 2`
collective 시나리오를 추가했습니다. 이 행은 ID payload와 launch 하나를 모두
제거합니다. 따라서 `PASS-UNPOWERED`가 나오더라도 measured implementation과
expert-major optimized substrate의 범위를 구분할 수 있습니다.

## 4. Empirical achievability gate

고정 `imbalance=1.5`와 암묵적 achievability 1.0을 모두 제거했습니다. 기존과
동일한 deterministic route generator를 CPU에서 5 seeds로 재생하고, 각
`(K, routing, seed)`에서 다음을 계산했습니다.

```text
achievability = (FIFO max - best-found max) / (FIFO max - mean)
cell recoverable load share
  = (imbalance - 1) / imbalance × achievability
```

서로 다른 marginal p25를 곱하지 않고, **cell별 결합값을 먼저 계산한 뒤**
`(K, routing)`별 seed p25를 취합니다. 최종 gate는 다음입니다.

```text
accessible_fraction(K)
× recoverable_load_fraction_p25(K, routing)
>= 2 × MDE
```

3개 K 전체를 하나의 p25로 묶지 않았습니다. 실제 CPU 결과에서 그 방식은
모든 K를 0으로 만들었기 때문입니다. routing class별 결과는 다음처럼 달랐습니다.

```text
request_correlated: recoverable p25 = 0.20 at K=1/16/64
strong_skew:        recoverable p25 = 0.00
uniform:            K=1은 opportunity가 있으나 K=16/64는 이미 balanced
```

즉 strong skew는 불균형이 크더라도 hot-expert 고유 부하라 composition으로
줄일 수 없었습니다. 이 결과는 hard-coded 1.5보다 훨씬 직접적인 advance screen입니다.

## 5. Dose-response identification

K/routing level의 15점 회귀가 collinearity로 식별되지 않는다는 지적을
수용했습니다. primary model은 cell fixed effects를 사용합니다.

```text
paired latency change
  ~ (seed, K, routing) cell fixed effects
  + achieved predicted-load-reduction dose
```

powered cell마다 다음 arm을 준비했습니다.

```text
FIFO = dose 0
coordinated_dose_25
coordinated_dose_50
coordinated_dose_75
coordinated_route_replay = best-found dose 1
```

중간 plan은 source-wise FIFO/best mixtures와 10,000개의 seeded valid random
composition에서 목표 dose에 가장 가까운 plan을 고릅니다. target label이 아니라
실제 achieved dose를 분석에 사용합니다. duplicate plan은 허용하지 않습니다.
5-seed CPU preflight에서 request-correlated K=1/16/64 모두 최소 4개의 distinct
dose를 확보했습니다.

최종 CI는 seed-cluster bootstrap을 사용하며 seed 5개를 요구합니다. 3 seeds만
측정되면 per-seed slope와 descriptive interval로만 남기고 결론을 승격하지 않습니다.

## 6. Repetition rule and denominator

`target_mde_percent=2.0`의 분모는 `nccl_real` GPU interval입니다. accessible
fraction과 MDE가 같은 분모를 사용하도록 report metadata에 명시했습니다.

paired-difference SD도 real-NCCL mean으로 정규화합니다. alpha 0.05 양측,
power 0.8, MDE 2%에서 10 repetitions 조건은 normalized paired SD <=2.26%입니다.
이를 넘으면 analyzer가 계산한 repetition 수로 증가합니다.

## 7. 구현 세부 질문 답변

1. `nccl_real`은 수정 후 shape-matched local-copy control을 수행합니다.
2. v1 compute 값은 rank 평균/최댓값이 아니라 critical-layer-rank 값이었습니다.
3. origin slot은 전송하지 않습니다. stable assignment order와 로컬 inverse
   permutation으로 복원합니다. int32 collective는 assignment당 4 bytes인 expert
   ID만 전송합니다.
4. 한 token의 top-2가 같은 rank에 있어도 hidden을 expert assignment마다 한 번씩
   보냅니다. same-rank top-k deduplication은 현재 없습니다.
5. gather-to-send packing과 scatter-to-restore unpacking은 CUDA data-plane interval
   안에 있습니다.
6. coordinate-descent 최종 궤적만으로는 일부 request-correlated 셀에서 2개 dose만
   나왔습니다. 그래서 valid random intermediate plans를 추가했고 5-seed preflight에서
   필요한 4-dose ladder를 확인했습니다.
7. MDE와 screened recoverable share 모두 `nccl_real` GPU interval을 분모로 합니다.

## 8. 검증 상태

- Ruff 통과
- pytest **55 passed**
- 5-seed CPU screening bundle 생성 완료
- 3/2-collective 및 2x/3x/5x substrate sensitivity 회계 재생성 완료
- 실제 H100 three-mode gate는 아직 실행하지 않음
- Vast.ai 활성 인스턴스 없음

따라서 다음 유료 작업은 변함없이 three-mode gate입니다. 다만 gate는 이제 mode
대칭성, empirical achievability, per-routing screening을 모두 반영하며, powered
cell에서만 dose-response replay를 허용합니다.
