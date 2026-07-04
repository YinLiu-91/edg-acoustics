---
name: optimize-c500-tilelang-gemm
description: Optimize fp64 TileLang skinny GEMM kernels for MetaX C500, especially EDG Acoustics derivative/lift shapes such as M=105,K=35,large-N and M=35,K=60,large-N. Use when tuning TileLang GEMM configs, reading C500 sweep logs, choosing block_M/block_N/block_K/thread/policy values, evaluating roofline results, or deciding whether a candidate should enter the EDG timestep runtime.
---

# Optimize C500 TileLang GEMM

## Workflow

1. Identify the exact GEMM shape, dtype, input layout, and runtime context.
2. Read `references/c500-fp64-gemm.md` before making C500-specific tuning decisions.
3. Start from known-good configs instead of generic TileLang defaults:
   - Derivative: `bm112_bn64_bk12_s1_t256_fullcol`
   - Lift: `bm48_bn64_bk16_s0_t256_fullcol`
4. Benchmark standalone kernels with real or representative `N` values before touching runtime code.
5. Promote a candidate only after it wins both standalone and full timestep benchmarks.

## Required Checks

Use these metrics together:

- logical TFLOPS and padded TFLOPS
- work inflation from tile padding
- bandwidth and arithmetic intensity
- percentage of C500 fp64 peak
- correctness against `torch.mm`
- full CUDA Graph capture/replay support
- `scenario1_benchmark.py` end-to-end `ms_per_step`

Do not accept a candidate based only on standalone TFLOPS if it regresses the full timestep or fails CUDA Graph validation.

## EDG Commands

Standalone derivative GEMM sweep:

```bash
rtk python benchmarks/mma_tilelang_v6.py \
  --shape derivative \
  --N 181140 362280 724560 1377572 \
  --profile-backend event \
  --sweep-level c500-next \
  --repeat 3
```

Full timestep A/B:

```bash
rtk python benchmarks/scenario1_benchmark.py \
  --device cuda \
  --mesh-name scenario1_profile_lc0p20.msh \
  --steps 50 \
  --cuda-graph \
  --no-record-receivers \
  --enable-tilelang-lift \
  --enable-tilelang-derivative-gemm
```

```bash
rtk python benchmarks/scenario1_benchmark.py \
  --device cuda \
  --mesh-name scenario1_profile_lc0p20.msh \
  --steps 50 \
  --cuda-graph \
  --no-record-receivers \
  --enable-tilelang-lift \
  --disable-tilelang-derivative-gemm
```

## Promotion Rule

Promote a new derivative GEMM config only if:

- standalone `N=181140` beats `bm112_bn64_bk12_s1_t256_fullcol` by at least 2%;
- full `scenario1_profile_lc0p20.msh` timestep beats current default by at least 1.5%;
- `tilelang_derivative_gemm_graph_capture_supported=1`;
- `tilelang_derivative_gemm_fallback_reason=none`;
- fp64 correctness tests pass.
