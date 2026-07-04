# C500 FP64 TileLang GEMM Reference

## Hardware Facts

Use these defaults for MetaX C500 unless the current machine reports otherwise:

- SM/CU: 104
- warp size: 64
- shared memory per block: 64 KiB
- L2 cache: 8 MiB
- fp64 peak: about 4 TFLOPS
- memory bandwidth peak: about 1.8 TB/s
- register file: 每个SM中512KB，L1data cache 32KB，L1C为8KB，large enough that wider accumulator tiles can win if padding stays low

Implications:

- `threads=256` means 4 C500 warps and often beats `threads=128`.
- `threads=512` is not automatically better and can increase scheduling/register pressure.
- Shared memory is tight enough that staged A/B tiles and optional C shared-store must be budgeted explicitly.
- For EDG skinny GEMMs, the bottleneck is usually compute/fragment scheduling, not raw bandwidth.

## EDG Shapes

Derivative:

```text
C[105, N] = D_merged[105, 35] @ Q[35, N]
```

Lift:

```text
surface[35, N] = lift[35, 60] @ flux[60, N]
```

Both are fp64, small `M/K`, very large `N`, and use global `B[K,N]` loaded into shared as `B_shared[block_N, block_K]` with `transpose_B=True`.

## Known Good Configs

Derivative current default:

```text
bm112_bn64_bk12_s1_t256_fullcol
```

Observed standalone derivative result on C500:

- `2.55-2.57 TFLOPS`
- `3.94-3.97 ms` at representative large `N`
- `1.34x-1.35x` over mcBLAS baseline
- about `64%` of C500 fp64 peak

Full timestep effect on `scenario1_profile_lc0p20.msh` with TileLang lift:

- baseline derivative GEMM: `7.368351 ms/step`
- TileLang derivative GEMM: `6.829143 ms/step`
- end-to-end speedup: `1.07896x`

Lift current default:

```text
bm48_bn64_bk16_s0_t256_fullcol
```

## Why the Derivative Default Wins

- `block_M=112` covers `M=105` in one CTA with only 7 padded rows.
- `block_K=12` pads `K=35` to 36 instead of padding to 48 with `block_K=16`.
- `block_N=64` gives enough N-direction work while keeping shared memory around 16 KiB.
- `threads=256` provides 4 C500 warps and is much faster than the same tile at 128 threads.
- `num_stages=1` is slightly but consistently faster than `0` for the winning tile.
- `FullCol` maps better to small-M/large-N work and avoids many `Square` fragment layout failures.

## Known Negative Directions

- `bm32_bn64_bk16_*` has high padded work inflation for `M=105,K=35`.
- `block_K=16` is usually worse than `12` for derivative because of K padding.
- `block_K=32/64` usually wastes work and shared memory for these K sizes.
- `C_local -> C_shared -> C` was slower for tested fp64 derivative configs.
- Persistent kernels were slower in the tested generic implementation.
- Many `Square` policy configs fail with `Loop layout is not injective`; prefer `FullCol` near the winner.
- Generic TileLang autotune examples are mostly TensorCore/low-precision oriented and should not be trusted directly for fp64 C500 skinny GEMM.

## Next Experiments

Prioritize these in order:

1. Re-test the current winner at true runtime `N=181140`, not only large sweep `N=1377572`.
2. Add a narrow `c500-next` sweep around `BM=96/112/128`, `BN=48/64/80/96`, `BK=4/8/12`, `threads=192/256/320/384`, `policy=fullcol`.
3. Test a padded-K runtime variant: pre-pad `D_merged` and Q to `K=36` so the kernel can remove K boundary masking.
4. Test an A-hoist persistent variant that keeps the fixed derivative matrix resident while looping multiple N tiles per CTA.

Stop an experiment if standalone speed improves but full timestep `ms_per_step` does not improve.
