# FP64 CUDA Optimization Design and Implementation Status

## Scope and constraints

This document records the landed optimization work for the `edg-acoustics` Torch/CUDA hot path, the current code structure, and the measured performance changes.

The optimization target is the repeated time-stepping path:

```text
AcousticsSimulation.time_integration()
  -> TSI_TI.step_dt()
    -> AcousticsSimulation.RHS_operator()
```

Hard constraints:

- solver precision stays **`torch.float64`** end to end;
- CPU and CUDA execution must both remain supported;
- correctness is protected by scenario1 golden tests before accepting performance changes.

## Code design

### 1. Explicit device selection

`edg_acoustics/device_ini.py` now resolves the device from:

- `EDG_ACOUSTICS_DEVICE=auto`
- `EDG_ACOUSTICS_DEVICE=cpu`
- `EDG_ACOUSTICS_DEVICE=cuda`

This keeps benchmarking reproducible and avoids manually editing device selection for CPU/GPU comparisons.

### 2. Canonical packed state

The solver state is now stored in packed layout:

```python
Q_flat: [Np, 4 * N_tets]
Q:      [Np, 4, N_tets]
```

with public compatibility views:

- `P  = Q[:, 0, :]`
- `Vx = Q[:, 1, :]`
- `Vy = Q[:, 2, :]`
- `Vz = Q[:, 3, :]`

This removes repeated hot-path repacking as the primary state representation.

### 3. Packed RHS and packed Taylor integrator

The hot path now uses:

- `AcousticsSimulation.RHS_operator_packed()`
- `TSI_TI.step_dt_packed()`

This reduces Python dispatch overhead and avoids four independent RHS/update paths inside every Taylor substep.

### 4. Batched derivative and lift GEMMs

The original derivative work was collapsed into three larger GEMMs:

- `Dr @ Q_flat`
- `Ds @ Q_flat`
- `Dt @ Q_flat`

On CUDA this is further packed into one batched DGEMM:

```python
torch.bmm(stack([Dr, Ds, Dt]), Q_flat.expand(3, -1, -1))
```

Set `EDG_ACOUSTICS_BATCHED_DERIVATIVES=0` to force the original three-GEMM fallback.

For larger CUDA meshes, the derivative path now uses a merged cuBLAS DGEMM by default:

```python
D_merged = torch.cat((Dr, Ds, Dt), dim=0)  # [3 * Np, Np]
D_merged @ Q_flat                          # [3 * Np, 4 * N_tets]
```

This keeps the operation in cuBLAS fp64 while using a single taller GEMM shape. It is enabled automatically for CUDA meshes with at least `10_000` tetrahedra. Override with `EDG_ACOUSTICS_MERGED_DERIVATIVES=0` or `1`.

and face lift work was packed into one GEMM:

- `lift @ Flux_flat`

This improves cuBLAS utilization and reduces kernel launches.

### 5. Packed flux path

Interior fluxes and boundary overrides now use the packed face layout directly:

```python
Flux_flat: [4 * Nfp, 4 * N_tets]
Flux:      [4 * Nfp, 4, N_tets]
```

Boundary face override indices are precomputed once and reused during time stepping.

### 6. Fused volume RHS assembly

Volume metric scaling is now written directly into packed RHS buffers using pre-scaled metric tensors instead of assembling multiple temporary outputs first and scaling later.

On CUDA this path is additionally fused with a Triton kernel by default. The kernel computes all four packed RHS components in one pass over `(Np, N_tets)`, reducing graph-internal elementwise launches and memory traffic. When surface lift is available, `volume_surface_rhs_kernel` also adds the lifted surface contribution while writing RHS, removing four per-RHS surface-add kernels. Set `EDG_ACOUSTICS_TRITON_VOLUME_RHS=0` or `EDG_ACOUSTICS_TRITON_VOLUME_SURFACE_RHS=0` to force the older fallback paths.

### 7. Fused interior flux assembly

On CUDA, interior face gather, jump calculation, and upwind flux assembly are fused into a Triton kernel by default. The fallback remains the packed PyTorch path.

The fused kernel writes the same packed face layout:

```python
Flux_flat: [4 * Nfp, 4 * N_tets]
```

Boundary ADE/impedance overrides still run after interior flux assembly. When all CUDA boundary writers are enabled, the interior and boundary flux kernels write `Fscale`-scaled flux directly and skip the global `flux_view.mul_(Fscale)` pass. Set `EDG_ACOUSTICS_TRITON_INTERIOR_FLUX=0` or `EDG_ACOUSTICS_SCALED_FLUX_KERNELS=0` to force fallback behavior.

The default CUDA interior flux kernel now computes the homogeneous acoustic upwind flux from `nx/ny/nz` and `Fscale` directly:

```text
dvn = nx * dVx + ny * dVy + nz * dVz
flux_v = 0.5 * n * (dP / rho0 - c0 * dvn)
flux_p = -0.5 * c0 * dP + 0.5 * c0^2 * rho0 * dvn
```

This replaces loads from the expanded coefficient tensors (`cn*`, `n*rho`, `csn*rho`) with fewer geometry loads and extra fp64 arithmetic. It is enabled by default on CUDA; set `EDG_ACOUSTICS_COMPACT_FLUX_COEFFICIENTS=0` to restore the coefficient-load path.

### 8. Fused RI-only boundary flux

The largest scenario1 boundary group is RI-only and does not carry RP/CP ADE state. On CUDA that group is fused into one Triton kernel that gathers boundary state, computes `vn/ou/in`, and writes packed boundary fluxes. RP-only and RP+CP impedance groups now have separate Triton kernels that also update recursive ADE state (`phi`, `kexi1`, `kexi2`) in place. Set `EDG_ACOUSTICS_TRITON_BOUNDARY_RI=0` or `EDG_ACOUSTICS_TRITON_BOUNDARY_ADE=0` to force the fallback.

### 9. Benchmark-gated experimental kernels

Several deeper kernels are implemented and benchmark-gated because they do not all beat the current cuBLAS/PyTorch path on V100:

- fused state accumulation fuses RHS write with `Q_flat += coefficient * RHS_Q`;
- `EDG_ACOUSTICS_PAIRED_INTERIOR_FLUX=1` computes each physical interior face-node pair once and writes both element-side fluxes;
- `EDG_ACOUSTICS_TRITON_DERIVATIVE_VOLUME=1` combines derivative application and volume RHS;
- `EDG_ACOUSTICS_TRITON_LIFT_SURFACE=1` replaces `lift @ Flux_flat` with a direct Triton lift kernel.

Fused state accumulation is now enabled automatically for CUDA meshes with at least `10_000` tetrahedra because the generated fine-geometry profile mesh benefits from it. Override with `EDG_ACOUSTICS_FUSED_STATE_ACCUMULATION=0` or `1`. The paired-face, derivative-volume, and lift-surface Triton kernels remain default-off because they were slower on the tested V100/profile-mesh shape.

### 10. CUDA Graph replay and optional chunking

For CUDA runs, `time_integration(..., use_cuda_graph=True)` captures and replays one full packed Taylor step:

1. snapshot current packed state and boundary ADE state;
2. warm up one packed step;
3. capture one full `step_dt_packed()` into a `torch.cuda.CUDAGraph`;
4. restore the original state;
5. replay the graph at every time step.

This is the most important optimization for scenario1-sized fp64 workloads because it removes the CPU launch overhead of many small CUDA kernels.

`time_integration(..., cuda_graph_chunk_steps=K)` can capture `K` consecutive fixed steps and receiver sampling into a graph chunk. This reduces graph launch count, but after Triton fusion the benefit is small for scenario1 because the remaining runtime is mostly real graph-internal FP64 work.

## Validation strategy

The current optimization stack is protected by:

- `tests/test_scenario1_golden.py`
  - RHS fp64 golden
  - short time-integration fp64 golden
  - CUDA Graph fp64 golden
- `tests/test_scenario1.py::test_scenario1_simulation`

Useful validation commands:

```bash
python -m py_compile edg_acoustics/*.py benchmarks/scenario1_benchmark.py tests/scenario1_utils.py tests/test_scenario1_golden.py
pytest tests/test_scenario1_golden.py tests/test_scenario1.py::test_scenario1_simulation -q
```

## Profiling workflow: how hotspots were found

The optimization work was driven by fixed-step benchmarking plus `torch.profiler`, not by changing code blindly.

### 1. Measure normal runtime first

Always start with the non-profiled benchmark because profiler overhead distorts absolute timings:

```bash
python benchmarks/scenario1_benchmark.py --device cpu --steps 1000
python benchmarks/scenario1_benchmark.py --device cuda --steps 1000
python benchmarks/scenario1_benchmark.py --device cuda --steps 1000 --cuda-graph
python benchmarks/scenario1_benchmark.py --device cuda --steps 1000 --cuda-graph --no-record-receivers
```

This answers:

- is CUDA actually faster than CPU for the current workload?
- does CUDA Graph still matter?
- are receiver writes masking solver improvements?

### 2. Use profiler only to identify *where* time goes

Use the benchmark wrapper so the profiler runs the same reproducible scenario:

```bash
python benchmarks/scenario1_benchmark.py --device cuda --steps 300 --profile
python benchmarks/scenario1_benchmark.py --device cuda --steps 1000 --cuda-graph --profile
python benchmarks/scenario1_benchmark.py --device cuda --steps 1000 --cuda-graph --profile --profile-row-limit 30
```

The benchmark also supports real-case `time_integration(total_time=...)` timing without postprocessing or result-file writes:

```bash
python benchmarks/scenario1_benchmark.py --device cpu --cpu-threads 8 --mesh-name scenario1_coarser.msh --real-case-total-time 0.005
python benchmarks/scenario1_benchmark.py --device cuda --mesh-name scenario1_coarser.msh --real-case-total-time 0.005 --cuda-graph
python benchmarks/scenario1_benchmark.py --device cpu --cpu-threads 8 --mesh-name scenario1_profile_lc0p20.msh --real-case-total-time 0.001
python benchmarks/scenario1_benchmark.py --device cuda --mesh-name scenario1_profile_lc0p20.msh --real-case-total-time 0.001 --cuda-graph
```

The benchmark also supports A/B isolation of each optimization path:

```bash
python benchmarks/scenario1_benchmark.py --device cuda --steps 1000 --cuda-graph --disable-triton-volume-rhs
python benchmarks/scenario1_benchmark.py --device cuda --steps 1000 --cuda-graph --disable-triton-boundary-ade
python benchmarks/scenario1_benchmark.py --device cuda --steps 1000 --cuda-graph --disable-scaled-flux-kernels
python benchmarks/scenario1_benchmark.py --device cuda --steps 1000 --cuda-graph --enable-fused-state-accumulation
```

For fine-geometry profiling, select the generated profile mesh explicitly:

```bash
python benchmarks/scenario1_benchmark.py --device cuda --mesh-name scenario1_profile_lc0p20.msh --steps 50 --cuda-graph
python benchmarks/scenario1_benchmark.py --device cuda --mesh-name scenario1_profile_lc0p20.msh --steps 10 --cuda-graph --profile --profile-row-limit 30
```

### 3. How to read the profiler output

For this repository, the most useful signals are:

- repeated small cuBLAS kernels like `volta_dgemm_64x64_nn`;
- repeated `vectorized_elementwise_kernel`, `elementwise_kernel`, `reduce_kernel`;
- indexing kernels such as `indexSelect`, `index_elementwise_kernel`, `scatter_gather`;
- custom Triton kernels (`interior_flux_kernel`, `boundary_rp_cp_flux_kernel`, `volume_surface_rhs_kernel`).

Interpretation rules used in this repository:

- **benchmark decides whether a change stays**;
- **profiler decides where to investigate next**;
- profiler absolute `ms/step` is ignored because profiler overhead is large;
- if a fused kernel is correct but slower in the normal benchmark, it stays off by default behind an env flag.

## Current hotspot analysis

After the current fused path landed, the remaining cost profile changed substantially:

### GitHub research references used for the follow-up pass

The lc0p20 pass also checked open-source GPU DG/high-order solvers for comparable implementation patterns:

| Reference | Files / concept | Design takeaway for this repository |
| --- | --- | --- |
| `paranumal/libparanumal` | `solvers/acoustics/okl/acousticsSurfaceTet3D.okl`, `acousticsVolumeTet3D.okl` | Acoustic DG kernels compute flux from normals and apply local `LIFT`; volume kernels store affine metrics compactly. This motivated recomputing acoustic flux coefficients from `nx/ny/nz` instead of loading many expanded coefficient tensors. |
| `tcew/MIDG2` | `src/MaxwellsSurfaceKernel3D.occa`, `src/MaxwellsOKL3d.cpp` | Surface kernels use compact face metadata (`Fscale`, `nx`, `ny`, `nz`) and local lift-style accumulation, reinforcing the compact face-data direction. |
| `mfem/mfem` | `fem/restriction.hpp`, `fem/restriction.cpp` `L2FaceRestriction` | MFEM represents DG faces as a double-valued physical-face restriction with two traces and face-node permutations. This informed the experimental paired interior-face table. |
| `exapde/Exasim` | `backend/Common/pblas.h` | Pure dense high-order transforms are kept on BLAS/cuBLAS. This motivated the accepted merged-derivative GEMM instead of replacing derivatives with scalar Triton loops. |
| `CEED/NekRS` | `kernels/elliptic/ellipticPartialAxCoeffHex3D.okl` | Custom kernels are most useful when fusion removes enough global-memory traffic; otherwise dense operator application should stay with optimized libraries. |

The full libParanumal/MIDG2-style surface-flux plus `LIFT` fusion remains a strong future direction, but a naive Triton replacement of lift was already measured slower on this V100. This pass therefore landed lower-risk pieces that preserve cuBLAS for dense work and reduce memory traffic in the dominant flux kernel.

### Before deep fusion

The hot path was dominated by:

- many tiny graph-internal elementwise kernels;
- boundary `index_select`, `sum`, and `index_put_`;
- separate `flux_view.mul_(Fscale)`;
- separate surface add after volume RHS;
- repeated small DGEMMs.

That led to the following accepted optimizations:

| Observed hotspot | Accepted optimization |
| --- | --- |
| boundary gather/sum/index_put | Triton RI / RP / RP+CP boundary kernels |
| separate global `flux_view.mul_(Fscale)` | write scaled flux directly in fused CUDA kernels |
| separate volume RHS + surface add | `volume_surface_rhs_kernel` |
| many derivative GEMM launches | batched derivative DGEMM |

### After deep fusion

The current CUDA Graph profiler is now dominated by:

- `volta_dgemm_64x64_nn` from derivative/lift GEMMs;
- `interior_flux_kernel`;
- `boundary_rp_cp_flux_kernel`, `boundary_rp_flux_kernel`, `boundary_ri_flux_kernel`;
- `volume_surface_rhs_kernel`;
- a much smaller number of remaining elementwise/reduce kernels;
- receiver sampling kernels outside the graph.

That is why the next-tier experiments were treated differently:

- `EDG_ACOUSTICS_FUSED_STATE_ACCUMULATION=1`
- `EDG_ACOUSTICS_TRITON_DERIVATIVE_VOLUME=1`
- `EDG_ACOUSTICS_TRITON_LIFT_SURFACE=1`

On the original 305-tet scenario1 shape, all three were **correct** but did not consistently beat the default path. On the 45,285-tet profile mesh, fused state accumulation was consistently faster and is now size-gated on by default. The derivative-volume and lift-surface Triton paths remained slower and stay opt-in.

### Fine-geometry profile mesh pass

The full `scenario1_fine.msh` has about 659k tetrahedra and currently OOMs during CUDA initialization on the tested V100. To profile a much larger but runnable fine-geometry case, a deterministic intermediate mesh is generated from `scenario1_fine.geo`:

```bash
python benchmarks/generate_scenario1_mesh.py --lc 0.20 --output examples/scenario1/scenario1_profile_lc0p20.msh
```

Generated mesh:

| Mesh | Source | Vertices | Tetrahedra | Relative to coarser |
| --- | --- | ---: | ---: | ---: |
| `scenario1_coarser.msh` | existing coarse case | 120 | 305 | 1.0x |
| `scenario1_profile_lc0p20.msh` | `scenario1_fine.geo`, `lc=0.20` | 9,391 | 45,285 | 148x |
| `scenario1_fine.msh` | `scenario1_fine.geo`, `lc=0.08` | 117,756 | 659,212 | 2161x |

Profile-mesh hotspot before the final index optimization:

| Profiler row | CUDA time share |
| --- | ---: |
| `interior_flux_kernel` | 41.96% |
| `volume_surface_rhs_kernel` | 21.08% |
| derivative/lift DGEMMs | 25.87% |
| remaining elementwise and boundary kernels | 11.09% |

The accepted profile-mesh optimization avoids redundant packed-index reads in `interior_flux_kernel`: the left face-node index is derived from `Fmask`, and the right-side pressure base index is loaded once then reused for the three velocity components. This reduced the profiled `interior_flux_kernel` share from 41.96% to 38.79% and improved fixed-step runtime from about `9.38 ms/step` to `8.75 ms/step`.

Profile-mesh A/B results:

| Variant | ms/step | Decision |
| --- | ---: | --- |
| Default before profile-mesh pass | 9.38 | baseline for this pass |
| Optimized interior indexing | 8.96 | accepted |
| Optimized interior + fused state auto gate | 8.75 | accepted for `N_tets >= 10_000` |
| Disable fused state after auto gate | 8.92 | keep fused auto gate |
| `--enable-triton-derivative-volume` | 13.60 | rejected |
| `--enable-triton-lift-surface` | 19.72 | rejected |
| `--disable-scaled-flux-kernels` | 10.18 | rejected |
| `--disable-triton-volume-surface-rhs` | 10.00 | rejected |
| `--disable-triton-interior-flux` | 15.82 | rejected |

### lc0p20 follow-up pass

The second lc0p20 pass started from this measured baseline:

| Run | Steps | ms/step | Notes |
| --- | ---: | ---: | --- |
| Previous lc0p20 default | 100 | 8.69 | compact flux off, merged derivatives off |
| Compact flux coefficients | 100 | 7.85 | accepted |
| Compact flux + merged derivatives | 100 | 7.33 | accepted for `N_tets >= 10_000` |
| Compact flux only, merged disabled | 100 | 7.84 | confirms merged-derivative gain |
| Paired interior physical-face kernel | 100 | 12.72 | correct but rejected/default-off |
| Legacy coefficient-load path | 100 | 8.67 | kept as fallback |

Current lc0p20 profiler with accepted defaults:

| Profiler row | CUDA share | Per RHS call | Decision |
| --- | ---: | ---: | --- |
| `compact_interior_flux_kernel` | 36.34% | 0.920 ms | improved from original `interior_flux_kernel` 1.193 ms/RHS |
| `volume_surface_rhs_kernel` | 31.21% | 0.790 ms | still a major hotspot |
| merged derivative/lift DGEMMs | 25.41% | 0.322 ms per visible DGEMM row | merged derivative keeps cuBLAS and reduces fixed-step time |
| boundary Triton kernels | 4.37% | small | no longer first-order |

Rejected/default-off details:

- `EDG_ACOUSTICS_PAIRED_INTERIOR_FLUX=1` is correct after matching reciprocal `(vmapM, vmapP)` / `(vmapP, vmapM)` keys, but it is slower on V100 because the compact pair list introduces indirect loads and the kernel becomes less coalesced than the dense element-side pass.
- Affine face geometry is detected (`affine_face_geometry=1`), but the accepted compact kernel still uses per-face-node normals/Fscale to preserve strict fp64 equivalence. Per-face compressed geometry remains possible behind a future tolerance-gated path.
- Affine volume metrics vary only at about `1e-10` from node to node on lc0p20, but multiplying by acoustic pressure coefficients can amplify that enough to risk strict fp64 golden tolerances. The current `volume_surface_rhs_kernel` therefore remains the correctness-preserving default.

## How to use CUDA Graph in `examples/scenario1/main.py`

The scenario now enables CUDA Graph directly in:

```python
sim.time_integration(
    total_time=impulse_length,
    delta_step=save_every_Nstep,
    save_step=temporary_save_Nstep,
    format="mat",
    use_cuda_graph=True,
)
```

Run it with CUDA explicitly:

```bash
cd examples/scenario1
EDG_ACOUSTICS_DEVICE=cuda python main.py
```

If the process is not running on CUDA, the graph path is not used.

## Performance summary

### A. Historical coarser-mesh `examples/scenario1/main.py` progression

These rows record the original coarser-mesh optimization history. The runs did **not** all use the same total simulation time (`2.0s` vs `0.1s`), so direct wall-clock comparison is not apples-to-apples. The comparable metric is **time per step**.

| Stage | Workload | Steps | Observed time | ms/step | Relative to initial |
| --- | --- | ---: | ---: | ---: | ---: |
| Initial baseline | `main.py`, `total_time=2.0` | 65693 | 810.60 s wall | 12.34 | 1.00x |
| Earlier optimized, pre-CUDA-Graph | `main.py`, `total_time=0.1` | 3284 | 27.61 s wall | 8.41 | 1.47x faster |
| Current optimized + CUDA Graph | `main.py`, `total_time=0.1` | 3284 | 9.74 s wall | 2.97 | 4.16x faster |
| Current CUDA Graph + fused kernels | `main.py`, `total_time=0.1` | 3284 | 1.12 s solver loop | 0.34 | 36.0x faster |

Current solver-loop timing printed by `time_integration()` for the CUDA Graph run is also much lower:

| Stage | Steps | Solver time | Solver ms/step | Relative to previous optimized |
| --- | ---: | ---: | ---: | ---: |
| Earlier optimized, pre-CUDA-Graph | 3284 | 15.68 s | 4.77 | 1.00x |
| Current optimized + CUDA Graph | 3284 | 5.02 s | 1.53 | 3.12x faster |
| Current CUDA Graph + fused kernels | 3284 | 1.12 s | 0.34 | 14.0x faster |

### B. CPU / GPU hot-path benchmark comparison

All rows below use the coarser mesh and the same fixed-step benchmark:

```bash
python benchmarks/scenario1_benchmark.py --steps 1000 ...
```

| Mode | Device | CUDA Graph | 1000-step time | ms/step | Relative to CPU |
| --- | --- | --- | ---: | ---: | ---: |
| Fixed-step benchmark | CPU | No | 4552.87 ms | 4.55 | 1.00x |
| Fixed-step benchmark | CUDA | No | 913.46 ms | 0.91 | 4.99x faster |
| Fixed-step benchmark | CUDA | Yes | 222.75 ms | 0.22 | 20.44x faster |
| Fixed-step benchmark, no receiver sampling | CUDA | Yes | 211.79 ms | 0.21 | 21.50x faster |
| Fixed-step benchmark, chunk `K=32` | CUDA | Yes | 268.48 ms / 1024 steps | 0.26 | 17.37x faster |

Interpretation:

- on this small fp64 workload, packed CUDA eager is now faster than CPU after Triton fusion, but it is still limited by repeated launch overhead;
- **CUDA Graph is what makes GPU much faster** by removing that repeated launch overhead;
- Triton volume/surface, flux, RI/RP/CP boundary fusion, scaled flux writes, and batched derivatives reduce real graph-internal GPU work, so CUDA eager and CUDA Graph both improve;
- the current landed design keeps the packed PyTorch fallback and uses CUDA Graph plus CUDA-only fused kernels as the practical speedup path.

### C. Fine-geometry profile mesh performance

All rows use `scenario1_profile_lc0p20.msh` (`45,285` tetrahedra, fp64, V100). This mesh is large enough that real graph-internal work dominates; CUDA Graph removes little beyond launch overhead, but GPU is still much faster than CPU.

| Mode | Device | Steps | Observed time | ms/step | Relative to CPU |
| --- | --- | ---: | ---: | ---: | ---: |
| Fixed-step benchmark | CPU, 8 threads | 5 | 3298.15 ms | 659.63 | 1.00x |
| Fixed-step benchmark + CUDA Graph, first lc0p20 pass | CUDA | 50 | 437.64 ms | 8.75 | 75.36x faster |
| Fixed-step benchmark + CUDA Graph, current | CUDA | 100 | 732.77 ms | 7.33 | 90.02x faster |
| `main.py`, `impulse_length=0.001`, first lc0p20 pass | CUDA + CUDA Graph | 205 | 1.77 s solver loop | 8.62 | 76.52x faster by step rate |
| `main.py`, `impulse_length=0.001`, current | CUDA + CUDA Graph | 205 | 1.49 s solver loop | 7.26 | 90.86x faster by step rate |

The current `examples/scenario1/main.py` uses this profile mesh instead of the full `scenario1_fine.msh`, because the full fine mesh currently OOMs during CUDA initialization. The shorter `impulse_length=0.001` keeps iteration fast while preserving a representative 205-step fine-geometry run.

### D. Current real-case CPU / GPU time-integration comparison

The table below uses the new benchmark real-case mode, which times only `sim.time_integration(...)` with receiver sampling enabled and without postprocessing or result-file writes. CPU runs use 8 threads; GPU runs use the current CUDA path with CUDA Graph enabled.

| Mesh | Total time | Device | Steps | Solver loop time | Solver ms/step | Relative to CPU |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `scenario1_coarser.msh` | `0.005 s` | CPU, 8 threads | 164 | `0.856 s` | `5.22` | `1.00x` |
| `scenario1_coarser.msh` | `0.005 s` | CUDA + CUDA Graph | 164 | `0.0364 s` | `0.222` | `23.50x faster` |
| `scenario1_profile_lc0p20.msh` | `0.001 s` | CPU, 8 threads | 205 | `142.46 s` | `694.90` | `1.00x` |
| `scenario1_profile_lc0p20.msh` | `0.001 s` | CUDA + CUDA Graph | 205 | `1.50 s` | `7.33` | `94.81x faster` |

Interpretation:

- on the small coarser mesh, GPU is already much faster than CPU for the full real case, but the speedup is lower than the lc0p20 mesh because fixed overheads still matter more;
- on lc0p20, the solver is dominated by real fp64 element/face work and the optimized CUDA path is now about **94.8x** faster than the 8-thread CPU run for the same physical simulation window;
- the real-case lc0p20 solver-loop timing (`1.50 s`) matches the previously observed `main.py` timing closely, confirming that the current CUDA Graph path is the practical speedup route for the actual scenario run.

## Files most relevant to the final design

- `edg_acoustics/device_ini.py`
- `edg_acoustics/acoustics_simulation.py`
- `edg_acoustics/time_integration.py`
- `benchmarks/generate_scenario1_mesh.py`
- `benchmarks/scenario1_benchmark.py`
- `tests/test_scenario1_golden.py`
- `tests/test_scenario1_profile_mesh_optimizations.py`
- `tests/scenario1_utils.py`
- `examples/scenario1/main.py`
- `examples/scenario1/scenario1_profile_lc0p20.msh`

## Deferred work

Remaining deep-optimization candidates are correctness-riskier and should be benchmark-gated:

- custom derivative or lift kernels beyond cuBLAS for different `Np/N_tets` shapes or newer GPUs;
- larger graph chunks or full-loop capture only if graph launch count becomes significant again.
