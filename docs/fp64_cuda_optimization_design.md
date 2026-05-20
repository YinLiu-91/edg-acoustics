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

### 8. Fused RI-only boundary flux

The largest scenario1 boundary group is RI-only and does not carry RP/CP ADE state. On CUDA that group is fused into one Triton kernel that gathers boundary state, computes `vn/ou/in`, and writes packed boundary fluxes. RP-only and RP+CP impedance groups now have separate Triton kernels that also update recursive ADE state (`phi`, `kexi1`, `kexi2`) in place. Set `EDG_ACOUSTICS_TRITON_BOUNDARY_RI=0` or `EDG_ACOUSTICS_TRITON_BOUNDARY_ADE=0` to force the fallback.

### 9. Benchmark-gated experimental kernels

Several deeper kernels are implemented but default off because they did not consistently beat the current cuBLAS/PyTorch path on V100:

- `EDG_ACOUSTICS_FUSED_STATE_ACCUMULATION=1` fuses RHS write with `Q_flat += coefficient * RHS_Q`;
- `EDG_ACOUSTICS_TRITON_DERIVATIVE_VOLUME=1` combines derivative application and volume RHS;
- `EDG_ACOUSTICS_TRITON_LIFT_SURFACE=1` replaces `lift @ Flux_flat` with a direct Triton lift kernel.

They remain useful for future shape/GPU experiments, but the default path keeps the fastest measured configuration.

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

The benchmark also supports A/B isolation of each optimization path:

```bash
python benchmarks/scenario1_benchmark.py --device cuda --steps 1000 --cuda-graph --disable-triton-volume-rhs
python benchmarks/scenario1_benchmark.py --device cuda --steps 1000 --cuda-graph --disable-triton-boundary-ade
python benchmarks/scenario1_benchmark.py --device cuda --steps 1000 --cuda-graph --disable-scaled-flux-kernels
python benchmarks/scenario1_benchmark.py --device cuda --steps 1000 --cuda-graph --enable-fused-state-accumulation
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

All three are **correct**, but on the current V100 + scenario1 shape they were not consistently faster than the default path, so they remain benchmark-gated.

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

### A. End-to-end `examples/scenario1/main.py` progression

The original baseline and the current script do **not** use the same total simulation time (`2.0s` vs current `0.1s`), so direct wall-clock comparison is not apples-to-apples. The comparable metric is **time per step**.

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

All rows below use the same fixed-step benchmark:

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

## Files most relevant to the final design

- `edg_acoustics/device_ini.py`
- `edg_acoustics/acoustics_simulation.py`
- `edg_acoustics/time_integration.py`
- `benchmarks/scenario1_benchmark.py`
- `tests/test_scenario1_golden.py`
- `tests/scenario1_utils.py`
- `examples/scenario1/main.py`

## Deferred work

Remaining deep-optimization candidates are correctness-riskier and should be benchmark-gated:

- custom derivative or lift kernels beyond cuBLAS for different `Np/N_tets` shapes or newer GPUs;
- larger graph chunks or full-loop capture only if graph launch count becomes significant again.
