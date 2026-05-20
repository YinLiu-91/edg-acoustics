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

On CUDA this path is additionally fused with a Triton kernel by default. The kernel computes all four packed RHS components in one pass over `(Np, N_tets)`, reducing graph-internal elementwise launches and memory traffic. Set `EDG_ACOUSTICS_TRITON_VOLUME_RHS=0` to force the PyTorch fallback.

### 7. Fused interior flux assembly

On CUDA, interior face gather, jump calculation, and upwind flux assembly are fused into a Triton kernel by default. The fallback remains the packed PyTorch path.

The fused kernel writes the same packed face layout:

```python
Flux_flat: [4 * Nfp, 4 * N_tets]
```

Boundary ADE/impedance overrides still run after interior flux assembly, then the full face flux is scaled by `Fscale` and lifted. Set `EDG_ACOUSTICS_TRITON_INTERIOR_FLUX=0` to force the fallback.

### 8. Fused RI-only boundary flux

The largest scenario1 boundary group is RI-only and does not carry RP/CP ADE state. On CUDA that group is fused into one Triton kernel that gathers boundary state, computes `vn/ou/in`, and writes packed boundary fluxes. RP/CP impedance boundaries stay on the validated PyTorch path because they update recursive history variables. Set `EDG_ACOUSTICS_TRITON_BOUNDARY_RI=0` to force the fallback.

### 9. CUDA Graph replay and optional chunking

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
| Current CUDA Graph + fused kernels | `main.py`, `total_time=0.1` | 3284 | 2.80 s solver loop | 0.85 | 14.46x faster |

Current solver-loop timing printed by `time_integration()` for the CUDA Graph run is also much lower:

| Stage | Steps | Solver time | Solver ms/step | Relative to previous optimized |
| --- | ---: | ---: | ---: | ---: |
| Earlier optimized, pre-CUDA-Graph | 3284 | 15.68 s | 4.77 | 1.00x |
| Current optimized + CUDA Graph | 3284 | 5.02 s | 1.53 | 3.12x faster |
| Current CUDA Graph + fused kernels | 3284 | 2.80 s | 0.85 | 5.61x faster |

### B. CPU / GPU hot-path benchmark comparison

All rows below use the same fixed-step benchmark:

```bash
python benchmarks/scenario1_benchmark.py --steps 1000 ...
```

| Mode | Device | CUDA Graph | 1000-step time | ms/step | Relative to CPU |
| --- | --- | --- | ---: | ---: | ---: |
| Fixed-step benchmark | CPU | No | 5124.70 ms | 5.12 | 1.00x |
| Fixed-step benchmark | CUDA | No | 2667.92 ms | 2.67 | 1.92x faster |
| Fixed-step benchmark | CUDA | Yes | 790.88 ms | 0.79 | 6.48x faster |
| Fixed-step benchmark, no receiver sampling | CUDA | Yes | 779.20 ms | 0.78 | 6.58x faster |
| Fixed-step benchmark, chunk `K=32` | CUDA | Yes | 832.04 ms / 1024 steps | 0.81 | 6.31x faster |

Interpretation:

- on this small fp64 workload, packed CUDA eager is now faster than CPU after Triton fusion, but it is still limited by repeated launch overhead;
- **CUDA Graph is what makes GPU much faster** by removing that repeated launch overhead;
- Triton volume/flux/boundary fusion and batched derivatives reduce real graph-internal GPU work, so CUDA eager and CUDA Graph both improve;
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

- tensorized RP/CP ADE impedance boundary groups to remove the remaining boundary-history PyTorch kernels;
- custom derivative kernels beyond batched cuBLAS for `Np=35`;
- larger graph chunks or full-loop capture only if receiver-heavy workloads make graph launch count significant again.
