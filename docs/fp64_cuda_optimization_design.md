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

### 7. CUDA Graph replay

For CUDA runs, `time_integration(..., use_cuda_graph=True)` captures and replays one full packed Taylor step:

1. snapshot current packed state and boundary ADE state;
2. warm up one packed step;
3. capture one full `step_dt_packed()` into a `torch.cuda.CUDAGraph`;
4. restore the original state;
5. replay the graph at every time step.

This is the most important optimization for scenario1-sized fp64 workloads because it removes the CPU launch overhead of many small CUDA kernels.

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

Current solver-loop timing printed by `time_integration()` for the CUDA Graph run is also much lower:

| Stage | Steps | Solver time | Solver ms/step | Relative to previous optimized |
| --- | ---: | ---: | ---: | ---: |
| Earlier optimized, pre-CUDA-Graph | 3284 | 15.68 s | 4.77 | 1.00x |
| Current optimized + CUDA Graph | 3284 | 5.02 s | 1.53 | 3.12x faster |

### B. CPU / GPU hot-path benchmark comparison

All rows below use the same fixed-step benchmark:

```bash
python benchmarks/scenario1_benchmark.py --steps 1000 ...
```

| Mode | Device | CUDA Graph | 1000-step time | ms/step | Relative to CPU |
| --- | --- | --- | ---: | ---: | ---: |
| Fixed-step benchmark | CPU | No | 4035.35 ms | 4.04 | 1.00x |
| Fixed-step benchmark | CUDA | No | 4338.09 ms | 4.34 | 0.93x |
| Fixed-step benchmark | CUDA | Yes | 1486.32 ms | 1.49 | 2.72x faster |

Interpretation:

- on this small fp64 workload, **CUDA eager is not enough** to beat CPU clearly;
- **CUDA Graph is what makes GPU clearly faster** by removing repeated launch overhead;
- the current landed design therefore keeps the packed PyTorch fallback and uses CUDA Graph as the practical speedup path.

## Files most relevant to the final design

- `edg_acoustics/device_ini.py`
- `edg_acoustics/acoustics_simulation.py`
- `edg_acoustics/time_integration.py`
- `benchmarks/scenario1_benchmark.py`
- `tests/test_scenario1_golden.py`
- `tests/scenario1_utils.py`
- `examples/scenario1/main.py`

## Deferred work

An optional fused Triton interior flux kernel remains deferred.

Reason:

- CUDA Graph already delivers a clear speedup on the target workload;
- the current packed PyTorch path is easier to validate and maintain;
- Triton can still be added later with the current implementation as the correctness fallback.
