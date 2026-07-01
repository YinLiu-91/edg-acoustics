# main.py --steps 100 性能分析

> 数据来源：`mcTracer` 采集的 `tracer_out-219863.json`（100 步，Nt=4，N_tets=45285）

## 总览

- GPU 总耗时：**2478.5 ms**（100 步）
- 单步耗时：**~24.8 ms**
- 总事件数：56,013
- 唯一 kernel 名：165

## 单步计算流程

TSI_TI（4 阶泰勒）每步执行 3 次 RHS 求值，每次依次触发：

```
1. BLAS GEMM (D×Q → dQ)           ← 最大瓶颈
2. compact_interior_flux_kernel     ← 内部面通量 (Triton)
3. boundary_ri_flux_kernel          ← 边界 11（硬墙）
4. boundary_rp_flux_kernel          ← 边界 13（carpet）
5. boundary_rp_cp_flux_kernel       ← 边界 14（panel）
6. volume_surface_rhs_kernel        ← 体积 RHS + 表面积累 (Triton)
7. memcpy DTOD (buffer 拷贝)
```

## 算子耗时排序

| 算子 | 耗时占比 | 总耗时(ms) | 单次(us) | 调用次数 | 类型 |
|------|---------|-----------|----------|---------|------|
| **BLAS GEMM (dgemm 128×128)** | **41.3%** | 1022.5 | 827.3 | 1236 | PyTorch mm |
| **compact_interior_flux_kernel** | **28.0%** | 693.0 | 1121.3 | 618 | Triton |
| **volume_surface_rhs_kernel** | **11.6%** | 287.3 | 465.0 | 618 | Triton |
| **boundary_ri_flux_kernel** | **9.2%** | 228.0 | 368.9 | 618 | Triton |
| **boundary_rp_cp_flux_kernel** | **2.9%** | 72.3 | 116.9 | 618 | Triton |
| **boundary_rp_flux_kernel** | **2.4%** | 58.4 | 94.5 | 618 | Triton |
| **memcpy DTOD** | **2.0%** | 49.2 | 35.9 | 1371 | 显存拷贝 |
| PyTorch elementwise | 1.5% | 36.5 | 3~659 | ~5400 | CUDA kernel |
| Sort/SearchSorted | 0.8% | 19.4 | 15~1089 | ~300 | CUDA kernel |
| 其他 | 0.3% | 7.8 | — | ~500 | — |

## 五个 Triton kernel 耗时占比

| Kernel | 总耗时(ms) | 占比 | 单次耗时(us) |
|--------|-----------|------|-------------|
| compact_interior_flux_kernel | 693.0 | 52.4% | 1121.3 |
| volume_surface_rhs_kernel | 287.3 | 21.7% | 465.0 |
| boundary_ri_flux_kernel | 228.0 | 17.2% | 368.9 |
| boundary_rp_cp_flux_kernel | 72.3 | 5.5% | 116.9 |
| boundary_rp_flux_kernel | 58.4 | 4.4% | 94.5 |
| **Triton 合计** | **1339.0** | **54.0%** | — |

## BLAS GEMM 详情

| GEMM 类型 | 总耗时(ms) | 调用次数 | 单次(us) |
|-----------|-----------|---------|----------|
| `dgemm_nn_128x128x16` | 1022.5 | 1236 | 827.3 |
| `dgemm_nn_128x64x64` | 1.3 | 9 | 139.2 |
| `dgemm_nt_32x32x128` | 0.02 | 1 | 17.9 |

GEMM 在 `_compute_packed_derivatives` 中通过 `torch.mm` 调用，用于计算 `D×Q` 体积微分矩阵。`D` 是 `[Np, Np]`（35×35），`Q` 是 `[Np, 4*N_tets]`（35×181140）。每次 RHS 求值触发 3 次 mm（Dr×Q, Ds×Q, Dt×Q）。

## 优化建议

1. **BLAS GEMM（收益最大）**：合并 Dr/Ds/Dt 为 `[3*Np, Np]` 的合并矩阵，一次性 batched mm 替代 3 次独立 mm（已在代码中作为 `_D_merged` + `_use_merged_derivatives` 实现，需确认是否开启）
2. **Triton kernel 融合**：将 compact_interior_flux + volume_surface_rhs 融合为单个 kernel，消除中间 buffer 和 memcpy
3. **边界 kernel 融合**：三个 boundary 各处理不同 BC，可尝试合并为单 kernel 处理所有 BC 类型

