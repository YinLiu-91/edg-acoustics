# trace 分析：AoS vs SoA 对比

## 总览

| 指标 | SoA（旧） | AoS（新） | 变化 |
|------|----------|----------|------|
| GPU 总耗时 | 2478.5ms | **2003.2ms** | **-19.2%** |
| 总事件数 | 56,013 | — | — |

## 算子耗时对比

| 算子 | SoA 占比 | SoA 单次(us) | AoS 占比 | AoS 单次(us) | 变化 |
|------|---------|-------------|---------|-------------|------|
| **BLAS GEMM (快)** | — | — | ~25% (500ms) | **725us** (每 RHS 1次) | — |
| **BLAS GEMM (慢)** | — | — | ~25% (501ms) | **894us** (每 RHS 1次) | — |
| **BLAS GEMM 合计** | 41.3% (1023ms) | 827 | **50.0%** (1001ms) | — | 不变 |
| **compact_interior_flux** | 28.0% (693ms) | 1121 | **18.8%** (377ms) | 610 | **-46%** ✅ |
| **volume_surface_rhs** | 11.6% (287ms) | 465 | **12.3%** (247ms) | 400 | -14% |
| **boundary_ri_flux** | 9.2% (228ms) | 369 | **3.4%** (67ms) | 109 | **-71%** ✅ |
| **boundary_rp_cp_flux** | 2.9% (72ms) | 117 | **1.3%** (25ms) | 41 | **-65%** ✅ |
| **boundary_rp_flux** | 2.4% (58ms) | 95 | **1.0%** (21ms) | 33 | **-64%** ✅ |
| **memcpy DTOD** | 2.0% (49ms) | 36 | **7.2%** (145ms) | 106 | ⚠️ +195% |
| 其他 | 2.6% | — | 6.0% | — | — |

## 关键发现

1. **AoS kernel 生效**：`compact_interior_flux_aos_kernel` 替代了旧 kernel，单次耗时 -46%
2. **边界 kernel 大幅加速**：3 个边界 kernel 合计从 14.5% → 5.7%，AoS q 读使邻接访存生效
3. **volume 核使用新 kernel**：`volume_surface_rhs_affine_metric_aos_vector_kernel` — edg 的高阶优化已触发
4. **memcpy 增大**：7.2% vs 2.0%，可能来自 AoS 布局转换的临时 buffer
5. **BLAS 占比相对升高**：BLAS 绝对值未变，但因其他 kernel 加速，占比从 41% → 50%，成为更突出的瓶颈

## 下一步

BLAS GEMM 占 50% 是唯一剩余大瓶颈。要破这个需要 `derivative_volume_surface_rhs_kernel`（融合导数+体积+表面积累），但会改 golden。


# v100
1. total time is 1.27s
1. in one time step:
    - 2 gemm: 370us*3 and 280us*3
    - volume_surface_rhs_affine_metric_aos_vector_kernel 700us*3
    - compact_interior_flux_aos_tile_local_u8_variant_kernel 590us*3

# v100 vs c500
1. surface rhs is 700us vs 400us
2. compact_interior_flux is 590us vs 610us
3. gemm is (370+280)/2=325us vs 810us
4. gemm is the hot spot for C500
5. C500 bandwidth is larger than v100