# TileLang Derivative GEMM on MetaX C500

本文档记录 `scenario1_profile_lc0p20.msh` 上 fp64 derivative GEMM 的
TileLang 接入结果、配置选择依据，以及完整 timestep 的实际收益。

## 背景

在 MetaX C500 上，`scenario1_profile_lc0p20.msh` 的 timestep 主路径已经完成：

- AoS 状态布局；
- compact interior flux；
- affine metric volume/surface RHS；
- full CUDA Graph；
- TileLang lift `bm48_bn64_bk16_s0_t256_fullcol`。

在这一基线上，剩余最大瓶颈之一是 merged derivative GEMM：

```text
[105, 35] @ [35, N]
```

也就是：

```text
D_merged[105, 35] @ Q_flat[35, 4 * N_tets]
```

对于 `scenario1_profile_lc0p20.msh`：

- `Np = 35`
- `N_tets = 45285`
- `N = 4 * N_tets = 181140`

对应 standalone kernel sweep 中的 GEMM shape 是：

```text
A = (105, 35)
B = (35, 1377572)
```

这里的 `1377572` 不是 timestep 中的真实 `N=181140`，而是 sweep 脚本使用的更大
long-column 规模，用来放大 steady-state kernel 差异。对配置选择真正关键的是：

- 同样的 `M=105, K=35`；
- 同样的 fp64 TileLang GEMM 实现路径；
- 同样的 `N >> 1` 长列访问模式。

## 最终接入的运行时配置

当前运行时固定接入的 TileLang derivative GEMM 配置为：

```text
bm128_bn64_bk4_s0_t256_fullcol
```

2026-07-04 更新：

- repo 早先默认值是 `bm112_bn64_bk12_s1_t256_fullcol`；
- 最新 `c500-next` focused sweep 显示，在真实 timestep 更接近的 `N=181140` 上，
  `bm128_bn64_bk4_s0_t256_fullcol` 为 `0.541696 ms`, `2.457797 TFLOPS`,
  `speedup=1.391x`；
- 对比旧默认 `bm112_bn64_bk12_s1_t256_fullcol` 的 `0.583936 ms`,
  `2.280009 TFLOPS`, `speedup=1.291x`，新配置在代表性 runtime 列规模上快约
  `7%`；
- 在更大的 `N=1377572` 上，新配置也仍然领先，`bm128_bn64_bk4_s1_t256_fullcol`
  达到 `3.6348 ms`, `2.7856 TFLOPS`, `speedup=1.463x`。

因此代码默认值已经切换到 `bm128_bn64_bk4_s0_t256_fullcol`。本文档后面的
`bm112_*` 结果保留为历史测量记录，用来说明这条优化路线的演进过程。

对应代码：

- `edg_acoustics/tilelang_derivative_gemm.py`
- `edg_acoustics/acoustics_simulation.py`

该路径替换的是 merged derivative 的默认 `torch.mm(self._D_merged, q_by_node, ...)`
调用；只在满足以下条件时启用：

- CUDA 设备；
- MetaX / MACA 运行时；
- merged derivatives 已启用；
- `Np = 35`；
- fp64；
- contiguous 输入输出 buffer。

如果编译、正确性校验或 CUDA Graph replay 校验失败，会自动回退到原始
`torch.mm` 路径。

## Standalone Sweep 结果

测试环境：

- GPU: MetaX C500
- fp64 theoretical peak: `4.0 TFLOPS`
- bandwidth theoretical peak: `1.8 TB/s`
- shared memory per block: `64 KiB`
- warp size: `64`

参考 baseline 是 MetaX 上的 `torch.mm` / mcBLAS 路径。

两次独立 deep sweep 的关键结果如下。

复现这一组 standalone sweep 的命令是：

```bash
rtk python benchmarks/mma_tilelang_v6.py \
    --shape derivative \
    --N 1377572 \
    --profile-backend event \
    --sweep-level c500-deep
```

| Config | Trial 1 | Trial 2 | 结论 |
| --- | ---: | ---: | --- |
| `torch.mm` baseline | `5.3238 ms`, `1.9019 TFLOPS` | `5.3196 ms`, `1.9034 TFLOPS` | baseline |
| `bm32_bn64_bk16_s1_t256_fullcol` | `5.4720 ms`, `0.973x` | `5.4738 ms`, `0.972x` | 旧配置，不如 baseline |
| `bm112_bn32_bk8_s0_t128_fullcol` | `4.7259 ms`, `1.127x` | `4.7251 ms`, `1.126x` | 有收益，但不是最优 |
| `bm112_bn64_bk12_s0_t256_fullcol` | `3.9750 ms`, `1.339x` | `3.9715 ms`, `1.339x` | 很强，接近最优 |
| `bm112_bn64_bk12_s1_t256_fullcol` | `3.9668 ms`, `1.342x` | `3.9363 ms`, `1.351x` | 最优 |

最优配置的 roofline 观测值：

- Trial 1: `2.5524 TFLOPS`, `0.389 TB/s`
- Trial 2: `2.5723 TFLOPS`, `0.392 TB/s`

这相当于：

- 约 `63.8% ~ 64.3%` 的 C500 fp64 峰值；
- 只使用了约 `21.6% ~ 21.8%` 的理论带宽峰值。

因此这一 kernel 在 C500 上主要是 **compute-bound**，不是 bandwidth-bound。

## 为什么 `bm112_bn64_bk12_s1_t256_fullcol` 最好

这个结论不是拍脑袋选出来的，而是由 shape、硬件和 TileLang fragment 约束共同决定的。

### 1. `bm112` 几乎贴合 `M=105`

真实 derivative GEMM 的 `M=105` 很尴尬：

- 太大，不适合 `bm32` 这类窄 tile；
- 又没有大到需要把 `M` 继续拆成很多 block。

`bm112` 的好处是：

- 一个 block 就能覆盖几乎整个 `M` 维；
- 只多 pad `7` 行；
- 配合 `bk12` 后，总 padded inflation 只有约 `1.10x`。

对比旧的 `bm32_bn64_bk16_*` 系列，日志里 padded inflation 约为 `1.67x`。  
也就是说，旧配置有太多“为 padding 做的无效计算”，而 `bm112` 明显更贴 shape。

### 2. `bk12` 比 `bk16` 更适合 `K=35`

这个 GEMM 的 `K=35` 很小。

- `bk16` 需要把 `K` pad 到 `48`；
- `bk12` 只需要 pad 到 `36`。

这直接减少了 K 维无效工作量。  
从日志也能看到，`bm112_bn64_bk12_*` 明显优于同家族 `bk16` 版本。

### 3. `bn64` 在大 N 场景上足够宽，而且仍然省 shared memory

这里的 `N` 非常大，本质上是“长列流式输出”。

`bn64` 的意义是：

- 让每次 block 处理足够多的列，摊薄 A/权重加载成本；
- 仍然把 shared memory 控制在 `16 KiB`；
- 配合 C500 的 `warp_size=64` 与 TileLang `fullcol` 策略，fragment 布局更稳定。

`bn32` 也能跑，但吞吐不如 `bn64`。

### 4. `t256` 明显优于 `t128`

同样的 `bm112_bn64_bk12_fullcol` 家族里：

- `t128` 约 `5.34 ms`
- `t256` 约 `3.94 ~ 3.97 ms`

差距非常大。  
这说明在这个 shape 上，`256` 线程块更适合：

- shared tile 装载；
- fragment 映射；
- GEMM 内部并行度展开。

### 5. `s1` 比 `s0` 小幅但稳定地更好

在同一 tile shape 下：

- Trial 1: `s1` 比 `s0` 快约 `0.2%`
- Trial 2: `s1` 比 `s0` 快约 `0.9%`

提升不算大，但方向稳定，因此最终取 `s1`。

### 6. `fullcol` 是可编译且最快的有效策略

很多 `square` 版本直接失败，典型报错是：

```text
Loop layout is not injective
```

也就是说，这不是单纯“square 比 fullcol 慢”，而是对这组
`M=105, K=35, N>>1`、`warp=64`、fp64 fragment 布局来说，`fullcol`
既更稳，也更快。

## 真实 timestep 接入验证

在接入实际 timestep 路径后，使用以下命令对比：

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

实测结果：

| Variant | ms/step | 说明 |
| --- | ---: | --- |
| `TileLang lift + TileLang derivative GEMM` | `6.709760` | full CUDA Graph，成功进入 graph |
| `TileLang lift + baseline derivative GEMM` | `7.368161` | full CUDA Graph |

端到端收益：

- 每 step 降低 `0.658401 ms`
- 整体加速 `1.09813x`
- step 时间下降 `8.94%`

运行时状态也符合预期：

- `tilelang_derivative_gemm_enabled=1`
- `tilelang_derivative_gemm_graph_capture_supported=1`
- `tilelang_derivative_gemm_fallback_reason=none`
- `cuda_graph_mode=full`

这说明该 kernel 不只是 standalone benchmark 漂亮，而是真正进入了完整 timestep
路径，并且可以被 full CUDA Graph 捕获和 replay。

同时，这个真实 timestep 结果也验证了新的 runtime 默认
`bm128_bn64_bk4_s0_t256_fullcol` 确实优于前一个 TileLang derivative 默认
`bm112_bn64_bk12_s1_t256_fullcol`。后者的历史 full-step 结果是 `6.829143 ms/step`，
因此这次默认切换又额外减少了 `0.119383 ms/step`，约 `1.78%`。

## 如何理解 1.39x kernel 提升只换来 1.10x step 提升

这是正常现象。

standalone sweep 中，新的 runtime 默认在真实 timestep 更接近的 `N=181140` 上相对
baseline 大约是：

- `1.391x`

但 timestep 里还有：

- `compact_interior_flux_kernel`
- `volume_surface_rhs_affine_metric_aos_vector_kernel`
- TileLang lift
- 各类 boundary kernel

因此端到端不会等于单个 kernel 的加速比。

根据观测到的：

- old step = `7.368161 ms`
- new step = `6.709760 ms`
- delta = `0.658401 ms`

结合 standalone derivative GEMM 的 `1.391x` 加速，可以反推出一个近似结论：

- 旧 timestep 中，derivative GEMM 大约占 `2.0 ~ 2.1 ms/step`
- 约为总 step 的 `28%` 左右

这里是基于端到端结果做的估算，不是 profiler 直接读数，但和之前的 hotspot
分析是一致的。

## 当前结论

在 MetaX C500 上，新的 runtime 默认 `bm128_bn64_bk4_s0_t256_fullcol` 已经有充足
standalone 证据表明它优于旧默认 `bm112_bn64_bk12_s1_t256_fullcol`，尤其是在真实
timestep 更接近的 `N=181140` 上。

同时，旧默认 `bm112_bn64_bk12_s1_t256_fullcol` 仍然保留了以下已验证历史结论：

- derivative GEMM 在此前 sweep 中的阶段性最优配置；
- 可以安全接入真实 timestep；
- 可以进入 full CUDA Graph；
- 可以在 `scenario1_profile_lc0p20.msh` 上带来稳定端到端收益。

因此当前推荐结论是：

1. 保留 `bm48_bn64_bk16_s0_t256_fullcol` 作为 TileLang lift 默认实现；
2. 采用 `bm128_bn64_bk4_s0_t256_fullcol` 作为当前 TileLang derivative GEMM 默认实现；
3. 继续使用 full CUDA Graph；
4. 后续性能工作重点应转向新的 step 主瓶颈，而不是再回到旧的 cuBLAS derivative 路径。

## 后续优化方向

在 derivative GEMM 获得这次收益后，下一阶段更值得继续看的通常是：

- `compact_interior_flux_kernel`
- `volume_surface_rhs_affine_metric_aos_vector_kernel`
- TileLang lift 与其周边数据排布

也就是说，derivative GEMM 这条线上已经从“明显短板”变成“已被有效收敛的热点”，
继续投入应以 profiler 再次确认新的第一瓶颈为前提。
