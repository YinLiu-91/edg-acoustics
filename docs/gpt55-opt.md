# Torch CUDA 性能优化方案

本文基于当前 `edg_acoustics` 代码中的 Torch/CUDA 使用方式，给出面向 DG 声学求解器的优化路线。当前 GPU 计算主要集中在 `edg_acoustics/acoustics_simulation.py`、`edg_acoustics/time_integration.py`、`edg_acoustics/preprocessing.py`、`edg_acoustics/boundary_condition.py` 和 `edg_acoustics/mesh.py`。

## 1. 当前性能热点判断

### 1.1 时间推进是主热点

`AcousticsSimulation.time_integration()` 在每个时间步调用 `time_integrator.step_dt()`，而 `TSI_TI.step_dt()` 又会按 `Nt` 多次调用 `RHS_operator()`。因此性能关键路径为：

```text
time_integration()
  -> TSI_TI.step_dt()
    -> RHS_operator()
      -> face jump / flux / boundary update / grad_3d / lift
```

优化时应优先关注 `RHS_operator()` 和 `grad_3d()`，其次才是初始化、后处理和文档示例脚本。

### 1.2 主要 CUDA 性能问题

当前代码存在以下典型瓶颈：

- 默认使用 `torch.float64`，GPU 吞吐和显存带宽压力明显高于 `float32`。
- `RHS_operator()` 每次调用都会新分配 `dVx/dVy/dVz/dP` 等临时张量。
- `grad_3d()` 对每个变量重复执行 `Dr @ U`、`Ds @ U`、`Dt @ U`，小矩阵乘和逐元素 kernel 数量多。
- 边界条件更新中存在 Python 层循环和多次高级索引，容易产生大量小 kernel。
- 初始化阶段有多处 `.cpu().numpy()` 和 NumPy/Modepy 计算，容易引入 CPU/GPU 往返。
- `time_integration()` 每步执行接收点采样 `torch.diag(self.sampleWeight @ self.P[:, self.nodeindex])`，对少量接收点会产生额外小矩阵乘。
- `@profile`、`line_profiler`、重复 `triton` import 和未启用的 Triton kernel 会增加维护成本，并可能影响生产环境依赖。

## 2. 优先级 P0：低风险、应立即完成

### 2.1 增加可配置 dtype，并默认评估 float32

当前 `edg_acoustics/device_ini.py` 写死：

```python
dtype = torch.float64
```

建议改为可配置：

```python
import os
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
_dtype_name = os.getenv("EDG_ACOUSTICS_DTYPE", "float32")
dtype = getattr(torch, _dtype_name)
```

然后分别运行：

```bash
EDG_ACOUSTICS_DTYPE=float64 python -m pytest
EDG_ACOUSTICS_DTYPE=float32 python -m pytest
```

如果 `float32` 与 golden result 的误差可接受，CUDA 路径应默认使用 `float32`。DG 显式时间推进对精度敏感，不能直接假设 `float32` 一定通过，需要用物理量、传递函数和最终误差阈值一起验证。

### 2.2 统一张量创建的 device/dtype

代码中大量使用：

```python
torch.zeros(shape).to(device_ini.device).to(device_ini.dtype)
```

建议统一改为：

```python
torch.zeros(shape, device=device_ini.device, dtype=device_ini.dtype)
```

适用位置包括：

- `compute_collocation_nodes()` 中的 `xyz`
- `compute_lift()` 中的 `Emat`
- `geometric_factors_3d()` 中的 `rst_xyz`
- `normals_3d()` 中的 `n_xyz`
- `RHS_operator()` 中的临时变量
- `BoundaryCondition.init_ADEvariables()`
- `InitialCondition.VXinit/VYinit/VZinit()`

这样可以减少不必要的默认 CPU 分配和 `.to()` 转换，也让代码更容易检查 dtype/device 是否一致。

### 2.3 移除生产路径上的 line_profiler 装饰器

`acoustics_simulation.py` 中 `build_maps_3d()`、`grad_3d()`、`RHS_operator()` 使用了 `@profile`，并直接导入：

```python
from line_profiler import profile
```

建议改成可选 profiling，避免普通运行依赖 `line_profiler`：

```python
try:
    from line_profiler import profile
except ImportError:
    def profile(func):
        return func
```

更进一步，建议在性能测试脚本中使用 `torch.profiler` 和 Nsight，而不是把 profiling 装饰器放在核心库代码里。

## 3. 优先级 P1：减少每步分配和小 kernel

### 3.1 预分配 RHS 临时缓冲区

`RHS_operator()` 每次调用都会创建：

```python
dVx = torch.zeros_like(self.Fscale)
dVy = torch.zeros_like(dVx)
dVz = torch.zeros_like(dVx)
dP = torch.zeros_like(dVx)
```

时间步数较多时，这会造成大量 GPU allocator 调用。建议在 `init_local_system()` 或 `init_TimeIntegrator()` 后预分配：

```python
def init_runtime_buffers(self):
    shape = self.Fscale.shape
    kwargs = {"device": self.device, "dtype": device_ini.dtype}
    self._dVx = torch.empty(shape, **kwargs)
    self._dVy = torch.empty(shape, **kwargs)
    self._dVz = torch.empty(shape, **kwargs)
    self._dP = torch.empty(shape, **kwargs)
    self._fluxVx = torch.empty(shape, **kwargs)
    self._fluxVy = torch.empty(shape, **kwargs)
    self._fluxVz = torch.empty(shape, **kwargs)
    self._fluxP = torch.empty(shape, **kwargs)
```

在 `RHS_operator()` 中复用这些 buffer，并用 `torch.index_select(..., out=...)` 或 `torch.sub(..., out=...)` 降低中间张量数量。需要注意：不要把这些 buffer 作为返回值长期保存到外部对象，否则下一次调用会覆盖内容。

### 3.2 缓存 flatten view 和索引

`RHS_operator()` 中频繁出现：

```python
Vx.reshape(-1)[self.vmapM]
P.reshape(-1)[self.BCnode[index]["vmap"]]
(self.n_xyz[0]).reshape(-1)[self.BCnode[index]["map"]]
```

建议在初始化阶段缓存不随时间变化的 flatten view 与索引：

```python
self._vmapM = self.vmapM.long()
self._vmapP = self.vmapP.long()
self._nx_flat = self.n_xyz[0].reshape(-1)
self._ny_flat = self.n_xyz[1].reshape(-1)
self._nz_flat = self.n_xyz[2].reshape(-1)
```

对每个 `BCnode` 也缓存：

```python
node["map"] = node["map"].long()
node["vmap"] = node["vmap"].long()
node["nx"] = self._nx_flat[node["map"]]
node["ny"] = self._ny_flat[node["map"]]
node["nz"] = self._nz_flat[node["map"]]
```

这样边界更新时不需要反复 reshape 和索引法向量。

### 3.3 合并 flux 计算

`UpwindFlux.FluxVx/FluxVy/FluxVz/FluxP()` 分别返回一个新张量，会触发多组逐元素 CUDA kernel。建议增加一个一次性计算四个 flux 的接口：

```python
def compute_all(self, dvx, dvy, dvz, dp, out_vx, out_vy, out_vz, out_p):
    torch.addcmul(self.n1rho * dp, self.cn1s, dvx, out=out_vx)
    out_vx.add_(self.cn1n2 * dvy).add_(self.cn1n3 * dvz)
    # 同理计算 out_vy/out_vz/out_p
```

如果 PyTorch 表达式融合效果不理想，可把四个 flux 合并为一个 Triton kernel。由于这里是同 shape 的逐元素运算，Triton 改造风险低、收益通常稳定。

### 3.4 优化接收点采样

当前每步执行：

```python
self.prec[:, StepIndex] = torch.diag(self.sampleWeight @ self.P[:, self.nodeindex])
```

如果接收点数 `N_rec` 较小，矩阵乘和 `diag` 都偏重。建议改成逐接收点加权求和：

```python
p_nodes = self.P[:, self.nodeindex].T
self.prec[:, StepIndex] = (self.sampleWeight * p_nodes).sum(dim=1)
```

这会避免构造 `N_rec x N_rec` 矩阵，语义也更直接。

## 4. 优先级 P2：改造核心数值算子

### 4.1 将 grad_3d 改为一次计算多个变量

`RHS_operator()` 中目前：

```python
dPdx, dPdy, dPdz = self.grad_3d(P, "xyz")
self.grad_3d(Vx, "x") + self.grad_3d(Vy, "y") + self.grad_3d(Vz, "z")
```

这会对 `P/Vx/Vy/Vz` 分别做多次小矩阵乘。建议新增批量版本，将变量堆叠成 `[4, Np, N_tets]`：

```python
Q = torch.stack([P, Vx, Vy, Vz], dim=0)
dQdr = torch.matmul(self.Dr, Q)
dQds = torch.matmul(self.Ds, Q)
dQdt = torch.matmul(self.Dt, Q)
```

然后用 `rst_xyz` 组合得到所需导数。这样可以减少 Python 调用次数，并让 CUDA 后端看到更大的 batch matmul。注意 `torch.matmul(self.Dr, Q)` 的维度广播需要实测；必要时可使用 `torch.einsum("ij,bjk->bik", self.Dr, Q)` 作为更清晰的实现。

### 4.2 融合 lift 与 Fscale

当前 RHS 中有四次：

```python
self.lift @ (self.Fscale * flux)
```

其中 `self.Fscale * flux` 会产生中间张量。建议预先检查 `Fscale` 是否仅依赖元素和面节点，若可接受，可将缩放并入 flux buffer：

```python
fluxP.mul_(self.Fscale)
surface_P = self.lift @ fluxP
```

这样可复用 flux buffer 作为输入，减少中间分配。若后续使用 `torch.compile`，这种 in-place 写法要与编译器兼容性一起测试。

### 4.3 边界条件 pole 维度向量化

当前 `RHS_operator()` 对 `RP` 和 `CP` pole 使用 Python 循环：

```python
for i in range(paras["RP"].shape[1]):
    BCvar[index]["in"] += paras["RP"][0, i] * BCvar[index]["phi"][i]
```

建议改成 pole 维度的向量化：

```python
A = paras["RP"][0, :].reshape(-1, 1)
zeta = paras["RP"][1, :].reshape(-1, 1)
BCvar[index]["in"].add_((A * BCvar[index]["phi"]).sum(dim=0))
BCvar[index]["phi"].copy_(BCvar[index]["ou"].unsqueeze(0) - zeta * BCvar[index]["phi"])
```

`CP` 也可同理向量化。这样可以把每个 pole 的多个小 kernel 合并成少量张量操作。

### 4.4 评估 torch.compile

在 PyTorch 2.x 环境下，可尝试编译核心 step：

```python
sim.RHS_operator = torch.compile(sim.RHS_operator, mode="reduce-overhead")
```

注意事项：

- `BCvar` 是 `list[dict]`，动态图结构可能导致 graph break。
- Python 循环、对象方法调用、动态 shape 和 `.item()` 会降低编译收益。
- 更推荐先把 `RHS_operator()` 中的数值核心拆成纯 tensor 函数，再对纯函数使用 `torch.compile`。

## 5. 优先级 P3：初始化和 mesh 构建优化

### 5.1 初始化阶段尽量留在 CPU，完成后一次搬到 GPU

`compute_collocation_nodes()`、`compute_lift()`、`sample3D()` 中多处使用 Modepy、NumPy、SciPy，这些库本身在 CPU 上运行。当前代码在 CPU/GPU 之间来回转换，例如：

```python
return rst.cpu().numpy(), xyz
faceR.cpu().numpy()
old_nodes[:, :, i].cpu()
```

建议策略：

1. 纯拓扑和几何初始化先在 CPU/NumPy 完成。
2. 得到最终 `Dr/Ds/Dt/lift/Fscale/vmapM/vmapP/BCnode` 后一次性转到 GPU。
3. 时间推进开始后，避免再出现 `.cpu()`、`.numpy()`、`.item()`。

这样可以降低初始化复杂度，并避免隐式 CUDA 同步。

### 5.2 build_maps_3d 不建议用 Python 双循环跑在 GPU 张量上

`build_maps_3d()` 中存在：

```python
for ke in range(N_tets):
    for face in range(4):
        ...
        idMP = torch.nonzero(...)
```

这是初始化阶段的热点。若只在启动时运行一次，建议先用 CPU/NumPy 的排序和哈希算法完成；如果必须在 GPU 上做，则应启用并完善已有的 Triton kernel 或改为批量向量化。当前文件中的 `build_maps_kernel` 被注释，不能产生收益。

## 6. Profiling 与验证流程

### 6.1 先建立基准

建议新增脚本 `benchmarks/profile_scenario1.py`，固定 mesh、`Nx`、`Nt`、步数和 dtype，输出：

- 初始化耗时
- 单步平均耗时
- `RHS_operator()` 平均耗时
- 显存峰值
- `float32` 与 `float64` 结果误差

最小 profiling 代码：

```python
import torch

torch.cuda.reset_peak_memory_stats()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
sim.time_integration(n_time_steps=100)
end.record()
torch.cuda.synchronize()

print("elapsed_ms", start.elapsed_time(end))
print("peak_memory_mb", torch.cuda.max_memory_allocated() / 1024**2)
```

### 6.2 使用 torch.profiler 定位 kernel

```python
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
) as prof:
    sim.time_integration(n_time_steps=20)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
```

重点观察：

- `aten::index`
- `aten::matmul` / `aten::mm`
- `aten::mul` / `aten::add`
- `aten::nonzero`
- `cudaMalloc` 或 memory allocation 相关事件

### 6.3 正确计时 CUDA

不要只用 `time.time()` 包围 CUDA 代码，因为 CUDA kernel 异步执行。性能计时前后必须使用 CUDA Event 或 `torch.cuda.synchronize()`。

## 7. 推荐实施顺序

1. 建立 `profile_scenario1.py` 基准，记录当前 `float64` 性能和结果。
2. 引入 dtype 配置，评估 `float32` 是否满足误差要求。
3. 替换 `torch.zeros(...).to(...)` 为直接指定 `device/dtype`。
4. 移除生产路径上的强制 `line_profiler` 依赖。
5. 预分配 `RHS_operator()` 临时 buffer。
6. 优化接收点采样，去掉 `diag(sampleWeight @ P)`。
7. 缓存 flatten view、BC 法向量和索引。
8. 向量化边界条件 pole 更新。
9. 批量化 `grad_3d()`，减少重复小矩阵乘。
10. 评估 `torch.compile` 或 Triton 融合 flux/jump kernel。

## 8. 预期收益

在不改变数值算法的前提下，最可能获得收益的改动是：

- `float64 -> float32`：如果精度允许，通常是最大收益点，同时降低显存占用。
- 预分配临时张量：减少长时间推进时的 allocator 开销和显存碎片。
- 合并/向量化小操作：减少 CUDA kernel launch 数量，对小 mesh 和低阶 `Nx` 尤其明显。
- 优化接收点采样：接收点少时可以减少不必要的矩阵构造。
- 批量化 `grad_3d()`：提升核心 RHS 的矩阵乘利用率。

所有优化都应以 `tests/test_scenario1.py` 和新增 profiling 基准共同验收：先保证数值结果一致或误差可解释，再比较 CUDA 时间和显存峰值。