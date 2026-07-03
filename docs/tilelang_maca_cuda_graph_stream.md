# TileLang MetaX CUDA Graph Stream 问题说明

本文档记录 TileLang lift kernel 在 MetaX/MACA 上最初无法进入 CUDA
Graph 的原因、定位过程、修复方案，以及为什么上游 TVM FFI adapter
里原本的 stream 代码被注释掉。

EDG 中涉及的 TileLang kernel 是 fp64 lift GEMM：

```text
surface_by_node[35, N] = lift[35, 60] @ flux_by_face[60, N]
```

当前使用的 TileLang 配置为：

```text
bm48_bn64_bk16_s0_t256_fullcol
```

## 结论

问题不在 TileLang kernel 的数学结果。TileLang eager 执行一直是正确的。
问题出在 PyTorch CUDA Graph capture stream 到 TVM FFI MACA launcher 的
stream 传递链路。

修复前，TVM FFI adapter 没有把 PyTorch 当前 stream 写入到 MACA launcher
实际读取的 stream key 上。MACA launcher 启动 kernel 时读取的是：

```cpp
TVMFFIEnvGetStream(kDLMACA, device_id)
```

但 PyTorch 暴露的是 CUDA stream。只调用 `tvm_ffi.use_torch_stream()` 时，
stream 可能只被写入 `kDLCUDA` 对应的上下文；而 MACA launcher 用
`kDLMACA` 读取，导致 launcher 拿不到 CUDA Graph capture stream。

结果就是：TileLang kernel 在 capture 期间确实执行了，但没有被记录进
CUDA Graph。PyTorch 会提示 empty graph，graph replay 时不会重新执行
TileLang kernel。

最终修复方案是在调用 TVM executable 前同时设置两套 stream context：

```text
1. tvm_ffi.use_torch_stream()
2. tvm_ffi.use_raw_stream(tvm_ffi.Device(kDLMACA, device_id), raw_stream)
```

这样既保留正常 CUDA/PyTorch stream 语义，又确保 MACA launcher 的
`TVMFFIEnvGetStream(kDLMACA, device_id)` 能读到同一个 capture stream。

## 初始现象

使用独立 probe 测试：

```bash
python benchmarks/tilelang_lift_graph_probe.py --n 4096
```

修复前关键失败输出为：

```text
probe=tilelang_default_stream
UserWarning: The CUDA Graph is empty
tilelang_default_stream_graph_capture_error=none
tilelang_default_stream_graph_replay_error=none
tilelang_default_stream_graph_after_replay_ok=0
tilelang_default_stream_graph_captured_and_replayed=0
```

同时 TileLang eager 是正确的：

```text
tilelang_eager_ok=1
tilelang_eager_max_abs=0.000000e+00
```

PyTorch 自带 `torch.mm` 的 CUDA Graph replay 也是正确的：

```text
torch_mm_graph_replay_error=none
torch_mm_graph_after_replay_ok=1
```

因此可以排除两个方向：

```text
不是 TileLang kernel 数值错误。
不是 MetaX PyTorch CUDA Graph 整体不可用。
```

问题被定位到 TileLang/TVM FFI launcher 的 stream 传播。

## EDG 框架中的表现

修复前，在 EDG full timestep CUDA Graph 中启用 TileLang lift 时，graph
校验失败。EDG 检测到 replay 后状态不匹配，因此禁用 TileLang 并回退：

```text
tilelang_lift_enabled=0
tilelang_lift_graph_capture_supported=0
tilelang_lift_fallback_reason=cuda graph replay validation failed; falling back from TileLang lift
```

为了在修复 TileLang adapter 前保留性能收益，EDG 临时加入了 segmented
CUDA Graph fallback：

```text
pre graph -> TileLang eager lift -> post graph
```

这个路径可以让 timestep 中除 TileLang lift 外的其他 kernel 继续使用
CUDA Graph replay，而 TileLang lift 单独 eager 执行。它是正确的，并且
比完全禁用 TileLang 快，但它不是完整 full graph capture。

## 根因分析

### CUDA Graph 只捕获当前 capture stream 上的工作

PyTorch CUDA Graph capture 会记录当前 capture stream 上提交的 CUDA work。
如果自定义 launcher 把 kernel 发射到其他 stream 上，那么 capture 过程
可能不报错，但 graph 内部是空的，或者缺少该 kernel。

本问题中正是这种情况：

```text
CUDA Graph is empty
```

capture 期间 TileLang kernel 实际执行过，所以某些检查中 capture 后的
output 可能是正确的。但 graph replay 时 kernel 没有被重放，output
保持旧值，因此 replay correctness 失败。

### TVM FFI adapter 原本没有设置可被 MACA 读取的 stream

相关文件：

```text
/app/tilelang-metax/tilelang/jit/adapter/tvm_ffi.py
```

上游 adapter 中有 stream 相关注释，但真正的 stream functor 被注释掉：

```python
# Capture thunks that reflect Torch's current stream and device.
# These are evaluated at call time to align TVM execution with the
# caller's active PyTorch stream/device.
# current_stream_functor = self.get_current_stream_functor()
current_device_functor = self.get_current_device_functor()
```

而 executable 调用点原本只是：

```python
executable(*tensor_list)
```

这意味着 TVM FFI 执行时并没有可靠安装 PyTorch 当前 capture stream。

### MACA launcher 读取的是 `kDLMACA`

`tilelang-metax` 的 MACA runtime launcher 中，kernel launch stream 来自：

```cpp
mcStream_t strm =
    static_cast<mcStream_t>(TVMFFIEnvGetStream(kDLMACA, device_id));
```

然后该 stream 被传给 MACA kernel launch：

```cpp
mcModuleLaunchKernel(..., strm, ...);
```

所以对 MetaX/MACA 来说，关键不是仅仅“设置了一个 CUDA stream”，而是
必须保证 `TVMFFIEnvGetStream(kDLMACA, device_id)` 能读到 PyTorch 当前
capture stream。

### 只使用 `use_torch_stream()` 仍然不够

第一版修复尝试是在 adapter 中加入：

```python
with tvm_ffi.use_torch_stream():
    executable(*tensor_list)
```

但 standalone probe 仍然显示：

```text
CUDA Graph is empty
tilelang_default_stream_graph_after_replay_ok=0
```

说明仅写入 PyTorch/CUDA stream context，在当前 MetaX TVM FFI 环境下没有
被 MACA launcher 以 `kDLMACA` 读取到。

### CMake 中的 CUDA 到 MACA remap 没有覆盖实际路径

`tilelang-metax/cmake/load_tvm.cmake` 里尝试 patch TVM FFI EnvContext：

```cmake
if (device_type == DLDeviceType::kDLCUDA) {
  device_type = (int32_t)DLDeviceType::kDLMACA;
}
```

理论上，这应该让 CUDA stream set/get 映射到 MACA。但实际测试中，仅使用
`use_torch_stream()` 后 graph 仍然为空。因此在当前安装环境中，依赖这层
remap 不够稳。

最终采用更直接的方案：在 Python adapter 中显式把同一个 raw stream 写入
`kDLMACA`。

## 为什么原作者注释掉了 stream 代码

这里需要区分“可证据化事实”和“合理推断”。

### 可证据化事实

通过 git blame：

```bash
git -C /media/liu/research/linux/tilelang-metax blame -L 169,174 -- tilelang/jit/adapter/tvm_ffi.py
```

可以看到这段代码来自：

```text
74da3696 [FFI] Use tvm ffi as the default execution backend (#1259)
```

这个 commit 在引入 `tilelang/jit/adapter/tvm_ffi.py` 时，就已经包含：

```text
# current_stream_functor = self.get_current_stream_functor()
current_device_functor = self.get_current_device_functor()
```

并且 executable 调用点是：

```python
executable(*tensor_list)
```

因此，这不是 EDG 后续修改导致的注释；该 stream 行在 TVM FFI adapter
初始引入时就已经被注释。

### 合理推断

commit message 没有明确说明作者为什么注释这行，所以不能断言作者的真实
动机。结合代码结构，更合理的解释是：

```text
作者希望 adapter 跟随 PyTorch stream。
文件头和注释明确写了要对齐当前 PyTorch stream/device。
```

但在迁移到 TVM FFI backend 时，stream 处理可能被认为应该由 TVM FFI
EnvContext 管理，于是旧的 raw stream functor 被保留为注释，没有真正接入
launch path。

这个问题不容易在普通 eager 测试中暴露，因为 eager kernel 数值可以正确。
只有在 CUDA Graph replay 或严格 multi-stream 测试中，才会看到 kernel
没有被记录进 graph。

因此，不应该把这理解为作者“有意禁用 CUDA Graph”或“有意禁用 stream”。
更准确的说法是：TVM FFI adapter 迁移时保留了 stream 对齐的设计意图，但
实现没有完成到 MACA CUDA Graph 所需的程度，测试也没有覆盖这个场景。

## 修复方案

修复文件：

```text
/app/tilelang-metax/tilelang/jit/adapter/tvm_ffi.py
```

源码仓库中也应保持同样修改：

```text
/media/liu/research/linux/tilelang-metax/tilelang/jit/adapter/tvm_ffi.py
```

### 补充 import

```python
import contextlib
```

### 修改 launch path

在调用 TVM executable 前检测 CUDA tensor，并安装 stream context：

```python
executable = get_executable()
has_cuda_tensor = any(
    isinstance(tensor, torch.Tensor) and tensor.device.type == "cuda"
    for tensor in tensor_list
)
if torch.cuda.is_available() and has_cuda_tensor:
    import tvm_ffi

    current_stream = torch.cuda.current_stream()
    stream_device = current_stream.device
    device_index = stream_device.index
    if device_index is None:
        device_index = torch.cuda.current_device()

    with contextlib.ExitStack() as stream_stack:
        stream_stack.enter_context(tvm_ffi.use_torch_stream())
        try:
            # MACA launchers read kDLMACA, while torch exposes a CUDA stream.
            maca_device_type = getattr(
                getattr(tvm_ffi, "DLDeviceType", object),
                "kDLMACA",
                19,
            )
            maca_device = tvm_ffi.Device(maca_device_type, device_index)
            stream_stack.enter_context(
                tvm_ffi.use_raw_stream(
                    maca_device,
                    current_stream.cuda_stream,
                )
            )
        except Exception:
            pass
        executable(*tensor_list)
else:
    executable(*tensor_list)
```

### 为什么要双写 stream

`tvm_ffi.use_torch_stream()` 是通用路径，用于跟随 PyTorch 当前 stream。
它对普通 CUDA 后端和 PyTorch 语义是正确的。

显式 `tvm_ffi.Device(19, device_id)` 是 MetaX/MACA 特定补充。在当前环境
中，`19` 对应 DLPack device type `kDLMACA`。把同一个 raw stream 写到
这个 device type 下，MACA launcher 才能通过：

```cpp
TVMFFIEnvGetStream(kDLMACA, device_id)
```

读取到 graph capture stream。

`try/except` 是为了保持可移植性：

```text
在 MetaX/MACA 上，Device(19, device_id) + use_raw_stream 正常生效。
在其他后端上，如果 MACA device type 不存在，则退回 use_torch_stream。
```

这是 Python 侧修复。只要实际 import 的 Python 文件更新，就不需要重编
C++ runtime。

## 验证方法

### 1. 确认实际 import 的 adapter 已更新

在 MetaX 容器中运行：

```bash
python - <<'PY'
import importlib
import inspect

m = importlib.import_module("tilelang.jit.adapter.tvm_ffi")
src = inspect.getsource(m.TVMFFIKernelAdapter._convert_torch_func)
print("file=", m.__file__)
print("has_use_torch_stream=", "use_torch_stream" in src)
print("has_use_raw_stream=", "use_raw_stream" in src)
print("has_Device_19=", "19" in src)
print("has_contextlib=", "contextlib.ExitStack" in src)
PY
```

期望输出：

```text
file= /app/tilelang-metax/tilelang/jit/adapter/tvm_ffi.py
has_use_torch_stream= True
has_use_raw_stream= True
has_Device_19= True
has_contextlib= True
```

如果函数体里有 `contextlib.ExitStack`，但文件顶部没有 `import contextlib`，
probe 会失败：

```text
NameError: name 'contextlib' is not defined
```

### 2. 运行独立 probe

```bash
python benchmarks/tilelang_lift_graph_probe.py --n 4096
```

修复成功后的关键输出：

```text
probe=tilelang_default_stream
tilelang_default_stream_graph_capture_error=none
tilelang_default_stream_graph_replay_error=none
tilelang_default_stream_graph_after_replay_ok=1
tilelang_default_stream_graph_after_replay_max_abs=0.000000e+00
```

`tilelang_explicit_current_stream` 仍可能失败：

```text
tilelang_explicit_current_stream_graph_capture_error=TypeError: ... unexpected keyword argument 'stream'
```

这是预期行为。当前 TVM FFI adapter 不支持 `stream=` 关键字参数。我们修复
的是 current stream context 路径，不是新增 `stream=` API。

当前 probe 中还有一个字段：

```text
tilelang_default_stream_graph_captured_and_replayed=0
```

这个字段目前偏保守，容易误导。它同时要求 `after_capture_ok=1`。但 PyTorch
自己的 `torch.mm` probe 也会出现：

```text
torch_mm_graph_after_capture_ok=0
torch_mm_graph_after_replay_ok=1
```

判断 CUDA Graph 是否可用，应重点看 replay 后是否正确：

```text
*_graph_after_replay_ok=1
```

### 3. 运行完整 EDG benchmark

不要强制 segmented graph，直接测 full graph：

```bash
python benchmarks/scenario1_benchmark.py \
  --device cuda \
  --mesh-name scenario1_profile_lc0p20.msh \
  --steps 50 \
  --cuda-graph \
  --no-record-receivers \
  --enable-tilelang-lift \
  --log-cuda-graph-selection
```

期望 graph 选择日志：

```text
cuda_graph_selection=begin chunk_steps=1 record_receivers=0 tilelang_lift=1 segmented_mode=auto
cuda_graph_selection=full_try
cuda_graph_selection=selected mode=full
cuda_graph_selection=cache_hit mode=full
```

期望 metadata：

```text
cuda_graph_mode=full
tilelang_lift_enabled=1
tilelang_lift_graph_capture_supported=1
tilelang_lift_segmented_graph_supported=0
tilelang_lift_fallback_reason=none
```

实际在 MetaX C500、`scenario1_profile_lc0p20.msh`、50 steps 上观察到：

```text
No TileLang, full graph:          8.7386 ms/step
TileLang, segmented graph:        7.4721 ms/step
TileLang, full graph after fix:   7.3772 ms/step
```

因此最终推荐路径是 full CUDA Graph + TileLang lift。segmented graph 保留为
fallback。

## EDG 中的 fallback 策略

默认：

```text
EDG_ACOUSTICS_TILELANG_SEGMENTED_CUDA_GRAPH=auto
```

选择顺序：

```text
1. 先尝试 full CUDA Graph。
2. 如果 full graph 校验失败，且 TileLang lift 启用，则尝试 segmented graph。
3. 如果 segmented graph 也失败，则禁用 TileLang lift，并回退到无 TileLang 的 full graph。
```

开启 graph 选择日志：

```bash
export EDG_ACOUSTICS_CUDA_GRAPH_SELECTION_LOG=1
```

或使用 benchmark 参数：

```bash
--log-cuda-graph-selection
```

强制使用 segmented graph：

```bash
--enable-tilelang-segmented-graph
```

禁用 segmented fallback：

```bash
--disable-tilelang-segmented-graph
```

## 排查清单

如果后续重新遇到 TileLang 不能进入 CUDA Graph，可以按下面顺序检查：

```text
1. TileLang eager 是否正确。
2. PyTorch torch.mm CUDA Graph replay 是否正确。
3. tilelang_default_stream_graph_after_replay_ok 是否为 1。
4. 实际 import 的 tvm_ffi.py 是否包含 use_torch_stream、use_raw_stream、Device(19)。
5. 文件顶部是否有 import contextlib。
6. EDG benchmark 是否输出 cuda_graph_mode=full。
7. 如果 standalone probe 通过但 EDG 不通过，打开 --log-cuda-graph-selection。
8. 如果 Python patch 存在但仍然 empty graph，再检查 C++ 层 TVMFFIEnvGetStream(kDLMACA, device_id)。
```

常见非致命现象：

```text
skip_tensor_validation TypeError:
  该 TVM FFI adapter 不支持这个 kwarg，EDG/probe 会自动去掉后重试。

explicit_current_stream TypeError:
  adapter 不支持 stream= kwarg；当前支持的是 current stream context 路径。

captured_and_replayed=0 但 after_replay_ok=1:
  当前 probe 字段判断过严，应以后者为准。
```

## 长期改进建议

当前 Python patch 是最小可用修复。长期可以在 `tilelang-metax` 中进一步清理：

```text
1. 在 tvm_ffi.DLDeviceType 中正式暴露 kDLMACA。
2. 支持 tvm_ffi.device("maca:0")。
3. 给 TVM FFI adapter 添加 MetaX/MACA CUDA Graph replay 回归测试。
4. 添加严格 multi-stream 测试，确认 kernel 使用当前 PyTorch stream。
5. 让 CMake 中 CUDA 到 MACA 的 EnvContext remap 可在运行时验证。
6. 可选地支持 adapter 的 stream= kwarg，但 current stream context 已足够支持 CUDA Graph。
```
