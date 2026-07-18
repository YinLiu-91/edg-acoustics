# 2D 声学 PML 算法设计与实现说明

本文记录当前仓库中二维声学 PML, perfectly matched layer, 的算法设计、方程形式、代码实现和测试覆盖。实现目标是用 PML 外圈吸收层替代 `examples/dgfem_acoustic_2d/porous_absorber_time_domain/main.py` 中原来的 sponge 阻尼层，同时不破坏已有 extended-reaction, ER, 多孔材料方程、packed RHS、Triton deep fused RHS、partitioned ER RHS、GEMM 后端和 CUDA Graph 路径。

## 1. 设计目标

PML 的工程目标有四点：

1. 外圈吸收传播到计算域边界的声波，替代原来的经验型 sponge 阻尼。
2. PML 只作用在外圈空气吸收层，不进入多孔材料 ER ADE 区域。
3. 旧 sponge 参数和调用方式继续可用，便于已有实验结果复现。
4. 所有当前 2D ER 快路径数值一致，包括 reference、packed、Triton deep fused、compact partitioned ER、GEMM backend 和 CUDA Graph replay。

因此实现采用“先装配原始 DG 声学/ER RHS，再追加 PML split-field 修正”的方式。这样 PML 可以作为外层增强项接在现有求解器之后，避免侵入 ER 多孔材料体方程和材料界面通量。

## 2. 求解区域和状态变量

porous absorber 算例中三角形单元物理标签为：

| 区域 | 物理标签 | 方程 | 额外状态 |
| --- | ---: | --- | --- |
| Air | `1` | 标准线性声学方程 | 无 |
| Porous | `2` | ER 多孔材料方程 | `z_beta`, `z_rho_x`, `z_rho_y` |
| PML | `3` | 标准声学方程加 PML split-field 记忆项 | `pml_psi` |

主变量仍沿用二维声学求解器布局：

$$
Q=(p,v_x,v_y,v_z).
$$

二维算例中 `v_z` 保持为零，PML 只作用前三个主变量：

$$
q=(p,v_x,v_y).
$$

PML 额外状态为：

$$
\Psi=(\psi_x,\psi_y).
$$

代码中的张量布局为：

| 张量 | 形状 | 含义 |
| --- | --- | --- |
| `Q` / packed state view | `[Np, 4, K]` | 每个节点、变量、单元的主状态 |
| `pml_sigma` | `[2, Np, K]` | 方向阻尼 `sigma_x`, `sigma_y` |
| `pml_psi` | `[Np, 2, K]` | PML 记忆状态 `psi_x`, `psi_y` |

其中 `Np` 是单元内节点数，`K` 是三角形单元数。

## 3. PML 连续方程

现有实现使用参考仓库中的二维一阶 split-field PML 形式。基础 DG RHS 先计算空气/ER 方程的半离散右端：

$$
\frac{\partial p}{\partial t},
\qquad
\frac{\partial v_x}{\partial t},
\qquad
\frac{\partial v_y}{\partial t}.
$$

随后在 PML 单元追加：

$$
\frac{\partial p}{\partial t}
\leftarrow
\frac{\partial p}{\partial t}
-(\sigma_x+\sigma_y)p,
$$

$$
\frac{\partial v_x}{\partial t}
\leftarrow
\frac{\partial v_x}{\partial t}
-\frac{\psi_x}{\rho_0c_0^2},
\qquad
\frac{\partial v_y}{\partial t}
\leftarrow
\frac{\partial v_y}{\partial t}
-\frac{\psi_y}{\rho_0c_0^2}.
$$

PML 记忆状态右端使用已经被 PML 修正后的速度 RHS：

$$
\frac{\partial \psi_x}{\partial t}
=
\rho_0c_0^2(\sigma_x-\sigma_y)
\frac{\partial v_x}{\partial t}
-\sigma_y\psi_x,
$$

$$
\frac{\partial \psi_y}{\partial t}
=
\rho_0c_0^2(\sigma_y-\sigma_x)
\frac{\partial v_y}{\partial t}
-\sigma_x\psi_y.
$$

这个顺序在代码里是显式保证的：先原地修正 `rhs_vx` 和 `rhs_vy`，再用修正后的 RHS 计算 `psi_rhs`。

## 4. 阻尼剖面

PML 阻尼由 `PMLRegion` 和 `PMLDamping` 两个组件共同生成。

`PMLRegion` 负责选出 PML 单元，支持两种模式：

| 模式 | 选择逻辑 | 典型用途 |
| --- | --- | --- |
| `centered` | 任一坐标轴满足 `abs(x_i) > cpml_i` | 中心对称计算域 |
| `ground` | `x` 方向两侧吸收，`y` 方向只吸收顶部 | 有刚性地面的二维房间/吸声层算例 |

porous absorber 使用：

```python
PMLRegion(
    (PHYSICAL_XMAX, PHYSICAL_YMAX),
    mode="ground",
    region_name="PML",
)
```

这里 `region_name="PML"` 会优先使用网格里的物理区域标签，保证 PML 掩码和 `.msh` 物理区域完全一致。构造求解器时还会检查：

```python
region_mask == self._pml_mask
```

如果用户传入的 PML 区域和网格物理标签不一致，直接报错，避免吸收层位置静默偏移。

`PMLDamping` 支持的 profile 为：

| profile | 归一化函数 |
| --- | --- |
| `constant` | `1` |
| `linear` | `s` |
| `quadratic` | `s^2` |
| `cubic` | `s^3` |
| `sine-linear` | `s - sin(2*pi*s)/(2*pi)` |

其中

$$
s=\operatorname{clamp}\left(\frac{d}{\Delta},0,1\right),
$$

`d` 是节点进入 PML 的距离，`\Delta` 是该方向 PML 厚度。默认参数为：

```python
PMLDamping(amp_sigma=1000.0, profile="quadratic")
```

默认 scale 规则沿用参考实现语义：对多项式 profile 使用 `profile_id + 1`。因此 porous absorber 中 PML 厚度为 `0.3 m` 时，外边界最大阻尼约为：

```text
1000 * 3 / 0.3 = 10000 1/s
```

## 5. 代码结构

公共 PML 组件位于：

```text
edg_acoustics/pml.py
```

并从 package 根导出：

```python
from edg_acoustics import PMLRegion, PMLDamping, PMLAugmentation
```

核心类职责如下：

| 类 | 职责 |
| --- | --- |
| `PMLRegion` | 基于坐标或 mesh physical region 生成 element mask |
| `PMLDamping` | 生成节点级方向阻尼 `sigma_x/sigma_y` |
| `PMLAugmentation` | 把 PML RHS 修正追加到已经装配好的声学 RHS，并计算 `psi_rhs` |

ER 求解器接入点位于：

```text
edg_acoustics/acoustics_simulation_2d_er.py
```

构造参数新增：

```python
absorbing_layer: str = "sponge"
pml_region: PMLRegion | None = None
pml_damping: PMLDamping | None = None
```

默认仍是 `sponge`，这是为了保持 `ExtendedReactionSimulation2D` 的源码级兼容。porous absorber 示例显式传入 `absorbing_layer="pml"`，所以该算例默认行为已经切换到 PML。

## 6. 与 ER 多孔材料方程的关系

ER 多孔材料已有三组 ADE 状态：

$$
z_\beta,\quad z_{\rho,x},\quad z_{\rho,y}.
$$

它们只在 `Porous` 单元中推进。代码通过 `_porous_mask_2d` 对材料状态输入加掩码：

```python
masked_P = P * self._porous_mask_2d
masked_Vx = Vx * self._porous_mask_2d
masked_Vy = Vy * self._porous_mask_2d
```

PML 不改变这部分逻辑。PML 的 `pml_psi` 是吸收层状态，只在 `PML` 单元中有非零阻尼；在 Air/Porous 单元中 `sigma_x=sigma_y=0`，因此 PML 修正为零。

这个设计带来的结果是：

- Porous 单元继续只求 ER 方程和 ER ADE。
- PML 单元求空气方程加 PML 记忆项。
- Air-Porous 界面通量仍由 `MaterialUpwindFlux2D` 处理。
- PML 不引入新的界面压力或界面速度变量。

## 7. RHS 接入顺序

PML 必须在完整 DG RHS 装配后追加，因为 PML 记忆状态方程需要使用最终速度 RHS。

reference 路径顺序为：

1. 计算单元内部梯度。
2. 计算材料 upwind flux。
3. 用 Air 参数形成基础 RHS。
4. 用 ER 本构覆盖 Porous 单元 RHS。
5. 计算 ER ADE RHS。
6. lift 面通量并加入 RHS。
7. 如果是 sponge 模式，追加旧 `-sigma*q` 阻尼。
8. 如果是 PML 模式，追加 PML 修正并计算 `pml_psi` RHS。

packed 非 deep 路径也保持同样顺序：

```python
self._compute_volume_rhs_packed(q_by_node, RHS_Q_view)
RHS_Q.add_(self._surface_by_node)
self._apply_pml_rhs_torch(self._state_view(q_by_node), RHS_Q_view)
```

deep fused 路径中，基础 RHS 和 ER ADE 已在 Triton kernel 内计算和累加。PML 作为独立 post-kernel 追加：

```python
self._apply_pml_rhs_triton(
    q_by_node,
    RHS_Q,
    q_accumulate,
    accumulate_coefficient,
)
```

## 8. Taylor 时间积分中的辅助状态

时间积分器 `TSI_TI` 对主变量和辅助状态都要做 Taylor 级数累加。ER 求解器通过 `_aux_state_names` 注册辅助状态：

```python
self._aux_state_names = ["z_beta", "z_rho_x", "z_rho_y"]
if self.absorbing_layer == "pml":
    self._aux_state_names.append("pml_psi")
```

非 compact 路径中，`pml_psi` 和 ER ADE 一样进入：

- `_taylor_aux_work`
- `_taylor_aux_rhs`
- `_accumulate_taylor_auxiliary_state()`

compact partitioned ER 路径中，ER ADE 被压缩到 Porous 单元子集：

```text
_compact_z_beta
_compact_z_rho_x
_compact_z_rho_y
```

PML 状态不能压缩到 Porous 子集，因为它属于外圈 PML 单元。因此 compact 路径保留完整：

```text
pml_psi
_pml_psi_work
```

这也是为什么 `_active_auxiliary_state("pml_psi")` 会优先返回 `_pml_psi_work`。Taylor 当前阶使用 work 状态，累加结果写回真实 `pml_psi`。

## 9. Triton 和 GEMM 快路径

Triton deep fused ER kernel 原本负责：

- volume derivative
- surface lift
- Air/Porous RHS
- sponge 阻尼
- ER memory collapse
- ER ADE work/update
- Taylor 主状态累加

PML 实现没有把 `psi` 逻辑塞回这个大 kernel，而是新增一个独立 post-kernel：

```text
launch_pml_auxiliary_rhs_2d()
```

对应文件：

```text
edg_acoustics/acoustics_2d_triton.py
```

这样做有两个原因：

1. PML 是外圈吸收层状态，和 Porous ER ADE 是不同物理区域，分开更容易验证。
2. GEMM backend 已经把 derivative/lift 拆到 `torch.mm` 和 post-kernel，PML 独立 post-kernel 可以同时复用在 legacy deep 和 GEMM deep 路径。

需要注意一个累加细节：deep fused 基础 kernel 已经把 base RHS 累加进 `q_accumulate`。因此 PML post-kernel 不能再次累加完整 RHS，只能累加 PML 修正量：

```text
q_accumulate += coefficient * pml_correction
pml_psi += coefficient * psi_rhs
```

同时它会把 `rhs_by_node` 中的主变量 RHS 覆盖为包含 PML 后的最终 RHS，便于下一阶 Taylor derivative 使用。

## 10. porous absorber 示例配置

示例入口位于：

```text
examples/dgfem_acoustic_2d/porous_absorber_time_domain/main.py
```

当前默认参数为：

```python
DEFAULT_ABSORBING_LAYER = "pml"
DEFAULT_PML_AMP_SIGMA = 1000.0
DEFAULT_PML_PROFILE = "quadratic"
DOMAIN_LABELS = {"Air": 1, "Porous": 2, "PML": 3}
```

构建仿真时：

```python
pml_region = edg_acoustics.PMLRegion(
    (PHYSICAL_XMAX, PHYSICAL_YMAX),
    mode="ground",
    region_name="PML",
)
pml_damping = edg_acoustics.PMLDamping(
    amp_sigma=pml_amp_sigma,
    profile=pml_profile,
)
```

命令行参数为：

```bash
rtk python main.py \
  --thickness both \
  --absorbing-layer pml \
  --pml-amp-sigma 1000 \
  --pml-profile quadratic
```

旧 sponge 模式仍可显式启用：

```bash
rtk python main.py \
  --thickness both \
  --absorbing-layer sponge \
  --sponge-sigma-max 2500
```

## 11. 网格和物理标签

`.geo` 和提交内 `.msh` 文件使用：

```text
Physical Surface("Air", 1)
Physical Surface("Porous", 2)
Physical Surface("PML", 3)
```

这里只改变物理区域名称，物理编号仍保持 `3`。这样已有单元拓扑、区域划分和历史 mesh label 判断仍然兼容。

`validate_mesh()` 校验三角单元物理编号集合：

```python
EXPECTED_TRIANGLE_LABELS = {1, 2, 3}
```

PML 区域名称由 `Mesh2D` 的 `domain_elements["PML"]` 提供给 `PMLRegion`。

## 12. 结果元数据

结果文件中新增或保留以下 metadata：

| 字段 | 含义 |
| --- | --- |
| `absorbing_layer` | 实际吸收层模式，`pml` 或 `sponge` |
| `pml_amp_sigma` | PML 输入振幅 |
| `pml_profile` | PML profile |
| `pml_sigma_max` | 网格节点上的最大实际 PML 阻尼 |
| `sponge_sigma_max` | 旧 sponge 模式阻尼参数 |
| `sponge_thickness` | 旧 sponge 模式厚度；PML 模式下也保留为兼容字段 |

## 13. 测试覆盖

新增和更新的测试覆盖以下层级：

| 测试 | 覆盖内容 |
| --- | --- |
| `tests/test_pml.py` | PML region mask、阻尼 profile、split-field 方程、错误输入 |
| `test_porous_absorber_default_uses_pml_layer_cpu` | porous absorber 默认 PML 构造、sigma mask、`pml_psi` 状态 |
| `test_porous_absorber_explicit_sponge_layer_cpu` | 显式 sponge 兼容模式 |
| packed/reference parity | CPU reference 和 packed RHS 的 PML 一致性 |
| Triton/deep/partitioned/GEMM parity | CUDA fast path 的 `Q`、receiver、ER ADE、`pml_psi` 一致性 |
| CUDA Graph parity | eager 和 graph replay 的 PML 状态一致性 |
| COMSOL regression | `5 cm` 和 `15 cm` 默认 PML 算例仍满足 golden 误差阈值 |

关键测试命令：

```bash
rtk /home/liu/anaconda3/envs/torch_210/bin/python -m pytest \
  tests/test_pml.py \
  tests/test_porous_absorber_2d.py -q

rtk /home/liu/anaconda3/envs/torch_210/bin/python -m pytest \
  tests/test_acoustics_2d.py -q
```

当前验证结果：

```text
tests/test_pml.py tests/test_porous_absorber_2d.py: 23 passed
tests/test_acoustics_2d.py: 14 passed
```

## 14. 维护注意事项

后续修改 PML 时需要重点检查以下约束：

1. PML 修正必须在 base RHS 和 surface lift 之后追加。
2. `psi_rhs` 必须使用已经加入 PML 速度修正后的 `rhs_vx/rhs_vy`。
3. deep fused 路径中 `q_accumulate` 只能追加 PML correction，不能重复累加 base RHS。
4. compact partitioned ER 只压缩 Porous ADE，不压缩 `pml_psi`。
5. `PMLRegion(region_name="PML")` 应继续和 mesh physical label 做一致性检查。
6. 如果未来把 PML 扩展到 3D，需要新增 `psi_z` 和 `sigma_z`，不能复用当前二维 `PMLAugmentation` 的形状假设。

## 15. 文件索引

| 文件 | 作用 |
| --- | --- |
| `edg_acoustics/pml.py` | PML 公共组件 |
| `edg_acoustics/__init__.py` | 导出 PML API |
| `edg_acoustics/acoustics_simulation_2d_er.py` | ER 求解器 PML 接入 |
| `edg_acoustics/acoustics_2d_triton.py` | Triton PML post-kernel |
| `examples/dgfem_acoustic_2d/porous_absorber_time_domain/main.py` | 默认 PML 示例入口 |
| `examples/dgfem_acoustic_2d/porous_absorber_time_domain/porous_absorber_time_domain.geo` | PML 物理区域定义 |
| `tests/test_pml.py` | PML 单元测试 |
| `tests/test_porous_absorber_2d.py` | porous absorber PML 集成测试 |
