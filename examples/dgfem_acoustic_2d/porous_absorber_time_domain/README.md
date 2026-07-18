# 2D ER 多孔吸声层时域算例

这个目录复现 COMSOL `porous_absorber_time_domain` 教程里的二维扩展反应 `ER, extended reaction` 算例，只复现 ER 方法，不包含文档中的 `LR, local reaction` 阻抗边界近似。

算例的物理图像是一个二维空气域，下边界有一层刚性背衬的多孔吸声层，厚度取 `5 cm` 或 `15 cm`。声源和接收点都在空气域中，声源不是连续激励，而是一个初始压力脉冲。多孔材料不是用单一常数参数，而是用随频率变化的等熵压缩率 `beta` 和密度 `rho`，先做向量拟合，再在时域里通过辅助状态 `ADE` 演化。

当前实现基于本仓库的 2D DG 求解器，时间推进使用和 3D 算例一致的 `TSI_TI`。

## 1. 目录内容

- `main.py`
  这个 case 的主入口。负责生成拟合文件、生成网格、构建仿真、执行时域推进并写出结果。
- `porous_absorber_time_domain.geo`
  Gmsh 几何文件。参数 `w0` 控制多孔层厚度，默认 `0.05`。几何里包含空气域、多孔层和外圈 sponge 吸收层。
- `fit_er_material.m`
  用 Octave 调 `vectfit3`，把频率相关的压缩率和密度表拟合成实状态空间模型，输出 `er_material_fit.mat`。
- `er_material_fit.mat`
  向量拟合结果。包含 `A_beta/B_beta/C_beta/D_beta` 和 `A_rho/B_rho/C_rho/D_rho`，会被 Python 侧直接读入。
- `porous_absorber_time_domain_compressibility.zh_CN.txt`
  多孔材料等熵压缩率表。
- `porous_absorber_time_domain_density.zh_CN.txt`
  多孔材料等效密度表。
- `porous_absorber_time_domain_admittance.zh_CN.txt`
  COMSOL 教程里的导纳表。当前 ER 复现不直接用它，它主要对应文档里的 LR 方法。
- `5cm_er_comsol_golden.txt`
  COMSOL 导出的 `5 cm` 接收点总声压 golden 数据。
- `15cm_er_comsol_golden.txt`
  COMSOL 导出的 `15 cm` 接收点总声压 golden 数据。
- `plot_results.py`
  读取 `results_on_the_run.mat`，叠加 COMSOL golden，并画出压力、绝对误差和相对误差。
- `outputs/`
  默认输出目录。通常有 `outputs/5cm` 和 `outputs/15cm` 两个子目录。
- `outputs_sponge_1000/`、`outputs_sponge_5000/`
  不同 sponge 强度的对比结果目录。
- `receiver_history_with_error.png`
  当前目录里已经生成的一张示例图，包含压力曲线和与 COMSOL 的误差图。
- `MinerU_markdown_models.aco.porous_absorber_time_domain.zh_CN_2076138063190925312.md`
  COMSOL 教程的 Markdown 转写稿，用来核对几何、参数、物理设置和参考图。
- `MinerU_markdown_Wang_and_Hornikx_-_2023_-_Extended_reacting_boundary_modeling_of_porous_materials_with_thin_coverings_for_time-domain_room_aco.md`
  Wang 和 Hornikx (2023) 论文的 Markdown 转写稿，是本 README 中扩展反应性多孔材料、ADE 状态变量和界面通量说明的理论来源。

## 2. 运行前提

常规运行需要：

- Python 能导入本仓库的 `edg_acoustics`
- `numpy`
- `scipy`
- `matplotlib`
- `meshio`
- `torch`

重新生成网格时还需要：

- `gmsh`

重新生成 ER 向量拟合文件时还需要：

- `octave`
- `examples/material_fit` 目录下的 `vectfit3`

建议在这个目录下运行，并设置 `PYTHONPATH` 指向仓库根目录：

```bash
cd /media/liu/research/linux/edg-muxi/edg-acoustics/examples/dgfem_acoustic_2d/porous_absorber_time_domain
export PYTHONPATH=/media/liu/research/linux/edg-muxi/edg-acoustics
```

## 3. 快速开始

运行默认 `5 cm` 和 `15 cm` 两组 case：

```bash
rtk python main.py \
  --thickness both \
  --save-step 1000 \
  --save-mesh-at-ms 5.5
```

如果要强制重新做材料拟合和重新生成网格：

```bash
rtk python main.py \
  --thickness both \
  --force-fit \
  --force-mesh \
  --save-step 1000 \
  --save-mesh-at-ms 5.5
```

如果只想跑单个厚度：

```bash
rtk python main.py --thickness 0.05
rtk python main.py --thickness 0.15
```

说明：

- `--thickness both` 会依次运行 `5 cm` 和 `15 cm`
- `--thickness 0.05` / `0.15` 用米作为单位
- `--force-fit` 会重新生成 `er_material_fit.mat`
- `--force-mesh` 会重新生成 `.msh`
- 默认启用 packed RHS、2D Triton kernel、deep-fused ER RHS、compact ER 和 CUDA graph，无需额外优化参数

## 4. 重新生成网格

`main.py` 内部就是通过下面这类 Gmsh 命令生成网格：

```bash
rtk gmsh -2 porous_absorber_time_domain.geo -setnumber w0 0.05 -format msh2 -o porous_absorber_time_domain_5cm.msh
rtk gmsh -2 porous_absorber_time_domain.geo -setnumber w0 0.15 -format msh2 -o porous_absorber_time_domain_15cm.msh
```

更常用的做法还是交给 `main.py`：

```bash
rtk python main.py --thickness both --force-mesh
```

当前 `.geo` 使用的网格尺度与文档一致：

- 最大单元尺寸 `lc = 343 / 4000 / 1.5`

## 5. 画图和与 COMSOL 对比

画接收点曲线，并自动叠加本目录下的 COMSOL golden：

```bash
rtk python plot_results.py \
  outputs/5cm/results_on_the_run.mat \
  outputs/15cm/results_on_the_run.mat \
  --output receiver_history_with_error.png
```

如果想手动指定 golden 文件：

```bash
rtk python plot_results.py \
  outputs/5cm/results_on_the_run.mat \
  outputs/15cm/results_on_the_run.mat \
  --golden 5cm_er_comsol_golden.txt 15cm_er_comsol_golden.txt \
  --output receiver_history_with_error.png
```

`plot_results.py` 现在会输出四张子图：

- 接收点全时域压力历史
- `5 ms` 到 `10 ms` 的放大图
- 相对 COMSOL 的绝对误差 `|p - p_ref|`
- 相对 COMSOL 的逐点相对误差

相对误差不是简单的 `|e| / |p_ref|`，而是：

```text
|e| / max(|p_ref|, ratio * max|p_ref|)
```

默认 `ratio = 1e-3`，对应参数：

```bash
--relative-error-floor-ratio 1e-3
```

这样做是为了避免 golden 在过零或尾部接近零时，相对误差被无限放大，图上只剩尖峰。

## 6. 代码主要流程

`main.py` 的主流程可以概括成四步。

### 6.1 材料拟合

函数 `ensure_material_fit()` 会检查 `er_material_fit.mat` 是否存在。如果不存在，或者传了 `--force-fit`，就调用 `fit_er_material.m`：

- 读取 `compressibility` 和 `density` 频响表
- 分别做 8 极点实状态空间向量拟合
- 输出 `beta` 和 `rho` 的 `A/B/C/D` 矩阵

这个 `.mat` 文件随后由 `ExtendedReactionMaterialFit.from_mat()` 读入。

### 6.2 网格生成和校验

函数 `ensure_mesh()` 会根据厚度选择：

- `porous_absorber_time_domain_5cm.msh`
- `porous_absorber_time_domain_15cm.msh`

如果网格不存在，或者传了 `--force-mesh`，就调用 Gmsh 生成。生成后 `validate_mesh()` 会校验：

- 是否同时包含 `triangle` 和 `line`
- 三角单元物理标签是否为 `Air=1, Porous=2, Sponge=3`
- 边界物理标签是否为 `Outer=11, Rigid=12`
- `ymin` 是否对应当前厚度

### 6.3 构建求解器

函数 `build_simulation()` 会做这些事：

- 用 `.msh` 构建 `Mesh2D`
- 读入 ER 拟合材料
- 构建 `ExtendedReactionSimulation2D`
- 设置外边界吸收条件和刚性边界
- 设置初始压力脉冲 `RadialPressurePulse2D_IC`
- 设置接收点
- 用 `TSI_TI` 初始化时间积分器

这个算例里的关键常数是：

- 空气密度 `rho0 = 1.213`
- 声速 `c0 = 343`
- 几何高度 `L0 = 1.5`
- 声源位置 `(-1.0, 0.5)`
- 接收点位置 `(1.0, 0.5)`
- 脉冲宽度 `B = 0.045`
- 默认 DG 阶数 `Nx = 4`
- 默认 TSI 阶数 `Nt = 4`
- 默认 CFL `0.25`
- sponge 厚度 `L0 / 5 = 0.3`

### 6.4 时域推进和结果写出

函数 `run_case()` 会：

- 调 `sim.time_integration()`
- 保存接收点历史 `results_on_the_run.mat`
- 保存末时刻全场 `snapshot.mat`
- 按配置导出中间场 `.msh`

如果传了：

- `--save-step N`
  每隔 `N` 步保存一次 `results_on_the_run.mat`
- `--save-mesh-step N`
  每隔 `N` 步导出一次场 `.msh`
- `--save-mesh-at-ms 5.5`
  在最接近 `5.5 ms` 的时刻额外导出一份 `.msh`

## 7. 文献公式与代码对应

本算例对应 Wang 和 Hornikx (2023) 的 extended reacting porous material time-domain formulation。当前实现包含多孔材料的频率相关反应性，但没有实现论文中的薄覆盖层状态变量；因此覆盖层界面阻抗在本算例中取为 `Z_t=0`，代码实际求解的是无遮盖层的相邻介质界面。

### 7.0 三类区域分别求解什么

几何文件把三角形单元标记为 Air=1、Porous=2 和 Sponge=3。这三个标签不是三个完全独立的 PDE：

| 区域 | 主变量方程 | 额外状态 | 代码中的处理 |
| --- | --- | --- | --- |
| Air | 标准线性声学方程 | 无 | 使用空气的 rho0、c0 |
| Porous | 仍然求解 P/Vx/Vy，但本构关系变为频率相关的 ER 模型 | z_beta、z_rho_x、z_rho_y | 用 beta_D/rho_D 和 A/B/C 状态空间项替换空气右端 |
| Sponge | 标准声学方程加空间阻尼 | 无 | 只在 Sponge 单元中令 sigma>0 |

因此，二维代码的主未知量可以写成

$$
U=(p,v_x,v_y),
$$

多孔单元还要附带材料记忆状态

$$
Z=(z_\beta,z_{\rho,x},z_{\rho,y}).
$$

RHS_operator(P, Vx, Vy, Vz, BCvar) 每次被时间积分器调用时，执行如下顺序：

1. P/Vx/Vy 是所有单元上的主变量；_gradient() 计算单元内部梯度，得到 dPdx、dPdy、dVxdx、dVydy 和 divV。
2. MaterialUpwindFlux2D 根据相邻单元的材料参数计算所有内部面的数值通量。
3. 先用 k_inf 和 inv_rho_inf 形成所有单元的标准声学右端，再用 _porous_mask_2d 只覆盖多孔单元的右端。
4. 用 z_beta、z_rho_x、z_rho_y 计算材料记忆项，并同时计算三个 ADE 状态的右端。
5. 用 Fscale 和 lift 把面通量加入体积分右端，最后在 Sponge 单元中加入 -sigma*field。

这个顺序很重要：材料记忆项属于多孔单元内部的体方程；空气-多孔交界面本身不存储一个单独的“界面压力”或“界面速度”，而是由数值通量弱形式地施加界面条件。

#### 7.0.1 空气区域的方程和代码

空气区域求解的是无耗散、无平均流背景下的线性一阶声学方程：

$$
\frac{\partial \boldsymbol v}{\partial t}
  + \frac{1}{\rho_a}\nabla p = 0,
\qquad
\frac{\partial p}{\partial t}
  + \rho_a c_a^2\nabla\cdot\boldsymbol v = 0.
$$

本算例中空气参数来自 main.py：

$$
\rho_a=\text{RHO0}=1.213,
\qquad
c_a=\text{C0}=343.0,
\qquad
K_a=\rho_a c_a^2.
$$

在 ExtendedReactionSimulation2D._build_material_model() 中，所有单元先默认填入

$$
\beta_a=\frac{1}{\rho_a c_a^2},
\qquad
\rho_a=\text{RHO0},
\qquad
K_a=\frac{1}{\beta_a}.
$$

因此，对空气单元，RHS_operator() 的基础右端就是

$$
\dot p=-K_a
\left(\frac{\partial v_x}{\partial x}
      +\frac{\partial v_y}{\partial y}\right),
\qquad
\dot v_x=-\frac{1}{\rho_a}\frac{\partial p}{\partial x},
\qquad
\dot v_y=-\frac{1}{\rho_a}\frac{\partial p}{\partial y}.
$$

对应代码逻辑为：

```text
dPdx, dPdy = self._gradient(P)
dVxdx, _ = self._gradient(Vx)
_, dVydy = self._gradient(Vy)
divV = dVxdx + dVydy

rhs_P  = -self.k_inf_elements * divV
rhs_Vx = -self.inv_rho_inf_elements * dPdx
rhs_Vy = -self.inv_rho_inf_elements * dPdy
```

_gradient() 不是有限差分，而是利用 DG 基函数导数矩阵 Dr、Ds 和单元几何映射得到物理坐标梯度。P、Vx、Vy 的张量形状为 (Np, N_elements)，所以每个单元内部都有一组独立的 DG 多项式自由度。

#### 7.0.2 多孔材料区域的方程和代码

多孔区域的主变量仍然是压力 p 和速度 v，但材料本构关系不再是常数 rho_a、K_a，而是频率相关的有效密度 rho_eff 和有效压缩率 beta_eff：

$$
i\omega\rho_{\mathrm{eff}}(\omega)\widehat{\boldsymbol v}
  +\nabla\widehat p=0,
\qquad
i\omega\beta_{\mathrm{eff}}(\omega)\widehat p
  +\nabla\cdot\widehat{\boldsymbol v}=0.
$$

fit_er_material.m 先用 vectfit3 将频域数据拟合为

$$
\rho_{\mathrm{eff}}(s)
  =D_\rho+C_\rho(sI-A_\rho)^{-1}B_\rho,
\qquad
\beta_{\mathrm{eff}}(s)
  =D_\beta+C_\beta(sI-A_\beta)^{-1}B_\beta.
$$

代码把 D_beta、D_rho 作为多孔材料主方程的瞬时系数：

$$
\beta_D=D_\beta,
\qquad
\rho_D=D_\rho,
\qquad
K_\infty=\frac{1}{\beta_D},
\qquad
c_\infty=\sqrt{\frac{K_\infty}{\rho_D}}.
$$

对应的多孔区域主方程为

$$
\begin{aligned}
\beta_D\dot p
  &=-\nabla\cdot\boldsymbol v
    -C_\beta A_\beta z_\beta
    -C_\beta B_\beta p,\\
\rho_D\dot v_x
  &=-\frac{\partial p}{\partial x}
    -C_\rho A_\rho z_{\rho,x}
    -C_\rho B_\rho v_x,\\
\rho_D\dot v_y
  &=-\frac{\partial p}{\partial y}
    -C_\rho A_\rho z_{\rho,y}
    -C_\rho B_\rho v_y.
\end{aligned}
$$

三个材料状态满足

$$
\dot z_\beta=A_\beta z_\beta+B_\beta p,
\qquad
\dot z_{\rho,x}=A_\rho z_{\rho,x}+B_\rho v_x,
\qquad
\dot z_{\rho,y}=A_\rho z_{\rho,y}+B_\rho v_y.
$$

代码对应关系如下：

| 数学量 | 代码 |
| --- | --- |
| z_beta、z_rho_x、z_rho_y | _aux_state_names 和对应的状态张量 |
| A_beta*z_beta+B_beta*p | _material_rhs(beta_A, beta_B, z_beta, masked_P) |
| A_rho*z_rho_x+B_rho*Vx | _material_rhs(rho_A, rho_B, z_rho_x, masked_Vx) |
| C_beta*A_beta*z_beta | beta_memory = _collapse_memory(beta_CA, z_beta, P) |
| C_rho*A_rho*z_rho_x/y | rho_memory_x/y = _collapse_memory(rho_CA, z_rho_x/y, Vx/Vy) |
| 多孔压力右端 | rhs_porous_P = -(divV + beta_memory + beta_CB*P) / beta_D |
| 多孔速度右端 | rhs_porous_Vx/Vy = -(dPdx/dPdy + rho_memory_x/y + rho_CB*Vx/Vy) / rho_D |

_init_material_state_space() 读入 A/B/C/D，_allocate_auxiliary_state() 为每个多孔单元分配 (n_state, Np, N_elements) 的状态张量并置零。_material_rhs() 执行 Az+Bu，_collapse_memory() 执行 C z 的收缩。

注意，基础声学右端会先对全部单元计算。随后代码用

```text
rhs_P  = torch.where(self._porous_mask_2d, rhs_porous_P, rhs_P)
rhs_Vx = torch.where(self._porous_mask_2d, rhs_porous_Vx, rhs_Vx)
rhs_Vy = torch.where(self._porous_mask_2d, rhs_porous_Vy, rhs_Vy)
```

只把 Porous 单元切换到 ER 方程。材料状态的输入也先乘以 _porous_mask_2d，因此 Air 和 Sponge 单元不会推进多孔材料 ADE。

### 7.1 空气域：标准线性声学

论文中的空气域方程为

$$
\frac{\partial \boldsymbol v}{\partial t}
  + \frac{1}{\rho_a}\nabla p = 0,
\qquad
\frac{\partial p}{\partial t}
  + \rho_a c_a^2\nabla\cdot\boldsymbol v = 0.
$$

其中 `p` 是声压，`v=(v_x,v_y)` 是质点速度，`rho_a` 和 `c_a` 分别是空气密度和声速。代码中的对应参数是 `main.py` 的 `RHO0=1.213` 和 `C0=343.0`；`build_simulation()` 将它们传入 `ExtendedReactionSimulation2D`。在 `RHS_operator()` 中，非多孔单元使用

$$
\frac{\partial p}{\partial t}=-K_\infty\nabla\cdot\boldsymbol v,
\qquad
\frac{\partial \boldsymbol v}{\partial t}
  =-\frac{1}{\rho_\infty}\nabla p,
$$

其中 `K_inf=rho_inf*c_inf^2`。对于空气，`rho_inf=rho_a`、`c_inf=c_a`，所以它就是上面的线性声学方程。

### 7.2 多孔材料：频域模型到时间域 ADE

论文在频域中用有效密度和有效压缩率描述反应性多孔材料：

$$
i\omega\rho_{\mathrm{eff}}(\omega)\widehat{\boldsymbol v}
  +\nabla\widehat p=0,
\qquad
i\omega\mathcal C_{\mathrm{eff}}(\omega)\widehat p
  +\nabla\cdot\widehat{\boldsymbol v}=0.
$$

代码使用 `beta` 表示论文中的 `\mathcal C`（压缩率）。有理函数拟合可写成

$$
\rho_{\mathrm{eff}}(s)\approx \rho_m
  +\sum_k\frac{B_{\rho k}}{s+\zeta_{\rho k}},
\qquad
\beta_{\mathrm{eff}}(s)\approx \beta_m
  +\sum_k\frac{B_{\beta k}}{s+\zeta_{\beta k}}.
$$

论文中的逐极点 ADE 形式为

$$
\dot{\phi}_{\rho k}+\zeta_{\rho k}\phi_{\rho k}=v,
\qquad
\dot{\phi}_{\beta k}+\zeta_{\beta k}\phi_{\beta k}=p.
$$

因此，代码里的 `z_rho_x`、`z_rho_y` 和 `z_beta` 是这些逐极点变量经过状态空间组合后的向量表示，而不是额外的物理场。`A/B` 矩阵负责状态演化，`C` 矩阵把记忆状态反馈到压力或速度方程，`D` 则给出瞬时高频项。

在本目录的 `fit_er_material.m` 中，`vectfit3` 将实测/计算的频域 `beta` 和 `rho` 数据分别拟合成实状态空间形式

$$
H(s)=D+C(sI-A)^{-1}B,
\qquad
\dot z=Az+Bu.
$$

拟合结果保存为 `A_beta/B_beta/C_beta/D_beta` 和 `A_rho/B_rho/C_rho/D_rho`。`n_poles=8` 表示每个材料响应使用 8 个极点；`.mat` 文件由 `ExtendedReactionMaterialFit.from_mat()` 读入。

### 7.3 `RHS_operator()` 中的材料记忆项

`ExtendedReactionSimulation2D._build_material_model()` 从拟合结果取出 `D_beta`、`D_rho` 作为高频常数 `beta_D`、`rho_D`，并计算

$$
K_\infty=\frac{1}{\beta_D},
\qquad
c_\infty=\sqrt{\frac{K_\infty}{\rho_D}},
\qquad
Z_\infty=\rho_Dc_\infty.
$$

对每个多孔单元，代码保存三组材料状态：一个压力状态 `z_beta`，以及两个方向的速度状态 `z_rho_x`、`z_rho_y`。在 `RHS_operator()` 中，压缩率状态对应

$$
\dot z_\beta=A_\beta z_\beta+B_\beta p,
$$

并通过

$$
\beta_D\frac{\partial p}{\partial t}
 +\nabla\cdot\boldsymbol v
 +C_\beta A_\beta z_\beta
 +C_\beta B_\beta p=0
$$

得到压力右端项。代码中这两项分别预先存成 `beta_CA=C_beta@A_beta` 和 `beta_CB=C_beta@B_beta`，然后实现为：

```text
beta_memory = beta_CA * z_beta
rhs_porous_P = -(divV + beta_memory + beta_CB * P) / beta_D
```

有效密度对两个速度分量分别使用同一个材料模型：

$$
\dot z_{\rho,x}=A_\rho z_{\rho,x}+B_\rho v_x,
\qquad
\rho_D\frac{\partial v_x}{\partial t}
 +\partial_xp+C_\rho A_\rho z_{\rho,x}
 +C_\rho B_\rho v_x=0,
$$

`y` 方向完全相同。对应代码为 `rho_memory_x`、`rho_memory_y` 和 `rhs_porous_Vx/Vy`。这正是论文中把卷积型频域材料关系改写为 auxiliary differential equations (ADE) 后的时间域实现。

### 7.4 DG 空间离散与材料界面上风通量

论文的统一一阶双曲系统可抽象写成

$$
\frac{\partial q}{\partial t}
 +A_x\frac{\partial q}{\partial x}
 +A_y\frac{\partial q}{\partial y}
 +Dq=g,
\qquad
q=[v_x,v_y,p]^T,
$$

材料记忆状态被包含在 `Dq` 和额外的 ADE 方程中。空间离散使用 DG-FEM：`Mesh2D` 提供单元和面信息，基函数梯度计算体积分项，数值通量通过 lifting 加回单元右端。

`MaterialUpwindFlux2D.compute_all()` 在每个面上使用左右状态的

$$
Z_L=\rho_Lc_L,
\qquad
Z_R=\rho_Rc_R,
\qquad
K_L=\rho_Lc_L^2,
$$

并令 `Delta v_n = n_x Delta v_x + n_y Delta v_y`，计算压力和法向速度通量：

$$
F_p=K_L\frac{Z_R\,\Delta v_n-\Delta p}{Z_L+Z_R},
\qquad
F_{v,n}=\frac{c_R\,\Delta p-c_LZ_R\,\Delta v_n}{Z_L+Z_R}.
$$

代码再用面法向量把 `F_v,n` 投影为 `F_vx` 和 `F_vy`。这对应论文的 Riemann/upwind 界面处理；本算例的 `BC_PARA` 只配置了外边界吸收 (`RI=0`) 和刚性边界 (`RI=1`)。论文薄覆盖层的界面条件为

$$
\boldsymbol v_a\cdot\boldsymbol n_a=-\boldsymbol v_m\cdot\boldsymbol n_m,
\qquad
p_a-p_m=Z_t(\boldsymbol v_a\cdot\boldsymbol n_a),
$$

但当前 `MaterialUpwindFlux2D` 没有额外的 `Z_t` 或覆盖层 ADE，因此这里相当于 `Z_t=0`。

#### 7.4.1 空气-多孔材料交界面的实际求解

空气和多孔材料之间没有单独的界面未知量。对于无遮盖层界面，论文条件可以写成

$$
p_a-p_m=0,
\qquad
\boldsymbol v_a\cdot\boldsymbol n_a
  =-\boldsymbol v_m\cdot\boldsymbol n_m.
$$

如果两侧都使用同一个面法向 n，则第二个条件就是法向速度通量连续。DG 方法不在离散层面直接把两侧的 p 和 v 赋成相同数值，而是通过数值通量在弱形式中施加上述条件。

一个内部面的代码路径是：

1. AcousticsSimulation2D._jump() 使用 vmapM 和 vmapP 取出面两侧的差值：

   `dP = P[vmapM] - P[vmapP]`，
   `dVx = Vx[vmapM] - Vx[vmapP]`，
   `dVy = Vy[vmapM] - Vy[vmapP]`。

2. ExtendedReactionSimulation2D._facewise_property() 将当前单元属性放在 left，将 mesh.EToE 指向的邻居属性放在 right。也就是说：

   - Air-Air 面的两侧都是 rho_a、c_a；
   - Porous-Porous 面的两侧都是 rho_D、c_infty；
   - Air-Porous 面的两侧分别使用空气和多孔材料的参数。

3. MaterialUpwindFlux2D.compute_all() 计算法向速度差和两侧阻抗：

   $$
   \Delta v_n=n_x\Delta v_x+n_y\Delta v_y,
   \qquad
   Z_L=\rho_Lc_L,
   \qquad
   Z_R=\rho_Rc_R,
   \qquad
   K_L=\rho_Lc_L^2.
   $$

   压力通量和法向速度通量为

   $$
   F_p=K_L
   \frac{Z_R\Delta v_n-\Delta p}{Z_L+Z_R},
   \qquad
   F_{v,n}=
   \frac{c_R\Delta p-c_LZ_R\Delta v_n}{Z_L+Z_R}.
   $$

   最后用面法向投影速度通量：

   $$
   F_{v_x}=n_xF_{v,n},
   \qquad
   F_{v_y}=n_yF_{v,n}.
   $$

4. 这些面通量乘以几何缩放 Fscale，再通过 lift 加回单元右端：

   ```text
   fluxP  *= Fscale
   fluxVx *= Fscale
   fluxVy *= Fscale

   rhs_P  += lift @ fluxP
   rhs_Vx += lift @ fluxVx
   rhs_Vy += lift @ fluxVy
   ```

因此，Air-Porous 面的主要作用是使用不同的左右阻抗 Z_L、Z_R 自动产生反射和透射；多孔材料的频率记忆并不直接进入 MaterialUpwindFlux2D，而是通过多孔单元内部的 z_beta、z_rho_x、z_rho_y 进入体方程。二者分工如下：

| 物理效应 | 离散位置 | 代码 |
| --- | --- | --- |
| 单元内声压/速度梯度 | DG 体积分 | _gradient() |
| 多孔材料频率记忆、耗散 | Porous 单元体方程 | _material_rhs()、_collapse_memory() |
| 空气-多孔阻抗不连续 | 共享内部面 | MaterialUpwindFlux2D |
| 面积分和单元耦合 | DG lifting | Fscale、lift |

当前界面通量只使用 D_beta、D_rho 导出的高频主部参数 rho_inf、c_inf。若要实现论文中的薄覆盖层，需要在界面通量中引入 Z_t，并额外推进覆盖层的 ADE 状态；当前代码没有这组界面状态，因此本算例对应 Z_t=0 的无遮盖层界面。

#### 7.4.2 外边界不是材料交界面

材料内部交界面有邻居单元，可以使用 left/right Riemann 通量；Outer 和 Rigid 是几何外边界，没有邻居单元，由 _compute_boundary_flux() 单独处理。代码先根据边界法向速度和边界阻抗计算出射特征

$$
u_{\mathrm{out}}=v_n+\frac{p}{Z},
$$

再设置入射特征

$$
u_{\mathrm{in}}=R_Iu_{\mathrm{out}}.
$$

main.py 中的参数是：

```text
Outer: RI = 0.0  -> 入射特征为 0，吸收边界
Rigid: RI = 1.0  -> 入射特征等于出射特征，刚性反射
```

所以 RI 不参与 Air-Porous 内部交界面；内部界面的反射和透射完全由左右材料参数以及 MaterialUpwindFlux2D 的上风通量决定。

### 7.5 激励、吸收层和时间推进

初始条件 `RadialPressurePulse2D_IC` 使用 COMSOL 对照算例中的二维径向压力脉冲：

$$
p_0(r)=\left(1-\frac{r^2}{B^2}\right)
\exp\left(-\frac{r^2}{2B^2}\right),
\qquad
r^2=(x-x_s)^2+(y-y_s)^2,
$$

脉冲中心为 `SOURCE_XYZ=(-1.0,0.5,0.0)`，宽度由 `PULSE_B=0.045` 控制，初始速度为零。`RECEIVER_XYZ` 指定接收点，时间历程由 `TSI_TI` 采样并保存到 `outputs/`。

几何文件中的 `Air=1`、`Porous=2`、`Sponge=3` 与 `main.py` 的 `DOMAIN_LABELS` 一致。海绵层不是论文 ADE 材料模型的一部分，而是代码在 `RHS_operator()` 最后加入的空间阻尼：

$$
\frac{\partial p}{\partial t}\leftarrow\frac{\partial p}{\partial t}-\sigma p,
\qquad
\frac{\partial \boldsymbol v}{\partial t}
\leftarrow\frac{\partial \boldsymbol v}{\partial t}-\sigma\boldsymbol v.
$$

`SPONGE_THICKNESS=L0/5` 决定阻尼带厚度；它只作用于 `Sponge` 标签内的单元。最后，`TSI_TI` 对上述半离散右端进行时间积分，`DEFAULT_ORDER=4`、`DEFAULT_NT=4` 和 `DEFAULT_CFL=0.25` 控制本算例的 DG 阶数、时间积分阶数和 CFL 步长。

## 8. 输出结果说明

默认输出目录结构大致是：

```text
outputs/
  5cm/
    results_on_the_run.mat
    snapshot.mat
    results_on_the_run_msh/
  15cm/
    results_on_the_run.mat
    snapshot.mat
    results_on_the_run_msh/
```

### 8.1 `results_on_the_run.mat`

这个文件是后处理主入口。当前默认输出里能看到这些关键字段：

- `time`
  接收点采样时刻
- `prec`
  接收点声压历史
- `dt`
  时间步长
- `Ntimesteps`
  总步数
- `total_time`
  总模拟时长
- `P`, `Vx`, `Vy`, `Vz`
  末时刻场变量
- `thickness_m`
  多孔层厚度
- `mesh_filename`
  使用的网格文件
- `fit_filename`
  使用的 ER 拟合文件
- `source_xyz`, `receiver_xyz`
  声源和接收点位置
- `beta_fit_rmserr`, `rho_fit_rmserr`
  材料拟合误差
- `sponge_sigma_max`, `sponge_thickness`
  sponge 参数

当前这份默认输出大致是：

- `5 cm`: `dt = 9.4797e-7`, `Ntimesteps = 10548`, `total_time ≈ 9.9992 ms`
- `15 cm`: `dt = 1.0311e-6`, `Ntimesteps = 9698`, `total_time ≈ 9.9998 ms`

### 8.2 `snapshot.mat`

保存末时刻全场：

- `P`
- `Vx`
- `Vy`
- `Vz`
- `Nx`
- `N_elements`
- `dt`
- `rho0`
- `c0`
- `mesh_filename`

### 8.3 `results_on_the_run_msh/`

这个目录下是可直接用 Gmsh 打开的场快照。文件名里会编码：

- 厚度
- 步号
- 物理时间

例如：

```text
porous_absorber_time_domain_5cm_step005802_t5.500134e-03.msh
```

表示 `5 cm` case、步号 `5802`、物理时刻约 `5.500134 ms`。

## 9. 当前结果和 COMSOL golden 的对比

以当前默认 `outputs/` 为例，用 `plot_results.py` 对比 COMSOL golden，得到的整段 `5 ms` 到 `10 ms` 相对 `L2` 误差大致是：

- `5 cm`: `6.60%`
- `15 cm`: `7.13%`

更细一点：

- 早期主反射段 `5-7 ms`，误差更小，约 `4.81%` 和 `5.50%`
- 后段 `7-10 ms` 相对误差会明显变大

这里要注意：后段“相对误差变大”不等于“绝对误差失控”。尾波本身已经很小，`15 cm` 尤其如此，所以同量级的绝对误差会被相对指标放大。这也是 `plot_results.py` 里加入相对误差下限的原因。

当前默认输出的绝对误差量级大致是：

- `5 cm`: `RMSE ≈ 8.75e-4`
- `15 cm`: `RMSE ≈ 9.04e-4`

从主峰位置和幅值看，当前实现没有明显的 source、receiver、厚度或时间轴配置错误；差异更多来自：

- DG 三角网格与 COMSOL 网格不完全相同
- sponge 吸收层和 COMSOL absorbing layer 不是同一种实现
- 多孔材料时域离散与跨材料界面数值通量带来的差异

## 10. 调 sponge 强度

sponge 强度由：

```bash
--sponge-sigma-max
```

控制，默认 `2500`。数值越大，吸收层阻尼越强；越小则越弱。

示例：

```bash
rtk python main.py \
  --thickness both \
  --sponge-sigma-max 1000 \
  --output-root outputs_sponge_1000

rtk python main.py \
  --thickness both \
  --sponge-sigma-max 5000 \
  --output-root outputs_sponge_5000
```

建议先比较：

- `1000`
- `2500`
- `5000`

然后再看 `8-10 ms` 尾部的绝对误差和相对误差是否更贴近 COMSOL。

## 11. CUDA graph 和 2D ER 深度优化性能

这一节记录 2D ER case 在 V100 上的时间迭代性能、最终保留的优化，以及 CUDA graph 节点变化。`11.1` 到 `11.7` 对应此前 `7107` 单元 `5 cm` 网格上的 compact-ER 优化记录；`11.8` 追加当前 `126363` 单元大网格上引入 `ER RHS backend = gemm` 之后的实测结果。

### 11.1 对比模式

- `packed_no_graph`：packed RHS，不使用 CUDA graph。
- `packed_graph`：packed RHS + CUDA graph，不使用 Triton timestep kernel。
- `graph_deep_fused_legacy`：CUDA graph + Triton flux/boundary + 原有 deep-fused ER RHS。
- `graph_compact_er`：在 legacy deep-fused 基础上，再启用紧凑 porous ADE 状态、合并 simple-RI 边界和接收器采样 kernel。

`graph_compact_er` 是当前推荐路径。案例参数 `--use-2d-partitioned-er-rhs` 控制这一层优化；名称保留了开发阶段的接口，但当前实现不是把全域 RHS 拆成多个材料 kernel，而是让全域 fused RHS 读取按 porous 区域紧凑存储的 ADE 状态。

### 11.2 测试方法

测试环境和参数：

- GPU：`Tesla V100-SXM2-16GB`
- 厚度：`0.05 m`
- 网格单元数：`7107`
- porous 单元数：`316`
- DG 阶数：`Nx = 4`，因此 `Np = 15`
- TSI 阶数：`Nt = 4`
- CFL：`0.25`
- 时间步数：`1000`
- CUDA graph chunk：`1`
- 每种模式运行三次，表中采用中位数
- `progress=False`
- `synchronize_timing=True`

复现脚本：

```bash
rtk env PYTHONPATH=/media/liu/research/linux/edg-muxi/edg-acoustics python - <<'PY'
import importlib.util
from pathlib import Path
import statistics
import torch

repo = Path("/media/liu/research/linux/edg-muxi/edg-acoustics")
main_path = (
    repo
    / "examples"
    / "dgfem_acoustic_2d"
    / "porous_absorber_time_domain"
    / "main.py"
)
spec = importlib.util.spec_from_file_location("porous_main_perf", main_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

configs = [
    ("packed_no_graph", False, False, False, False),
    ("packed_graph", True, False, False, False),
    ("graph_deep_fused_legacy", True, True, True, False),
    ("graph_compact_er", True, True, True, True),
]

for name, use_graph, use_triton, use_deep, use_compact in configs:
    samples = []
    for _ in range(3):
        sim = mod.build_simulation(
            thickness=0.05,
            fit_path=mod.DEFAULT_FIT,
            mesh_path=mod.default_mesh_path(0.05),
            Nx=4,
            Nt=4,
            use_packed_rhs=True,
            use_triton_kernels=use_triton,
            use_triton_deep_rhs=use_deep,
            use_triton_partitioned_er_rhs=use_compact,
        )
        sim.time_integration(
            n_time_steps=1000,
            progress=False,
            use_cuda_graph=use_graph,
            cuda_graph_chunk_steps=1,
            synchronize_timing=True,
        )
        samples.append(sim.last_time_integration_elapsed_s)
        del sim
        torch.cuda.empty_cache()
    elapsed = statistics.median(samples)
    print(name, f"elapsed_s={elapsed:.6f}", f"per_step_ms={elapsed:.6f}")
PY
```

因为这里固定运行 `1000` 步，所以数值上 `elapsed_s` 与 `per_step_ms` 相同。

### 11.3 当前性能结果

`2026-07-17` 的三次中位数：

| 模式 | `1000` 步耗时 | 每步时间 | 相对 `packed_no_graph` 加速 |
| --- | ---: | ---: | ---: |
| `packed_no_graph` | `6.105324 s` | `6.105324 ms` | `1.00x` |
| `packed_graph` | `4.198190 s` | `4.198190 ms` | `1.45x` |
| `graph_deep_fused_legacy` | `0.379678 s` | `0.379678 ms` | `16.08x` |
| `graph_compact_er` | `0.308403 s` | `0.308403 ms` | `19.80x` |

最终 `graph_compact_er` 相对：

- `packed_graph`：约 `13.61x`
- 当前 `graph_deep_fused_legacy`：约 `1.23x`
- 上一版 README 记录的 `0.558 ms/step` deep-fused 结果：约 `1.81x`

### 11.4 最终保留的优化

1. 全域 fused RHS 的加载和算术优化

- `Dr/Ds` 两个方向共用同一次 `P/Vx/Vy` 加载，不再为每个方向重复读全局内存。
- 2D 模型中 `Vz` 的数值通量恒为零，deep-fused 累加路径不再读取、lift 或回写无效的 `Vz` surface 项。
- `1/beta_D`、`beta_CB/beta_D`、`1/rho_D`、`rho_CB/rho_D` 在 launcher 中预计算，避免 kernel 内逐点除法。
- V100 的 `Np=15`、8 个 ADE 状态使用实测较优的 `BLOCK_SIZE=64, num_warps=2`；内界面 kernel 使用 `BLOCK_SIZE=128`。

2. 紧凑 porous ADE 状态

原路径为每个 ADE 变量保存全域张量：

```text
[8 states, 15 nodes, 7107 elements] = 6,822,720 bytes
```

当前 graph 内工作状态只保存 porous 单元：

```text
[8 states, 15 nodes, 316 porous elements] = 303,360 bytes
```

`z_beta/z_rho_x/z_rho_y` 在进入时间循环前收集到紧凑张量，循环结束后再写回公开的全域状态。每个时间步仍保持 Taylor 方法要求的三次 ADE state copy，但单份 copy 缩小约 `22.49x`。

3. 合并 simple-RI 边界

当前 case 的两个边界共有：

- Outer：`840` 个 face nodes
- Rigid：`395` 个 face nodes

静态 `vmap_q/flux_map_q/nx/ny/rho/c/z/k/Fscale/RI` 在初始化时拼接。每个 Taylor stage 原来的两个 `boundary_ri_flux_2d_kernel` 合并为一个 `combined_boundary_ri_flux_2d_kernel`，并通过切片维持原 `BCvar["vn"/"ou"/"in"]` 接口。

4. 接收器采样

原来的 `index_select + multiply + sum` 三个 device kernel 合并为一个 `sample_receivers_2d_kernel`。对当前单接收器 case，graph 内接收器节点从 3 个降为 1 个。

### 11.5 未采用的实验路径

开发中实际测试过以下两种更激进的方案，但没有保留在运行路径中：

- 把全域 RHS 拆成 non-porous 与 porous 两个 kernel。
  当前优化后的单个 `fused_er_rhs_2d_kernel` 约 `45.5 us/stage`；拆分后约为 `49.9 + 25.4 us/stage`，节点减少但总执行时间增加。
- 对内界面成对处理，一次读取两侧状态并写两个方向。
  在 V100 上，成对 heterogeneous-material flux 约 `27.6 us/stage`，慢于当前方向化 kernel 的约 `19.3 us/stage`。

这说明 CUDA graph 中“节点更少”不是充分条件；寄存器压力、并行块数量和每个 kernel 的有效工作量必须一起测量。

### 11.6 CUDA graph 节点和复制量

最终 graph 每步包含：

| graph 节点 | 节点数 |
| --- | ---: |
| `interior_material_flux_2d_kernel` | `4` |
| `combined_boundary_ri_flux_2d_kernel` | `4` |
| `fused_er_rhs_2d_kernel` | `4` |
| `fused_er_aux_update_diag_kernel` | `4` |
| `sample_receivers_2d_kernel` | `1` |
| **kernel 合计** | **`17`** |
| D2D memcpy | `4` |

节点变化：

- 旧 packed + Triton flux/boundary：`275` 个 kernel 节点
- 上一版 deep-fused 报告：`23` 个 kernel 节点
- 当前 legacy deep-fused（已含本轮 receiver kernel）：`21` 个 kernel 节点
- 当前 compact ER：`17` 个 kernel 节点

因此当前相对上一版 `23` 节点减少 `26.1%`，相对最初 `275` 节点压缩约 `16.18x`。

D2D 节点数没有变化，但复制字节数明显下降：

| 模式 | `Q` copy | 三份 ADE copy | 每步 D2D 总量 |
| --- | ---: | ---: | ---: |
| legacy deep-fused | `3,411,360 B` | `3 x 6,822,720 B` | `23,879,520 B` |
| compact ER | `3,411,360 B` | `3 x 303,360 B` | `4,321,440 B` |

每步 D2D 字节数减少约 `81.9%`。唯一的全域 `Q` copy 用于保持 Taylor 第一阶段“原始 Q 作为 RHS 输入、最终 Q 作为累加输出”的语义，不能在不同 Triton program 并发读写时直接原地消除。

### 11.7 启用和回退

当前案例默认启用 packed RHS、Triton kernel、deep-fused RHS、partitioned ER 和 CUDA graph。`--use-2d-partitioned-er-rhs` 只有在前三层快速路径均实际启用、且 porous 单元连续时才会生效。当前版本还增加了 `--er-rhs-backend {auto,legacy,gemm}`：

- `auto`：默认值。仅在 `CUDA + Triton + fp64 + partitioned ER + V100 + 单元数不小于阈值` 时自动切到 `gemm`。
- `legacy`：强制使用原有 `fused_er_rhs_2d_kernel + fused_er_aux_update_diag_kernel` 路径。
- `gemm`：强制使用“大网格 ER RHS”路径：`[Dr;Ds]` 合并 GEMM + `lift` GEMM + `nonporous/porous` 两个 Triton post-kernel。

推荐命令：

```bash
rtk env PYTHONPATH=/media/liu/research/linux/edg-muxi/edg-acoustics \
  python ./main.py \
  --thickness 0.05 \
  --n-time-steps 1000
```

显式回到 legacy deep-fused 路径：

```bash
rtk env PYTHONPATH=/media/liu/research/linux/edg-muxi/edg-acoustics \
  python ./main.py \
  --thickness 0.05 \
  --n-time-steps 1000 \
  --use-2d-packed-rhs \
  --use-2d-triton-kernels \
  --use-2d-partitioned-er-rhs \
  --er-rhs-backend legacy
```

Nsight Systems 复现命令：

```bash
rtk env PYTHONPATH=/media/liu/research/linux/edg-muxi/edg-acoustics \
  nsys profile -o 005-thickness-cuda-graph-compact-er \
  --trace=cuda,nvtx,cublas,osrt \
  --cuda-graph-trace=node \
  --force-overwrite=true \
  --cuda-memory-usage=true \
  --stats=true \
  python ./main.py \
  --thickness 0.05 \
  --n-time-steps 10 \
  --no-progress \
  --save-mesh-at-ms 0 \
  --use-2d-packed-rhs \
  --use-2d-triton-kernels \
  --use-2d-partitioned-er-rhs
```

### 11.8 当前大网格补充更新

`2026-07-18` 在当前工作区的 `5 cm` 网格上，`gmsh` 细化后默认 `porous_absorber_time_domain_5cm.msh` 已变为：

- 总单元数：`126363`
- Air：`90338`
- Porous：`2868`
- Sponge：`33157`
- GPU：`Tesla V100-SXM2-16GB`
- DG 阶数：`Nx = 4`，因此 `Np = 15`
- TSI 阶数：`Nt = 4`
- 计时口径：先 warmup `2` 步，随后正式跑 `30` 步，`progress=False`，`synchronize_timing=True`

这张大网格上，原先单个 `fused_er_rhs_2d_kernel` 已经不再是“越深融合越快”的形态。瓶颈转成：

- 一个 kernel 内重复遍历 `15 x 15` 本地节点，`P/Vx/Vy` 和 ADE work 状态被多次从 HBM 读回
- `Dr/Ds` 与 `lift` 的局部矩阵很小，但右矩阵 `3 * N_elements` 很长，更适合直接交给 cuBLAS GEMM
- porous 区的 ADE 更新仍然需要自定义 kernel，但 nonporous 和 porous 可以分成两个简单的 post-kernel

因此当前代码新增 `ER RHS backend = gemm`：

1. `torch.mm([Dr;Ds], q[:, :3N])` 一次生成 `dr/ds` 两个方向的 `P/Vx/Vy`
2. `torch.mm(lift, flux[:, :3N])` 一次生成 surface 项
3. Triton `nonporous` post-kernel 负责 Air/Sponge 的 metric、volume、surface、sponge 和 Taylor 累加
4. Triton `porous` post-kernel 负责 porous 的 metric、memory collapse、ADE work/update 和 Taylor 累加

### 11.9 当前大网格性能结果

下面的三组数据都来自当前 `126363` 单元 `5 cm` 网格：

| 模式 | backend | CUDA graph | `30` 步耗时 | 每步时间 | 相对 `legacy_eager` |
| --- | --- | --- | ---: | ---: | ---: |
| `legacy_eager` | `legacy` | `False` | `0.314020 s` | `10.467339 ms` | `1.00x` |
| `legacy_graph` | `legacy` | `True` | `0.314233 s` | `10.474443 ms` | `1.00x` |
| `gemm_graph` | `gemm` | `True` | `0.175657 s` | `5.855227 ms` | `1.79x` |

当前结论：

- 在这张大网格上，`legacy` 路径即使挂到 CUDA graph，收益也已经接近于零。主耗时不再是“很多碎 kernel 的 launch 开销”，而是大 kernel 本身反复读写全局内存。
- `gemm_graph` 相对 `legacy_graph` 约 `1.79x`，相对 `legacy_eager` 也约 `1.79x`。这和前面基于 Nsight 估算的 `1.7x - 2.0x` 目标区间一致。
- 默认 `--er-rhs-backend auto` 在当前 V100 和这张 `126363` 单元网格上会自动选到 `gemm`。如果切回较小网格，或者运行环境不满足 `CUDA + Triton + fp64 + V100 + 阈值`，则会自动回退到 `legacy`，避免对原有小网格路径造成回归。

当前大网格复现命令：

```bash
rtk env PYTHONPATH=/media/liu/research/linux/edg-muxi/edg-acoustics \
  python ./main.py \
  --thickness 0.05 \
  --n-time-steps 30 \
  --no-progress \
  --save-mesh-at-ms 0 \
  --er-rhs-backend auto
```

如果想强制比较两条路径：

```bash
rtk env PYTHONPATH=/media/liu/research/linux/edg-muxi/edg-acoustics \
  python ./main.py \
  --thickness 0.05 \
  --n-time-steps 30 \
  --no-progress \
  --save-mesh-at-ms 0 \
  --er-rhs-backend legacy
```

```bash
rtk env PYTHONPATH=/media/liu/research/linux/edg-muxi/edg-acoustics \
  python ./main.py \
  --thickness 0.05 \
  --n-time-steps 30 \
  --no-progress \
  --save-mesh-at-ms 0 \
  --er-rhs-backend gemm
```

## 12. 常见注意事项

- `results_on_the_run_msh/` 不会自动清空。
  同一个 `output_root` 重复运行时，旧快照会保留，新快照会继续追加。想保持目录干净，最好换新的 `--output-root`，或者先手动清空旧目录。
- `results_on_the_run.mat` 会被新的运行覆盖。
  如果你要保留多组结果，请改 `--output-root`。
- `gmsh` 和 `octave` 不是每次都需要。
  只有重新生成网格或重新做材料拟合时才需要。
- `15 cm` 的尾波更弱。
  所以它的逐点相对误差图会比 `5 cm` 更敏感，不应只盯着尾部相对误差峰值判断 case 是否错误。
