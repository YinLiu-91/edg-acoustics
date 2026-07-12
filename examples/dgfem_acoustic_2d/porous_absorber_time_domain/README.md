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

## 7. 主要代码在做什么

这个 case 的核心不只是“把空气波动方程搬到 2D”。

它实际做了三件更具体的事：

1. 在空气域内求解标准线性声学方程。
2. 在多孔域内用拟合后的 `beta(s)` 和 `rho(s)` 引入辅助状态，把频率相关材料转成时域可推进的形式。
3. 在几何外圈加 sponge 吸收层，减少有限计算域带来的边界反射。

这三部分分别对应：

- 空气域和边界基础框架：`edg_acoustics/acoustics_simulation_2d.py`
- ER 多孔材料与 sponge：`edg_acoustics/acoustics_simulation_2d_er.py`
- 跨材料界面的数值通量：`edg_acoustics/preprocessing.py` 里的 `MaterialUpwindFlux2D`
- 时间推进：`edg_acoustics/time_integration.py` 里的 `TSI_TI`

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

## 11. 常见注意事项

- `results_on_the_run_msh/` 不会自动清空。
  同一个 `output_root` 重复运行时，旧快照会保留，新快照会继续追加。想保持目录干净，最好换新的 `--output-root`，或者先手动清空旧目录。
- `results_on_the_run.mat` 会被新的运行覆盖。
  如果你要保留多组结果，请改 `--output-root`。
- `gmsh` 和 `octave` 不是每次都需要。
  只有重新生成网格或重新做材料拟合时才需要。
- `15 cm` 的尾波更弱。
  所以它的逐点相对误差图会比 `5 cm` 更敏感，不应只盯着尾部相对误差峰值判断 case 是否错误。
