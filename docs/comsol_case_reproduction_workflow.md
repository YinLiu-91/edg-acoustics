# COMSOL 声学 case 到 EDG 的复现流程

本文档沉淀从 COMSOL `.mph/.step` case 复现到 EDG Acoustics 的完整流程。目标不是“能读入几何”或“能跑出一条曲线”，而是在相同声学方程、边界语义、材料传递函数、声源、接收点和输出时间下，对 COMSOL 参考结果给出可解释的误差。

## 复现完成标准

一个 case 至少需要留下这些可审计产物：

- 输入清单：`.mph`、`.step`、材料表、COMSOL 参考结果、COMSOL 版本和本机 COMSOL 版本。
- MPH 语义报告：active physics features、named selections、参数、插值资源、source、receiver、study time range。
- 边界映射表：COMSOL entity id 到 EDG/Gmsh physical label，要求互斥、覆盖所有 acoustic boundary，且 volume label 单独存在。
- 网格诊断：bbox、physical labels、triangle/tetra 数、最小面积/体积/边长、最小内切球直径、估计显式时间步长和总步数。
- 材料拟合报告：输入 admittance/impedance 的物理量说明、`RI/RP/CP` `.mat`、RMS/max error、passivity 检查。
- EDG 求解脚本：mesh、边界、source、receiver、time step guard、输出 schema。
- 对比结果：EDG 与 COMSOL 在相同 receiver 和时间点的全时段误差、分时间窗误差和图。

## 1. 输入和版本清单

先记录文件和版本，不要直接从 STEP 推断边界：

```bash
rtk file case.mph
rtk unzip -l case.mph
rtk unzip -p case.mph modelinfo.xml
rtk /usr/local/comsol64/multiphysics/bin/comsol -version
```

如果本机 COMSOL 版本低于 `.mph` 保存版本，不要假设能可靠打开或重新计算模型；优先请求匹配版本或由 COMSOL 导出的 mesh/reference 数据。

## 2. 从 MPH 恢复物理语义

`.mph` 本质上是 ZIP archive。常用入口：

- `dmodel.xml`：named selections、physics features、selection entity ids。
- `smodel.json`：已求值参数、study/time 配置、表达式候选。
- `resources/*`：导入的 admittance/impedance/interpolation 表。
- `modelinfo.xml`：版本、计算信息、模型元数据。

关键规则：

- `PhysicsFeature` 是求解语义来源，helper/display selection 只能辅助解释。
- 如果 active impedance/normal velocity/source feature 和某个 hard-wall selection 冲突，以 active physics feature 为准。
- COMSOL selection 里可能出现 `-454,-1` 这类 sentinel，不能当成 surface id。
- 不允许按名称猜边界，例如看到 `roof` 就直接映射吸声边界；必须能追溯到 active feature 或明确的用户映射。

推荐先用 skill 模板生成原始报告：

```bash
rtk cp /home/liu/.codex/skills/comsol-step-gmsh-repro/assets/templates/recover_mph_case.py.template recover_mph_case.py
rtk python recover_mph_case.py case.mph --json-out mph_semantics_raw.json
```

人工审查后建立边界映射表，例如：

| EDG label | 名称 | COMSOL 来源 | 处理方式 | entity 数 |
|---:|---|---|---|---:|
| 11 | `DefaultHardWall` | active hard wall/default remainder | hard wall | 待填 |
| 15 | `Seats` | active impedance/admittance feature | fitted material | 待填 |
| 21 | `Source` | active normal velocity/source feature | prescribed source | 待填 |

## 3. 选择网格路径

有两条路径。选择依据是 entity id 是否可靠、显式时间步长是否可用，以及 COMSOL 是否已经有 defeature/virtual geometry mesh。

### STEP/OCC 到 Gmsh

适用条件：

- STEP 导入后 surface tag 可以和 MPH entity id 对上。
- bbox、单位、volume 数量合理。
- `gmsh -check` 无读取错误。
- 最小内切球直径给出的 EDG 时间步长可接受。

示例命令：

```bash
rtk gmsh -3 case.geo \
  -setnumber lc 0.12 \
  -format msh2 \
  -o case_lc0p12.msh
rtk gmsh -check case_lc0p12.msh -
```

注意：`gmsh -check` 能读入不等于适合瞬态 EDG。仍需要检查最小单元尺度和估计步数。

### COMSOL virtual/defeatured mesh 导出

当 STEP 存在极小碎片、HXT/Delaunay 仍生成极小内切球，或 `.mph` 已经有清理后的 `mesh2` 时，优先用 COMSOL 导出 mesh。流程是运行 COMSOL mesh，导出带几何引用的 NASTRAN，再转换为 Gmsh physical tags：

```bash
rtk cp /home/liu/.codex/skills/comsol-step-gmsh-repro/assets/templates/ExportComsolMesh.java.template ExportCaseMesh.java
# 将 __JAVA_CLASS__ 改成 ExportCaseMesh 后编译
rtk /usr/local/comsol64/multiphysics/bin/comsol compile ExportCaseMesh.java

rtk /usr/local/comsol64/multiphysics/bin/comsol batch \
  -inputfile ExportCaseMesh.class \
  case.mph case_mesh.nas comp1 mesh2 \
  -batchlog case_mesh_export.log \
  -batchlogout \
  -nosave
```

转换时必须提供人工审查后的 JSON mapping；未知 NASTRAN boundary ref 应失败，不能落到默认硬壁：

```bash
rtk cp /home/liu/.codex/skills/comsol-step-gmsh-repro/assets/templates/convert_nastran_to_gmsh.py.template convert_nastran_to_gmsh.py
rtk python convert_nastran_to_gmsh.py \
  --nastran case_mesh.nas \
  --mapping boundary_mapping.json \
  --output case_mesh.msh \
  --json-out case_mesh_report.json
```

## 4. 网格验收

至少检查：

- boundary physical labels 是否包含所有预期标签。
- 每个 boundary triangle 是否有且仅有一个 surface physical tag。
- acoustic volume 是否有 `Physical Volume(..., 1)` 或明确 volume label。
- bbox 是否和 COMSOL 模型尺度一致。
- 最小 triangle area、tet volume、edge length 是否异常。
- 最小 tetra insphere diameter 是否导致不可接受时间步长。

显式时间步长的估计可用：

```text
dt ~= CFL * min_insphere_diameter / ((2 * Nx + 1) * c0)
steps ~= total_time / dt
```

car-cabin 的 STEP/HXT mesh 虽然 physical groups 正确，但最小内切球约 `8.2307e-07 m`，`Nx=4, CFL=0.5` 下 `0.06 s` 约需 `4.50e8` 步，不适合作为默认复现网格。COMSOL virtual mesh 的最小内切球约 `1.788e-03 m`，同样设置约 `207196` 步，因此作为推荐求解网格。

## 5. 材料拟合

EDG `AbsorbBC` 使用的是反射系数的 rational approximation，不应直接复制 COMSOL 内部 rational coefficients，除非已经确认两边表示的传递量和符号约定完全一致。

如果 COMSOL 表给的是 admittance `Y`，先转成法向入射反射系数：

```text
R = (1 - rho0*c0*Y) / (1 + rho0*c0*Y)
```

如果表给的是 impedance `Z`，则：

```text
R = (Z - rho0*c0) / (Z + rho0*c0)
```

使用 material fit 模板：

```bash
rtk cp /home/liu/.codex/skills/comsol-step-gmsh-repro/assets/templates/fit_admittance.m.template fit_admittance.m
# 编辑 materials = {'name','table.txt',n_poles,rms_limit; ...}
rtk octave -qf fit_admittance.m
```

生成的 `.mat` 至少包含：

```text
RI AS lambdaS BS CS alphaS betaS freq ApproxValue trueValue rms_error max_error max_abs_R
```

验收标准应写入 case README：RMS/max error、`max_abs_R <= 1` passivity、拟合频段，以及误差较大的材料是否会主导 transient 差异。

## 6. EDG main.py 组织

`main.py` 应从恢复出的 COMSOL 语义配置，而不是从几何名称猜：

- zero initial condition 还是初始压力场。
- source 是 active normal velocity、pressure、volume source 还是初始激励。
- 每个 physical label 的 hard wall、constant reflection、impedance、fitted material。
- receiver 坐标和输出时间必须和 COMSOL 对齐。
- 对含 prescribed source 的边界，确保每步都能更新物理时间。
- 保存 `dt`、steps、mesh、receiver、boundary/material 摘要，便于之后解释误差。

可从模板开始：

```bash
rtk cp /home/liu/.codex/skills/comsol-step-gmsh-repro/assets/templates/edg_main.py.template main.py
```

模板只提供结构；项目内 API 和具体 COMSOL 方程需要按 case 校正。car-cabin 的实际脚本见 `examples/car_cabin_acoustics_transient_63_cleared/main.py`。

## 7. 验证和误差说明

推荐三层验证：

- 结构验证：MPH entity 覆盖、mapping 无重复/遗漏、mesh physical labels 正确、材料 `.mat` schema 正确。
- 短时 smoke：只跑波到达复杂吸收边界之前的时间窗，验证 source、receiver、时间步和输出 schema。
- 完整对比：在 COMSOL output times 上插值或同步输出，给出全时段和分时间窗误差。

误差报告至少包含：

```text
rms_error = sqrt(mean((p_edg - p_comsol)^2))
max_error = max(abs(p_edg - p_comsol))
relative_l2 = norm(p_edg - p_comsol) / norm(p_comsol)
```

如果后期误差显著增大，优先检查边界吸收和反射波，而不是只看前期 smoke。前期声波尚未到达吸收边界时，sponge/PML/material 边界差异可能被掩盖。

## 8. car-cabin worked example

本仓库的具体示例在 `examples/car_cabin_acoustics_transient_63_cleared/`：

- `README.md`：已恢复的 COMSOL boundary groups、STEP/Gmsh 与 COMSOL virtual mesh 的对比、材料拟合误差、运行命令。
- `car_cabin_boundary_groups.py`：从 `.mph` 恢复 active boundary model，并检查 mesh physical groups。
- `ExportCarCabinMesh2.java`：运行 COMSOL `mesh2` 并导出 NASTRAN。
- `convert_comsol_nastran_to_gmsh.py`：将 COMSOL mesh export 转成 EDG/Gmsh physical labels。
- `fit_car_cabin_admittance.m`：将 carpet/roof/seat admittance 表拟合为 EDG `.mat`。
- `main.py`：使用推荐 COMSOL virtual mesh 和恢复边界运行 EDG transient。

后续可直接要求 Codex 使用本地 skill：

```text
使用 $comsol-step-gmsh-repro 复现这个 COMSOL .mph/.step 声学 case，并生成 EDG mesh、material fit、main.py 和对比报告。
```
