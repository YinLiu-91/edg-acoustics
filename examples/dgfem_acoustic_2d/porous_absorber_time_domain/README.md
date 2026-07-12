# 运行case
```bash
  PYTHONPATH=/media/liu/research/linux/edg-muxi/edg-acoustics \
  python main.py \
    --thickness both \
    --force-fit \
    --save-step 1000 \
    --save-mesh-at-ms 5.5
```
如果重新生成网格
```bash
  PYTHONPATH=/media/liu/research/linux/edg-muxi/edg-acoustics \
  python main.py \
    --thickness both \
    --force-fit \
    --save-step 1000 \
    --save-mesh-at-ms 5.5 \
    --force-mesh
```
1. sponge 强度由 --sponge-sigma-max 控制，默认是 2500，对应 examples/dgfem_acoustic_2d/porous_absorber_time_domain/main.py:327。数值越大，吸收层阻尼越强；数值越小，阻尼越弱，可以试试1000，5000，2500哪个吸收效果最好

```bash
  PYTHONPATH=/media/liu/research/linux/edg-muxi/edg-acoustics \
  python main.py \
    --thickness both \
    --force-fit \
    --save-step 1000 \
    --save-mesh-at-ms 5.5 \
    --force-mesh \
    --sponge-sigma-max 5000 \
    --output-root outputs_sponge_5000
```
# 画图

  前提是本机有 gmsh 和 octave。输出会落到 outputs/5cm 和 outputs/15cm，里面有 results_on_the_run.mat、snapshot.mat 和 results_on_the_run_msh/。画图用：
```bash
 python plot_results.py \
    outputs/5cm/results_on_the_run.mat \
    outputs/15cm/results_on_the_run.mat \
    --output receiver_history_with_golden.png
```

如果你想显式指定 golden 文件，也可以：
```bash
python plot_results.py \
    outputs/5cm/results_on_the_run.mat \
    outputs/15cm/results_on_the_run.mat \
    --golden \
    5cm_er_comsol_golden.txt \
    15cm_er_comsol_golden.txt \
    --output receiver_history_with_golden.png
```