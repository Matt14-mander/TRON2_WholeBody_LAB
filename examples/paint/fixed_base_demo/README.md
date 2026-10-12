# MuJoCo 固定底座初步估计器

这是初步效果演示，使用仓库原始 TRON2 MuJoCo 模型，加 0.5 kg 刚性箱形负载。
训练数据为 6 个姿态组，每组 zero/±Fx/±Fy/±Mz 七个回合；组级划分，无随机帧混合。
仅适用于当前 PD=60/6、小幅姿态变化、5 N/0.5 Nm、50 Hz 的固定底座条件。
不代表行走、跨负载或真机验证，不输出控制命令。

示例模型训练/验证版本为 **MuJoCo 3.2.7**。3.13.0 下已复现第一关节零外力响应变化及
Fy/Mz 估计偏置，不能直接沿用本示例的验证结论。请先锁定 3.2.7 做对照；
若要保留其他版本，应在该版本下重新采集训练，并验证跨版本泛化。

在仓库根目录执行：

```bash
python scripts/paint/run_mujoco.py --model_dir examples/paint/fixed_base_demo --arm_pd_backend explicit --output_dir logs/paint/demo_run --seed 142 --warmup 0.5 --duration 1 --recovery 0.5
python scripts/paint/analyze_pd_pairs.py logs/paint/demo_run
```

输出目录必须为空；有桌面时可加 `--viewer`。
此示例使用旧显式 PD 训练；复现上述指标须选择 `--arm_pd_backend explicit`。
新实验默认使用 native PD，其响应和估计误差需要单独验证。
完整 Linux 采集、训练和部署操作见 [开发指南](../../../docs/paint_intent_pd.md)。
