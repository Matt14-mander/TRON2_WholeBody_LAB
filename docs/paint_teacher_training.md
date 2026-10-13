# PAINT 高层 teacher：冻结 locomotion 的受力协调训练

## 本轮边界与实现状态

这轮 PPO 训练高层的「受力后怎样移动底座、怎样调整臂姿」，低层仍负责稳定行走。
Teacher 直接观察真实负载交互 `[Fx,Fy,Mz]`，没有 intent estimator，也不训练力预测网络。
这是独立 Isaac Lab manager-based 任务，不改变原 WholeBody/PAINT 数据采集入口。

本地已检查 Torch 控制契约、原 PPO 更新以及语法；**尚未执行 Isaac Sim/PhysX 闭环、未开始正式训练**。
本机当前 Python 环境只有 Isaac Lab 源码，没有 Isaac Sim runtime，且没有正式低层 checkpoint。
需要在已有 Linux Isaac 环境中先完成下述 smoke 和基线，才能评价带负载的稳定性。

2026-10-13 服务器诊断补充：旧 checkpoint 的零速回合在约 0.35 s 失败；`-0.25 m/s` 行走在无负载/
有负载下均完成 3 个 20 s 回合。当前先按 [低层站立与起停微调方案](locomotion_standwalk_finetune.md)
补齐低层能力，使用通过验收的新 checkpoint 后再执行本页训练流程。不能把稳定行走等同于稳定零速支撑。

| 阶段 | 优化对象 | 力输入 | 动作 |
|---|---|---|---|
| 原 WholeBody locomotion | 腿部 actor＋低层 history encoder | 未来 5×6 arm-on-base wrench | 十个腿关节位置 |
| 已有 intent estimator | 臂部历史到力的回归网络 | 已知负载力作为监督标签 | 三维力估计，旁路 |
| 本轮 teacher | 高层 actor/critic | 当前真实 external-on-payload Fx/Fy/Mz | 底座速度＋六个臂关节位置目标 |

## 控制与观测契约

- 物理 200 Hz (`dt=0.005`)，teacher 25 Hz (`decimation=8`)，冻结低层每四个物理步推理一次，即 50 Hz。
- 严格加载原 flat WholeBody ELU actor `78→512→512→256→128→10` 和 encoder `420→256→128→3`。
  只跳过低层 critic；actor 或 encoder 任何参数缺失/维度不符均报错，不允许部分加载。
- 低层输入顺序保持 `latent(3), proprioception(42), future wrench(30), base command(3)`。
  十帧低层历史保持 50 Hz、最旧帧在前；不以高层九维 action 替代低层十维 last action。
- 高层原始动作九维：`[vx, vy, wz, arm1..arm6]`。前三维缩放为机身坐标速度，后三维为
  `q_nominal + 0.25 * clip(action, -1, 1)`，不是累积关节增量；施加软限位及目标变化率限制。
  初始速度上限 `0.35/0.20/0.40 m/s,m/s,rad/s`，初始臂部目标变化率 `0.5 rad/s`。
- 臂部用固定 PD `Kp=60,Kd=6`，速度目标和前馈为零。固定夹爪与负载连接；第一版无 OCS2/IK。
- 负载复用独立生成的 USD：`0.5 kg, 0.20×0.12×0.08 m`，夹爪局部偏移 `(+0.15,0,0)`。
  启动时检查负载确实成为 articulation link 且导入质量匹配，源机器人 USD 不修改。
- 外力定义为**外部作用于负载 COM，以负载自身坐标表示**。每个高层区间保持常值，施力与 teacher
  所见三维力一致。观测时给出即将执行区间的当前已知力，不给出未来曲线、下次幅值或持续时间。
  后续 estimator 通常有延迟，不拥有这一特权；student 阶段需要显式处理这个差异。
- 三维交互力不转换或重复填入低层 30 维接口。第一版低层 future wrench 为零，并且完全移除原
  synthetic arm-wrench command；真实臂/负载的反作用由物理引擎传递，避免重复施力。

Teacher policy 为 99 维；runner 拼接三维导纳参考后，actor 输入为 102 维：

| 顺序 | 宽度 | 内容 |
|---|---:|---|
| 1 | 3 | 当前 Fx/Fy/Mz，分别除以 5 N / 5 N / 0.5 Nm |
| 2 | 9 | 底座线速度、角速度×0.25、重力投影 |
| 3 | 72 | 四帧臂部历史，每帧 q−nominal(6)、dq×0.1(6)、前一区间已执行 q_target−nominal(6) |
| 4 | 3 | 负载相对底座位置，机身坐标 |
| 5 | 3 | 负载局部重力投影 |
| 6 | 9 | 实际限幅/限速后的高层目标，按动作尺度归一化 |
| 7 | 3 | 有状态的负载局部导纳速度参考，来自 runner 的 commands group |

Critic 另加负载 COM 的局部线/角速度及世界高度，含 commands 共 109 维。
低层历史、臂部历史和导纳状态都按环境独立复位，不跨 episode；重复读观测不会重复推进历史。
为兼容现有 PPO，`obsHistory` 是一维占位，teacher encoder 输出宽度为零，不产生 latent，也没有额外
回归优化器。真实臂部历史直接进入 teacher actor，**不是新训练的 intent estimator**。

## 目标、奖励与终止

顺应参考使用 `M dv/dt + D v = wrench` 的精确离散更新，三轴初始参数：
`M=(4,4,0.5), D=(20,20,2)`，参考速度上限 `(0.35,0.20,0.40)`。
这些是虚拟导纳参数，不是识别得到的机器人惯量；后续根据期望响应调整。
跟踪奖励比较负载实际局部平面速度/局部 z 角速度与参考，不能仅评价输出的底座指令。

奖励覆盖顺应跟随、负载倾角与高度、竖直运动、底座倾斜和 roll/pitch 角速度、实际目标变化、
臂部限位/速度/力矩，以及零力时回归名义臂姿。负载倾角参考是 episode 初始 carry 姿态，允许
平面移动和 yaw 跟随，不要求固定世界水平位置。当前失败处罚权重为 -100；Isaac 的 reward manager
按高层 dt 缩放，因此一次失败实际贡献约 -4，需在正式训练中结合回报量级调节。

终止：底座倾斜超过 35°、底座高度低于 0.45 m、指定底座/大腿 link 非法接触、负载相对初始
carry 姿态倾斜超过 45°。20 s 是正常 timeout，与失败分开统计。臂部自碰撞仍由资产物理处理；
第一版未实现单独的臂部自碰撞分类处罚/终止，需要后续根据 contact 日志扩展。

每个环境先零速/名义臂姿稳定 2 s，再施力。20% 为零力回合，其他从正负单轴逐渐扩展到混合轴，
使用梯形和半正弦，活跃时长 2–4 s、采样周期 5–7 s，其余时间用于释放/恢复。
训练初始幅值是最大值的 20%，25,000 个高层步线性增加到最大值，组合轴比例同步增加。
第一版固定负载、固定 PD、关闭域随机化和 actuator delay；后续加随机化前先验证冻结低层的适用范围。

## Linux 启动与验收

在已安装 Isaac Sim 的现有 `limx` 环境使用该专用入口，普通 `train.py` 不负责生成 teacher 的负载 USD。
保留服务器现有文件；同步本次代码后执行：

```bash
conda activate limx
cd ~/workspace/isaaclab_tron2/TRON2_WholeBody_LAB
python -m pip install -e exts/bipedal_locomotion -e rsl_rl
export PAINT_CHECKPOINT="$PWD/logs/rsl_rl/sfyg_tron_2a_wholebody_flat/2026-09-10_22-16-15/model_20000.pt"
test -f "$PAINT_CHECKPOINT"
python -m unittest discover -s tests -p test_paint_teacher.py -v
```

先检查 4 环境的加载、复位与数值，零力条件下确认冻结低层能承载这个负载：

```bash
python scripts/rsl_rl/train_paint_teacher.py \
  --mode smoke --checkpoint_path "$PAINT_CHECKPOINT" \
  --num_envs 4 --steps 500 --zero_force --headless --device cuda:0
```

再执行完整幅值的固定臂部基线。`baseline` 输出始终为零速和名义臂姿；即使有失败也会完成记录，
**不是自动判定通过**。查看输出目录 `evaluation.json`、`rollout.jsonl`：

```bash
python scripts/rsl_rl/train_paint_teacher.py \
  --mode baseline --checkpoint_path "$PAINT_CHECKPOINT" \
  --num_envs 4 --steps 1000 --headless --device cuda:0
```

零力基线稳定后，先做两轮小规模 PPO 接口测试，再正式训练：

```bash
python scripts/rsl_rl/train_paint_teacher.py \
  --checkpoint_path "$PAINT_CHECKPOINT" \
  --num_envs 16 --max_iterations 2 --headless --device cuda:0

python scripts/rsl_rl/train_paint_teacher.py \
  --checkpoint_path "$PAINT_CHECKPOINT" \
  --num_envs 1024 --max_iterations 3000 --headless --device cuda:0
```

产物在新建的 `logs/rsl_rl/paint_teacher_flat/<timestamp>`，包括负载 USD、`contract.json`、TensorBoard、
teacher checkpoint 和 `teacher_actor.pt`。低层 checkpoint 只读；训练结束逐参数核验低层权重未改变。
checkpoint 不包含低层权重；contract 和 teacher checkpoint 保存低层文件 SHA256，便于固定来源。
第一版不支持续训，避免将原 runner 的部分加载行为用于高层；评估加载 teacher 权重时为 strict。

用同一个低层 checkpoint 评估 teacher，`TEACHER_CHECKPOINT` 换成刚生成的文件：

```bash
export TEACHER_CHECKPOINT="/absolute/path/to/paint_teacher_run/model_3000.pt"
python scripts/rsl_rl/train_paint_teacher.py \
  --mode play --checkpoint_path "$PAINT_CHECKPOINT" \
  --teacher_checkpoint "$TEACHER_CHECKPOINT" \
  --num_envs 1 --steps 1500 --seed 123 --headless --device cuda:0
```

评估统计失败/timeout、底座和负载峰值倾角、目标变化、有效响应的导纳跟踪 MSE。
`rollout.jsonl` 中记录的是 preceding interval 的力和参考，reset 后的响应标记无效，不拿它当跌倒前状态。
当前 MSE 是三轴未经量纲归一化的诊断值；正式报告应分开报告线速度与角速度误差。
当前随机评估还不是严格相同扰动的 paired benchmark；若比较基线与 teacher，下一步应加入固定测试
force suite，并按真实跟踪误差、释放停止时间、臂部工作空间余量和跌倒率验收。

Teacher 在真实力下稳定后，再采集它主动移动臂部时的历史数据，重训/适配 estimator，进行 student
训练或蒸馏。原 PD hold 数据上的 estimator 精度不代表主动控制下的精度。
