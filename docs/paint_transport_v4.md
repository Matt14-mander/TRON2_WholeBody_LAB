# PAINT transport environment v4

分支：`paint-transport-v4`，从 `locomotion-v4@4f923e8` 创建。
选择性迁入 `paint-teacher-guide@3136606` 的高层历史、受力坐标系、直接速度参考基线和 PPO 组织方式；不整体合并旧 teacher，不修改原 v4 训练任务。

参考：[PAINT Transport Environment](https://paint-bot.github.io/docs/environment.html)。本实现使用 Isaac Lab / Isaac Sim，不迁移到指南示例中的 legged_gym / Isaac Gym。

## 任务与当前交付边界

- `Isaac-Limx-SFYG-TRON2A-Paint-Transport-Teacher-v4`
- `Isaac-Limx-SFYG-TRON2A-Paint-Transport-Student-v4`
- 专用入口 `scripts/rsl_rl/train_transport_v4.py`，在 spawn 前生成独立负载 USD。
- 一个环境实现，两种 actor 观测；提供冻结 v4 低层、teacher PPO、固定臂基线、直接参考基线、低速探针、固定受力评估、可视化和日志。
- Student 环境已可构建和运行，当前不实现 student estimator / distillation 算法；专用入口禁止 student PPO 训练。未来算法应拼接 student144 与估计 wrench3，不能用真实 wrench 替代估计结果宣称 student 验收。
- 现阶段验证目标是环境与接口正确，不是已完成搬运策略训练。高层正式训练应等待低层名义站立、多方向、起停及带载/受力回归通过。

## Simulation Settings 与机器人

| 项目 | 本实现 |
|---|---|
| 物理时间步 | 0.005 s / 200 Hz |
| 高层与冻结低层 | decimation=4，均为50 Hz；低层每控制步推理一次 |
| episode | 10 s，500个控制步 |
| 机器人 | 完整 SFYG TRON2A，10腿关节＋6臂关节＋夹爪 |
| 负载 | 复用 v4/teacher 固定连接的独立刚体，箱体0.20×0.12×0.08 m，夹爪系偏移(0.15,0,0) m |
| 机械臂 PD | 60/6，与 v4 一致；guide 四足例子为50/4 |
| 自碰撞 | 开启；过滤负载与相邻夹爪链接之间的碰撞 |
| 地形 | flat 或 v4 相同的坡面/台阶/离散障碍课程 |

箱体没有额外建模两个把手或实体伙伴；人类伙伴通过负载上的 wrench 表示。保持现有负载质量、安装位置和碰撞几何，便于对齐正在训练的 v4。

## 冻结 v4 的严格适配

支持两种输入：

1. v4 `model_N.pt`：从 checkpoint 的 v4 contract 读取网络结构，严格加载 actor/速度估计器。
2. v4 `locomotion_v4.pt`：加载组合 TorchScript，并读取同目录 `contract.json`；可通过 `--low_manifest` 指定文件。

旧 v2/v3/StandWalk checkpoint 拒绝加载。低层要求 policy117、history10×36、command3、action10，关节顺序、动作比例、控制频率、arm PD 均核对。

低层历史在独立缓冲中每控制步采样一次，reset 后首次帧重复填满10帧；它使用自己上一实际腿动作，绝不使用九维高层动作。机身周围81点高度扫描与 v4 相同，使用 `base_z - terrain_z - 0.5` 并裁剪到[-1,1]。部署推理关闭低层观测噪声，与 v4 原生 eval 一致。

所有低层参数 `requires_grad=False`，推理使用 no_grad，PPO 优化器不包含它；运行前后核验完整参数 fingerprint，contract 记录低层文件 SHA256。高层 checkpoint 固定绑定该低层文件；不可静默切换底层。

当前 v4 训练时机械臂保持固定姿态，高层改变臂姿会改变重心/惯量。因此动作接口开发完成不等于新臂姿已验证；先小范围测试，再扩展低层臂姿/运动鲁棒性。

## Action space

九维有界动作：

- 前3维直接表示 m/s、m/s、rad/s，不再乘旧 teacher 的 velocity_scale。按底层训练范围限幅；平移范数≤0.02、yaw≤0.03归零。
- 后6维：`q_arm_target = q_nominal + arm_scale * action[3:9]`，默认 `arm_scale=0.1 rad`，经过软限位和1 rad/s目标限速。guide 原例为0.5 rad；不在 TRON2 首轮直接照搬。
- 冻结底层输出10腿动作，按 `q_leg_target = q_default + 0.25 * action` 执行；腿/臂目标在4个物理步内保持。
- 使用与 v4 一致的纯臂 PD，不迁入旧 v3 teacher 的准静态支撑 wrench / 重力前馈，也不对底座重复施力。

`hold` 控制器输出零速/名义臂姿；`reference` 将当前真实 `c(t)` 直接作为底座速度命令并固定臂；`teacher` 使用训练好的高层策略；`probe` 使用显式恒定速度、固定臂和严格零外力。

## Interaction wrench generation

每个 episode reset 采样一次三维目标，默认各分量实际峰值限制为(3 N,3 N,0.5 N·m)，2%回合主交互力为零。10秒内采用10%上升、80%保持、10%下降；不在回合中更换目标。

`c(t) = profile(t) * target / (40,40,10)` 同时用于 teacher 特权观测、未来 estimator 监督标签和底座速度跟踪参考。`max_wrench` 是每轴界限，故水平合力最大为两个轴界限的平方和开根。入口检查 `max_wrench/(40,40,10)` 不超过低层速度范围。

主交互力定义在底座系，在每个物理步按当前姿态转换到底座→世界→负载局部，施加在实体负载 COM。即使主力在控制步内保持，仍每个物理步重新旋转并施加。

`--disturbance_scale 1` 添加 guide 的周期末端扰动：世界系 `Fz=2sin(2πt)` N、`Tx=sin(2πt+π/4)`、`Ty=sin(2πt+π/2)` N·m，施加在 `gripper_base_Link`。默认0，便于先隔离验证主交互。`--zero_force` 同时关闭主交互和该扰动。普通 `zero` 案例仅表示主交互为零，若开启周期扰动它仍然存在。

低层 v4 的扰动力来自世界水平系，本环境按 guide 使用底座系；倾斜后会产生不同的世界竖直分量，需要作为集成回归内容。力范围、质量和臂姿超出底层训练分布时，不应仅靠延长高层 PPO 弥补。

## Observations

| group | 内容 | 宽度 |
|---|---|---|
| student | 8帧(q_arm−q_nominal、原单位dq_arm、前一区间已执行q_target−q_nominal)，旧→新 | 144 |
| teacher | 干净臂历史144＋当前真实c(t)3 | 147 |
| policy | teacher 或 student，依任务选择 | 147/144 |
| critic | teacher147＋末端/负载各15维状态 | 177 |
| commands | 不额外拼接导纳参考 | 0 |
| obsHistory | 兼容本仓库 runner 的占位；高层估计器输出0维 | 1 |

每个刚体的15维critic状态是底座系表达的相对位置3、COM线速度3、角速度3、COM线加速度3、角加速度3；速度/加速度是世界惯性量旋转到当前底座轴表达，不是对相对位置做时间微分。没有额外观测归一化。

Student 的关节位置加 uniform±0.01 rad、关节速度加 uniform±1.5 rad/s，上一目标不加噪声；teacher历史干净。高层历史零填充复位，重复读观测不会推进历史。每个环境独立复位。环境同时返回两个group，`extras.teacher_obs` 供后续算法查询teacher；student的 `policy` 不含真实力。

## Rewards、终止、地形课程

底座 xy 速度与 `c_xy(t)`、yaw rate与 `c_z(t)`分别做指数跟踪，sigma=0.25；另有负载底座系高度、关节力矩、关节速度/加速度、执行动作变化/幅值和机械臂偏离名义姿态的惩罚。

Guide的负载高度区间[0.05,0.25] m不能直接照搬：本机TRON2名义搬运姿态的负载COM约在底座上方0.54m。默认在初始reset后的名义姿态测量一次，使用名义高度±0.1m的固定区间，记录在contract；不是每个episode重新校准，也不随跌倒重设目标。可用 `--payload_height_range MIN MAX` 显式指定绝对底座系区间。标定发生在训练/诊断开始前，nominal/reference/teacher对比应使用相同高度区间。

Guide 的表格把 `exp(-error)` 跟踪项列成负权重，与最大化奖励的 PPO 方向冲突；本实现明确选用正权重0.02/0.01，而非盲目复制负号。其余表格起点：高度-0.003、力矩-2e-9、DoF运动-2e-4、动作-2e-4、臂姿-0.003；Isaac reward manager乘dt一次。失败权重-2500，对应每次失败-50。动作惩罚使用执行后的物理底座命令及按arm_scale归一化的臂目标；相对guide原始动作存在显式适配。以上权重仍需短训和量级检查。

保留TRON2工程终止：机身/大腿非法触地，倾角>35°，相对地形高度<0.45 m，机械臂/夹爪/负载触地；10s timeout单独统计并bootstrap。终止前状态在自动reset前捕获，失败与timeout同一步发生也不能判为成功。

粗糙地形训练按真实 wrench 速度参考积分形成世界系预期位移；实际位移在该方向上的进度与参考路程比较。非失败回合且进度>80%可升级，失败或<50%降级；零参考、纯yaw或参考方向积分抵消的回合不升级。这里是对guide原则的TRON2工程实现，不声称逐行复现legged_gym课程。

`--domain nominal` 固定初始状态、名义摩擦；质量仍可用参数明确指定。`randomized` 添加底座质量[-2,5]kg、摩擦[0.05,1.5]、reset xy±1m/yaw±π和线角速度±0.5。负载质量与惯量每reset同步随机化且不累乘；默认为0.5kg，未直接启用guide四足的0–10kg。

## Linux：先跑通，再看窗口

```bash
conda activate limx
cd ~/workspace/isaaclab_tron2/TRON2_WholeBody_LAB
python -m pip install -e exts/bipedal_locomotion -e rsl_rl
python -m unittest discover -s tests -p test_transport_v4.py -v

# 使用已验收的v4候选；中间checkpoint也支持，不必等最终TorchScript导出。
export LOW_CKPT="/absolute/path/to/v4_run/train/model_10000.pt"
export TRANSPORT_RUN="logs/rsl_rl/transport_v4_$(date +%Y%m%d_%H%M%S)"

LIVESTREAM=0 python -u scripts/rsl_rl/train_transport_v4.py \
  --mode smoke --locomotion_path "$LOW_CKPT" --zero_force \
  --num_envs 4 --steps 128 --headless --device cuda:0 \
  --output_dir "$TRANSPORT_RUN/smoke"
```

先核对 `completion.json` 中complete和frozen_weights_verified；这两个字段只说明流程完成及权重未变，不是物理稳定性通过。

有图形显示的机器，不带headless启动：

```bash
LIVESTREAM=0 python -u scripts/rsl_rl/train_transport_v4.py \
  --mode preview --controller reference --locomotion_path "$LOW_CKPT" \
  --num_envs 1 --steps 1500 --debug_vis --realtime --device cuda:0 \
  --output_dir "$TRANSPORT_RUN/preview"
```

窗口中可见完整机器人、负载、地形扫描点；绿色箭头表示底座速度命令，橙色为负载世界系力，蓝色为力矩轴矢量（不是圆弧动画）。`PAINT transport v4 | env 0` 面板显示频率、episode、命令/实际速度、参考力、臂位置/目标、倾角和逐项加权奖励。零向量不画箭头，零力预览时看不到橙色箭头是正常现象。

服务器无桌面可添加 `--video --headless` 录制场景；首次启用RTX可能需要编译shader。入口先做40次纯渲染预热，并检查非空/非全黑画面，避免把黑色占位帧当成功视频。视频是相机画面，不包含整个Isaac UI/HUD。训练时建议关闭debug_vis和video。

日志曲线可离线绘制：

```bash
python scripts/rsl_rl/plot_transport_v4.py "$TRANSPORT_RUN/preview"
```

输出 `transport_diagnostics.png`：负载外力、参考/实际速度、底座命令、机械臂实际/目标、奖励分项；红线标记终止/reset。跨case/episode横轴为拼接控制时间，不伪装成单个连续episode。

## 基线与固定受力评估

```bash
# 名义零力基线，完整10s验收。
LIVESTREAM=0 python -u scripts/rsl_rl/train_transport_v4.py \
  --mode eval --controller hold --locomotion_path "$LOW_CKPT" \
  --zero_force --cases zero --episodes 20 --seed 142 \
  --headless --device cuda:0 --output_dir "$TRANSPORT_RUN/eval_zero"

# 每回合固定受力方向与10/80/10时序；正负平移、转向和零主力。
LIVESTREAM=0 python -u scripts/rsl_rl/train_transport_v4.py \
  --mode eval --controller reference --locomotion_path "$LOW_CKPT" \
  --cases zero forward backward left right yaw_left yaw_right \
  --episodes 20 --seed 142 --headless --device cuda:0 \
  --output_dir "$TRANSPORT_RUN/eval_reference"

# 独立小速度探针，固定臂，严格零物理外力。
LIVESTREAM=0 python -u scripts/rsl_rl/train_transport_v4.py \
  --mode smoke --controller probe --probe_command 0.04 0 0 \
  --zero_force --locomotion_path "$LOW_CKPT" --steps 500 \
  --headless --device cuda:0 --output_dir "$TRANSPORT_RUN/probe"
```

入口保存逐步 `rollout.jsonl` 和逐案例/回合 `evaluation.json`。初始工程gate：完整10s、倾角<15°；有运动参考的轴，其RMSE门槛为 `min(0.1,max(0.01,0.5*reference_RMS))`，零参考轴门槛0.05。避免弱力参考本来小于0.1时，完全不响应的机器人也被判通过。这不是论文验收；smoke/preview永远不标为能力通过。提前跌倒明确记录失败。

`probe` 的跟踪目标是探针命令，物理reference仍为零；日志单独保存tracking_reference和tracking_target。会被v4死区去掉的探针输入直接拒绝。对比hold/reference/teacher时使用同样seed、受力case、地形和质量范围；若发生提前失败，不能把剩余轨迹当严格paired benchmark。

## 高层两轮PPO与teacher评估

```bash
LIVESTREAM=0 python -u scripts/rsl_rl/train_transport_v4.py \
  --mode train --locomotion_path "$LOW_CKPT" --num_envs 16 \
  --max_iterations 2 --save_interval 1 --headless --device cuda:0 \
  --output_dir "$TRANSPORT_RUN/ppo_smoke"

LIVESTREAM=0 python -u scripts/rsl_rl/train_transport_v4.py \
  --mode eval --controller teacher --locomotion_path "$LOW_CKPT" \
  --teacher_checkpoint "$TRANSPORT_RUN/ppo_smoke/model_2.pt" \
  --episodes 1 --headless --device cuda:0 \
  --output_dir "$TRANSPORT_RUN/eval_teacher"
```

两轮模型只验证PPO执行链路。通过稳定性和reference基线后，可增加num_envs/max_iterations做teacher短训。rough可用 `--terrain rough`，随机化与末端扰动需显式开启；先隔离变量再组合。

高层checkpoint含transport schema、固定低层SHA和动作契约，strict加载；`--teacher_checkpoint` 是权重warm-start，重新创建优化器并从iteration0计数，当前不实现高层优化器resume。旧teacher checkpoint不能直接迁入。训练产物另含 `teacher_actor.pt`，只导出高层，不把低层隐式塞进teacher权重。

使用低层TorchScript或查看student环境：

```bash
LIVESTREAM=0 python -u scripts/rsl_rl/train_transport_v4.py \
  --mode smoke --actor_mode student \
  --locomotion_path /absolute/path/to/locomotion_v4.pt --low_format torchscript \
  --zero_force --steps 128 --headless --device cuda:0 \
  --output_dir "$TRANSPORT_RUN/student_env_smoke"
```

每次run保存独立负载USD、contract、源码快照/SHA、完成标记；训练保存checkpoint/TensorBoard，诊断保存逐步日志和评估。正式结论需换成已验收底层后重新运行；本地仅两轮训练的v4 smoke权重不代表可搬运能力。

## 2026-10-23 本地验证

- 新增14项transport测试通过；v4原有9项测试回归通过。覆盖冻结checkpoint/TorchScript数值一致、独立历史/复位、噪声与特权隔离、逐物理步坐标变换、参考课程、PPO更新、动作契约、弱力不响应不能误判通过、timeout与失败同时发生仍算失败、TRON2名义高度区间。
- Isaac Sim中4环境完成零力smoke；rough＋随机化＋周期扰动完成两轮teacher PPO，保存model_1/model_2及teacher_actor。运行结束低层fingerprint一致。
- Student144与v4 TorchScript在rough中完成zero/forward/left/yaw_left四个固定case的评估链路；仍使用未收敛低层，提前跌倒如实记录为失败。
- 保存的teacher checkpoint已重新严格加载运行；调试面板与非零受力箭头代码完成离屏原生运行，MP4经抽帧检查可见完整机器人、蓝色负载和高度扫描点。修复短视频全黑预热帧问题；相机预热不会推进物理episode。
- 最终探针smoke验证严格零物理力、0.04m/s跟踪目标和名义负载高度标定；没有把0.16s检查当作完整10s验收。
- 诊断曲线已生成并检查。人工交互GUI操作、已收敛模型下的搬运效果和student训练尚未验收。

本地检查目录为 `logs/rsl_rl/transport_v4_smoke_01`、`transport_v4_ppo_01`、`transport_v4_student_jit_01`、`transport_v4_video_warmup` 和 `transport_v4_final_probe`；这些均是开发检查产物，不是正式结果。
