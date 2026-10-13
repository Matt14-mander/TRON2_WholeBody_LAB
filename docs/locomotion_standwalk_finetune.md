# WholeBody 低层微调：站立、行走与起停

## 为什么先补低层

2026-10-13 用户服务器回传：旧 `model_20000.pt` 在零速、零外力下，无负载约 0.34 s、
有负载约 0.36 s 达到倾角限制；teacher 同样在 2 s warmup 结束前失败。
改用 `[-0.25,0,0]` 后，无负载/0.5 kg 负载均完成 3 个 20 s 回合，最大倾角分别约 7.20°/7.56°。
因此本轮训练低层的零速支撑与起停，保留已有行走能力；teacher 暂不扩大训练。

开发分支为 `codex/locomotion-standwalk`。本地已做 Torch 单测、真实 PPO 更新检查和语法检查，
**没有 Isaac Sim runtime，也没有正式 checkpoint，尚未验证 PhysX 中的微调效果**。
以下服务器命令需要先同步本次新增代码。

## 实现与兼容性

独立任务 `Isaac-Limx-SFYG-TRON2A-WholeBody-StandWalk-Finetune-v0`，入口
`scripts/rsl_rl/finetune_standwalk.py`。原 WholeBody 和 teacher 任务继续保留。

- actor、critic 从旧 checkpoint 初始化；actor 输入保留 78 维，腿部 action 保留 10 维。
- history encoder 保留 420→3，默认冻结并关闭它的独立监督优化器；`--train_encoder` 才允许以 1e-5 更新。
- 低层控制保持 50 Hz、物理 200 Hz，固定名义臂姿、PD 60/6，夹爪固定负载连接。
- 默认无负载；`--payload_mode attached` 添加与 teacher 相同的 0.5 kg 负载，检查导入 link 和质量。
- 当前阶段 future wrench 为零，合成底座力、扰动和强随机化关闭。这不是原 wrench 补偿训练的全面验收；
  稳定后需另做 wrench 能力回归与恢复课程，不能宣称预测 wrench 性能已保留。
- asset/default joint pose、观测缩放、历史长度/顺序及 actor 输入顺序保持原接口。
- 复位为名义机器人状态；初期关闭 actuator delay 和观测噪声，以对齐已验证的行走基线。

所有 actor/logstd、encoder 和 critic 参数先完整验证，再加载。仅允许负载增减引起的 **critic 第一层
输入宽度 ±1**：显式重建该层 weight/bias，严格加载其余 critic，记录于 `load_report`。
其它形状变化或缺失参数均报错，不会静默跳过。新增 body mass 改变了后续输入列位置，所以不盲目复制
第一层旧输入列。源 checkpoint 不修改，SHA256 和源 iteration 写入 contract。

优化器及 iteration 计数不从源 checkpoint 恢复；每个阶段是新的 warm-start 微调 run。
默认 PPO learning rate 为固定 1e-4、entropy 0.005，显式设置动作噪声 std=0.2。
这是必要的：现有 ActorCritic 不读取 init_noise_std，仅改配置不能改变实际探索噪声。

## 命令课程与站立模式

| `--stage` | 每个 episode 的分布 | 用途 |
|---|---|---|
| `stand` | 60% 恒零速、40% 恒 `[-0.25,0,0]` | 学站立并维持旧行走基线 |
| `transition` | 20% 恒零、20% 恒行走、30% 静止→行走→静止、30% 行走→静止→行走 | 学起停 |
| `expanded` | 同 transition 的类型分布，移动目标一半保留 -0.25，其余在 vx±0.35/vy±0.20/wz±0.40 中采样 | 覆盖 teacher 速度范围 |

默认 20 s episode。转换序列为：首段 5 s，过渡 1 s，中间段 6 s，再过渡 1 s，其余为末段。
过渡采用 cubic smoothstep；`--transition_s 0.5` 可在稳定后缩短，默认不做瞬时跳变。
每次 reset 按环境重新采样类型，课程时钟由 episode time 定义，不跨 episode。
站立学习回合从零速开始，行走回合从已验证速度开始，没有强制所有回合先经过一个不稳定的零速 warmup。

站立 flag 从实际命令判定，进入阈值为平移≤0.02 m/s、yaw≤0.03 rad/s；
离开阈值为平移≥0.04 m/s 或 yaw≥0.06 rad/s，使用迟滞避免切换抖动。
站立时关闭 gait 奖励，并使用站立专属约束。步态时钟仍连续推进：观测与奖励都使用
`episode_time * frequency`，不新增观测通道，也不在站立时将频率设为零。
Teacher 原手写低层观测使用相同 episode clock；加载新 actor/encoder 不需要改 78/420/10 接口。

## 奖励与终止

保留速度跟踪、机身高度/倾角、脚底滑动、动作平滑、腿部力矩/限位等原奖励。
新增零速下的实际平移速度、yaw 速度、腿关节速度、双脚接触缺失惩罚。
移除本任务的强 default-pose `stand_still` 惩罚，将原膝部姿态惩罚限制到行走，避免压制站立平衡姿态。
修复共用 `stand_still_for_joints` 忽略 joint_ids 的问题；其他任务仍能使用该函数，现会正确遵循关节筛选。

新失败处罚权重 -250，在 50 Hz 下约为一次 -5；与原 alive/速度跟踪项共同作用，后续按回报量级调整。
增加机身倾角>35°或相对地面高度<0.45 m 的跌倒终止，保留原非法接触终止。20 s timeout 正常结束。
不通过提高倾角阈值把失败变成成功。

## Linux：先验证加载与 PPO，再训练

```bash
conda activate limx
cd /home/matt14/workspace/isaaclab_tron2/TRON2_WholeBody_LAB
python -m pip install -e exts/bipedal_locomotion -e rsl_rl
export PAINT_CHECKPOINT="$PWD/logs/rsl_rl/sfyg_tron_2a_wholebody_flat/2026-09-10_22-16-15/model_20000.pt"
test -f "$PAINT_CHECKPOINT"
python -m unittest discover -s tests -p test_standwalk.py -v
```

先做小规模两轮更新，确认启动日志为 `encoder_frozen=True`、学习率 1e-4，以及原接口尺寸匹配：

```bash
LIVESTREAM=0 python scripts/rsl_rl/finetune_standwalk.py \
  --checkpoint_path "$PAINT_CHECKPOINT" --stage stand --payload_mode none \
  --num_envs 32 --max_iterations 2 --headless --device cuda:0
```

两轮只证明接口/PPO 能执行，不能证明站立已学会。接着训练阶段 A，输出指定为新目录：

```bash
export STAND_RUN="$PWD/logs/rsl_rl/standwalk_A_$(date +%Y%m%d_%H%M%S)"
LIVESTREAM=0 python scripts/rsl_rl/finetune_standwalk.py \
  --checkpoint_path "$PAINT_CHECKPOINT" --stage stand --payload_mode none \
  --num_envs 512 --max_iterations 1000 --output_dir "$STAND_RUN" \
  --headless --device cuda:0
```

1000 轮是首轮观察窗口，不是保证收敛的预算。TensorBoard 观察 standing_fraction、跌倒、实际速度误差，
然后用固定测试验收。第 A 阶段只要求 stand/walk 通过：

```bash
export STAND_CHECKPOINT="$STAND_RUN/model_1000.pt"
LIVESTREAM=0 python scripts/rsl_rl/finetune_standwalk.py \
  --mode eval --checkpoint_path "$STAND_CHECKPOINT" --payload_mode none \
  --eval_cases stand walk --episodes 20 --seed 142 --headless --device cuda:0
```

评估每份周期 checkpoint 可用同样命令替换路径，人工选通过测试的 checkpoint；
脚本不自动把最后一份或训练回报最高的一份标成 best。

## 阶段 B/C 与带载回归

A 通过后，微调起停：

```bash
export TRANSITION_RUN="$PWD/logs/rsl_rl/standwalk_B_$(date +%Y%m%d_%H%M%S)"
LIVESTREAM=0 python scripts/rsl_rl/finetune_standwalk.py \
  --checkpoint_path "$STAND_CHECKPOINT" --stage transition --payload_mode none \
  --num_envs 512 --max_iterations 1000 --output_dir "$TRANSITION_RUN" \
  --headless --device cuda:0
export TRANSITION_CHECKPOINT="$TRANSITION_RUN/model_1000.pt"
```

用同一候选分别评估无负载/有负载，四种基本案例各 20 回合：

```bash
for MODE in none attached; do
  LIVESTREAM=0 python scripts/rsl_rl/finetune_standwalk.py \
    --mode eval --checkpoint_path "$TRANSITION_CHECKPOINT" --payload_mode "$MODE" \
    --episodes 20 --seed 242 --headless --device cuda:0
done
```

若无负载通过而有负载失败，从通过的无负载 checkpoint 加入负载继续 transition 阶段。
若都通过，可进入带载 expanded 阶段，训练参数与前述一致，替换为 `--stage expanded --payload_mode attached`。
第一层 critic 重建会在启动日志及 contract 中明确记录，actor/encoder 仍严格继承。
最后补测低速、前进、横移与转向：

```bash
# CANDIDATE_CHECKPOINT 换成已生成且准备验收的候选模型。
LIVESTREAM=0 python scripts/rsl_rl/finetune_standwalk.py \
  --mode eval --checkpoint_path "$CANDIDATE_CHECKPOINT" --payload_mode attached \
  --eval_cases stand walk start_stop stop_start slow forward left right yaw_left yaw_right \
  --episodes 20 --seed 342 --headless --device cuda:0
```

## 评估产物与回到 teacher 的条件

每个 run 保存独立机器人 USD、`contract.json`；训练另保存 checkpoint、TensorBoard、TorchScript
`policy.pt`/`encoder.pt`。评估保存 `evaluation.json`、逐回合 `evaluation_episodes.jsonl` 和逐步
`evaluation_steps.jsonl`。评估通过与否由 `all_passed` 表示，`complete` 日志仅表示流程执行完毕。

原生环境会在自动复位前保存 terminal 速度、位置、高度和倾角，终止步不会被 reset 状态替代。
停止时间从下发命令进入零速阈值开始，直到实际水平速度/yaw 均<0.05；站立 RMS 忽略每个零速段最初 1 s。
分别记录每段站立的最大漂移；漂移仅报告，当前不作为自动 gate。

初始 gate：完整 20 s timeout、最大倾角<15°、平面速度跟踪 RMSE<0.1 m/s、yaw 跟踪 RMSE<0.1 rad/s；含站立的案例还要求
站立实际速度 RMS<0.05 m/s、yaw RMS<0.05 rad/s；起停案例要求停止时间<2 s。
这些是初始工程门槛，需要结合硬件响应调整，但不能把提前跌倒当完成。
转向案例同时检查 yaw 跟踪误差，不能以平移速度误差为零替代转向验收。

只有无负载与带载的基本站立/行走/起停案例均通过，才冻结候选模型。
之后用新 checkpoint 跑 teacher 的零力 smoke，再小幅施力、少量 PPO，最后扩大 teacher 训练。
Teacher 不需要 estimator；后续主动臂部运动与受力的低层适应性仍需独立验证。
