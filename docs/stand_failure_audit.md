# 零速失败：臂部控制与 wrench 训练的归因检查

## 已经知道的事实

用户服务器 SF 示例 ONNX 在 MuJoCo 中：2 s PD 准备后，实际 ONNX 零速运行 20 s，
1000 次 policy 推理，RL 阶段最大 IMU 倾角 1.34°。这说明该 SF 示例在这一执行链路能保持直立。
这次没有根部高度、接触或平移速度日志，且不是当前 SFYG 资产，因此不能单凭该结果判定 wrench 训练造成退化。

旧 SFYG WholeBody `model_20000.pt` 在无负载、零外力、立即 RL、连续相位条件下约 0.35 s 失败。
200 轮微调后约 0.42 s 失败；行走仍能完成 20 s，倾角约 3.77°，速度跟踪尚未达标。

## 静态检查

| 项目 | 原 SFYG 锁臂配置 | WholeBody 配置 |
|---|---|---|
| 机械臂控制 | PD lock，Kp=800/Kd=40 | arm_mpc，Kp=60/Kd=6 |
| 臂部 effort/velocity limit | 200 Nm / 10 rad/s | 100 Nm / 5 rad/s |
| arm2/arm3 初始状态 | 0/0，位于 URDF 边界附近 | 1.5708/-1.4835 rad |
| PPO action | 十腿关节 | 十腿关节，仍没有臂部 action |
| reset joints | 限定腿部 | 继承腿部限定 reset |
| solver iterations | 4 position / 0 velocity | 8 position / 2 velocity |
| actor | 48 维（flat） | 78 维，多了 30 维预测 wrench |

“不在 PPO action 中”不等于机械臂运动学被冻结。两个资产都保留真实臂部质量和关节动力学，
PD stiffness 决定姿态跟随强度。原始 WholeBody `train.py` 和任务没有显式向 arm_mpc 写入名义位置目标；
本地 Isaac Lab 的 Articulation target buffer 从零创建，reset_joints_by_offset 只写关节状态。
因此需在服务器记录 `joint_pos_target`：**初始化 q 非零，但目标可能仍是零**。这是代码层面发现的风险，
不是对该服务器旧训练运行的已验证事实。
当前 `collect_paint_pd.py`、teacher、standwalk 明确发送名义臂姿目标，与可能的旧训练条件存在差异。

WholeBody 保留原速度跟踪/平衡奖励，将部分关节惩罚限制到腿部，并增加臂部限位/速度/力矩处罚。
基础站立比例为 0.1；原零速关节惩罚和内部步态相位的问题在 wrench 改造前已经存在。
standwalk 新奖励不代表旧 model_20000 的训练目标，不用微调后的表现反推旧训练原因。

WholeBody 合成 wrench 作用于 `base_Link`：Fx/Fy±80 N、Fz[-120,40] N；Mx/My±30 Nm、Mz±20 Nm，
另有加速度相关未观测扰动上限 40 N/10 Nm。这与真实负载 COM 的交互力不是同一含义。
它可能改变最优步态与静止难度，实际训练 run 的参数需以该 run 配置为准。
actor 增加预测输入、网络宽度变化、资产和臂部控制变化同时发生，现有两个 checkpoint 不构成单变量因果实验。

SF SDK 示例零速将 gait_clock 固定为 `[0,1]`，并使用 2 s 腿部 PD 准备。
此前 Isaac 的观测相位按 episode time 推进，且从复位立即进入 RL。
SF/MuJoCo 与 SFYG/Isaac 还存在引擎、接触、形态和模型训练来源差异。

## 先用同一个旧 checkpoint 做执行条件对照

新增 `scripts/rsl_rl/probe_wholebody_support.py`，使用旧 WholeBody PLAY 资产，严格加载旧模型和 encoder，
无额外负载、合成 wrench 全零。十个腿部动作和原网络输入宽度不变。
`--arm_mode source` 不写臂目标，记录训练入口类似的 target buffer 行为；`hold` 显式保持名义姿态；
`stiff` 在同一名义姿态将 Kp/Kd 改为 800/40。stiff 保留 100 Nm/5 rad/s 限制，故仅隔离刚度与阻尼，
不声称完全复现原 SFYG arm_lock。
诊断记录 q/dq/实际 target/effort、实际 base wrench、终止前机身状态，35°或低于0.45 m终止。
`source` 是执行条件探针，不完全复现历史训练的随机 reset、噪声和 actuator delay。

同步新脚本后，在服务器执行；**使用旧 20000 模型，不用 200/500 轮微调模型**：

```bash
conda activate limx
cd /home/matt14/workspace/isaaclab_tron2/TRON2_WholeBody_LAB
export OLD_POLICY="$PWD/logs/rsl_rl/sfyg_tron_2a_wholebody_flat/2026-09-10_22-16-15/model_20000.pt"
export SUPPORT_AUDIT="$PWD/logs/paint/support_audit_$(date +%Y%m%d_%H%M%S)"
for MODE in source hold stiff; do
  LIVESTREAM=0 python -u scripts/rsl_rl/probe_wholebody_support.py \
    --checkpoint_path "$OLD_POLICY" --arm_mode "$MODE" \
    --phase source --warmup 0 --duration 20 \
    --output_dir "$SUPPORT_AUDIT/$MODE" --headless --device cuda:0
done
```

source/hold 差异说明目标或随之改变的臂姿有影响；hold/stiff 差异说明顺应刚度有影响。
这些是旧 checkpoint 对执行条件的敏感性，不能证明其训练时某个因素造成了能力退化。

再固定同一 hold 臂部控制，拆开相位和 PD 启动效应：

```bash
for PHASE in source frozen; do
  for PREP in 0 2; do
    LIVESTREAM=0 python -u scripts/rsl_rl/probe_wholebody_support.py \
      --checkpoint_path "$OLD_POLICY" --arm_mode hold --phase "$PHASE" \
      --warmup "$PREP" --duration 20 \
      --output_dir "$SUPPORT_AUDIT/hold_${PHASE}_prep${PREP}" --headless --device cuda:0
  done
done
```

PD 准备本身失败则结果标记 `failed_during=PD`，不能当作 RL 站立失败。
`result.json` 报告真正的 RL 秒数和 RL 峰值倾角；`steps.jsonl` 保存各项物理数据。
本机无 Isaac Sim runtime，只有语法/CLI 检查；这些探针尚未在服务器运行。

## 执行条件排除后，再判断训练外力的因果影响

需要至少两组匹配训练：同一 SFYG 资产、同一名义臂姿、相同 PD/limits、网络和输入尺寸、
站立课程、奖励、预算及多个 seeds。

1. 训练时不施加 synthetic wrench，预测输入保持零。
2. 训练时施加既定 synthetic wrench，并提供相同契约的预测输入。

两组统一在零速零力，以及受力场景评估，才能判断加入该 wrench 训练方案是否退化站立。
若要单独区分外力与预测观测，再加“施力但不提供预测”的第三组。
不能仅因零力下锁臂仍失败，就断言“wrench 训练破坏了站立”；训练质量、姿态、相位和模型来源仍可能影响结果。
