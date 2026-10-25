# PAINT guide 底层训练 v4

2026-10-25 双足重建：完整 SFYG + 实体负载的后续训练请使用显式 `--design biped`，见 [双足学走设计与训练流程](locomotion_v4_biped.md)。下面的旧配置仍对应 `--design baseline`，用于复查和对照；两者的 checkpoint 输入不兼容。

2026-10-25 更新：默认恢复相位步态，新输入为policy123/history360/command3，actor129→10。首次训练与对照安排见 [恢复步态训练计划](locomotion_v4_gait_retrain.md)。旧无步态v4用`--gait legacy`播放/评估；新旧checkpoint严格区分。

分支 `locomotion-v4`，基线 `main@457f0bb`。
参考：https://paint-bot.github.io/docs/locomotion.html

## 与 v2/v3 的区别

- 使用完整 SFYG 机器人、6 轴机械臂、夹爪和实体固定负载。腿策略输出 10 维关节目标增量，机械臂 PD 60/6 保持与 teacher 一致的默认搬运姿态，夹爪保持默认位置。
- 负载安装复用 teacher 的 `paint_payload.py`：`gripper_base_Link`，尺寸 0.20×0.12×0.08 m，夹爪系偏移 (0.15,0,0) m。源 USD 不修改，每次 run 另存 `robot_payload.usdc`。
- 训练中随机负载质量及惯量、底座质量、摩擦，向实体负载质心施加世界水平外力和世界竖直轴力矩。每个物理步重新转换为负载局部坐标并施力；没有合成机械臂底座 wrench。
- actor 不输入未来 wrench，也不输入机械臂关节；critic 可以看到实际随机质量和当前负载外力。
- 新策略从零初始化，旧 v2/v3/StandWalk checkpoint 拒绝加载。不同 v4 阶段可以严格加载同构权重。
- 沿用 Isaac Lab + 本仓库 RSL-RL，而非迁移到 guide 示例使用的 legged_gym。

guide 作者的 ANYmal D 内部 backbone 未公开；这里实现其推荐的完整机器人替代训练方法，不声称复现作者网络或论文 teacher/student RNN。guide 的 12 腿关节、0–10 kg、40 N、10 N·m、±1 m/s 和 ±1.5 rad/s 是四足示例，不直接作为 TRON2 初期参数。

## 任务与接口

任务：`Isaac-Limx-SFYG-TRON2A-Paint-Locomotion-v4`。
入口：`scripts/rsl_rl/train_locomotion_v4.py`；必须使用该入口，在创建环境前准备负载 USD。

- 物理 200 Hz，底层 50 Hz，20 s episode。
- proprio 36 维：机体系角速度×0.25、投影重力、10 个腿关节相对位置、腿关节速度×0.05、上一实际腿动作。
- policy：proprio + 机身周围 0.8×0.8 m、0.1 m 分辨率的 81 点地形扫描 + 6维步态，共123维（legacy117维）。
- history：最近 10 次控制步的 proprio，旧→新排列，10×36；reset 按 Isaac ObservationManager 的初始化规则填充。
- 估计器：360→3，监督目标为 critic 最前面的真实机体系线速度，输出 detach 后送 actor。
- actor 顺序：估计速度(3)、policy(123)、命令(3)，总计129→10（legacy123→10）。
- 关节严格按左腿 pitch/roll/yaw/knee/ankle，然后右腿同序；不依赖 URDF 遍历顺序。
- `q_target = q_default + 0.25 * clip(action,-5,5)`；名义机械臂/夹爪位置保持资产默认值。
- 速度命令默认 vx±0.35、vy±0.20 m/s、wz±0.40 rad/s。平移范数≤0.02、yaw≤0.03 时归零；站立判定有迟滞。
- `locomotion_v4.pt` 是组合 TorchScript，调用 `policy(policy_obs, flattened_history, command)`。调用方负责独立维护每个环境的上一腿动作和 history，并在 reset 时一起重置。

当前旧 teacher 的 `FrozenLocomotion`、MuJoCo v3 加载器仍固定为 78/420 输入，**不能直接加载 v4**。本任务先完成底层训练与原生评估；后续需使用 contract 为 teacher/MuJoCo 写新的适配器，且提供相同高度扫描。不能删掉扫描或填零后宣称等价部署。

## 课程

| stage | 负载 kg | 水平力范数 N | yaw 力矩 N·m | 底座质量倍率 | 摩擦 | 推扰速度 m/s | 地形 |
|---|---|---|---|---|---|---|---|
| nominal | 0.5 | 0 | 0 | 1 | 1 | 0 | 平地 |
| robust | 0.25–1.0 | 5 | 0.5 | 0.9–1.1 | 0.6–1.2 | ±0.05 | 平地 |
| interaction | 0.1–1.5 | 15 | 3 | 0.9–1.1 | 0.6–1.2 | ±0.10 | 平地 |
| rough | 0.1–1.5 | 15 | 3 | 0.9–1.1 | 0.6–1.2 | ±0.10 | 地形课程 |

nominal 关闭观测噪声，其余阶段开启。负载质量每次 reset 采样，同时按名义质量比例重算惯量，重复 reset 不累乘。训练时底座质量及摩擦在 startup 采样；评估时改为每次 reset 采样以覆盖多回合。推扰间隔 5–10 s。

负载外力每 1–3 s 在回合内重采样，25% 区间为释放（零外力）。水平力在圆盘内采样，限制的是合力范数，非每个轴各允许最大值。

固定环境配额约 20% 恒站、40% 移动、20% 站→走→站、20% 走→站→走。环境数非 10 的倍数时有取整差异，正式训练建议≥100；4/32 环境 smoke 不代表完整配额。移动目标每 5–8 s 采样，一半轴向、一半组合，包含正负侧移和纯 yaw。起停采用 1 s smoothstep；不强制所有行走环境先零速 warmup。

地形课程包含平地、坡面（0–0.15 坡度）、台阶（1–8 cm）、离散障碍（1–6 cm），从低难度初始化。沿用按速度进度的地形升级逻辑；零命令环境不应被其地形等级当作运动能力指标。

奖励保留速度跟踪、姿态/高度、脚滑动、动作平滑、关节力矩和限位；默认恢复0.8Hz相位步态奖励0.75，以此替换单脚支撑奖励。站立关闭步态奖励，保留实际速度、双脚接触、腿运动惩罚；膝姿/强default-pose约束仍关闭。legacy模式保留原single-support。35°倾角或离地高度<0.45 m终止，禁止机械臂、夹爪、负载撑地，终止统计在自动reset前捕获。

PPO 初始 LR 3e-4，自适应范围 [1e-5,3e-4]，KL early stop、梯度裁剪；32 步 rollout，初始动作 std=0.6。actor 使用 `5*tanh(x/5)` 均值边界，环境也裁剪至±5。保留动作饱和、速度估计损失及真实 rollout 站立占比日志。所有参数属于 TRON2 工程起点，需要训练验证，非 guide 给定超参数。

## Linux 训练

```bash
conda activate limx
cd ~/workspace/isaaclab_tron2/TRON2_WholeBody_LAB
python -m pip install -e exts/bipedal_locomotion -e rsl_rl
python -m unittest discover -s tests -p test_locomotion_v4.py -v

# 新目录；两轮更新只检查资产、接口、PPO、保存和导出。
LIVESTREAM=0 python -u scripts/rsl_rl/train_locomotion_v4.py \
  --stage nominal --num_envs 32 --max_iterations 2 \
  --output_dir logs/rsl_rl/v4_smoke --headless --device cuda:0

# 正式训练从零开始；1000轮只是首轮诊断窗口，不保证收敛。
export V4_A="logs/rsl_rl/v4_nominal_$(date +%Y%m%d_%H%M%S)"
LIVESTREAM=0 python -u scripts/rsl_rl/train_locomotion_v4.py \
  --stage nominal --num_envs 1000 --max_iterations 1000 \
  --output_dir "$V4_A" --seed 42 --headless --device cuda:0
```

先验收 nominal 候选，再晋级。不要默认最后 checkpoint 最好；定期选多个候选分别评估。首轮通过后，至少用 42/43/44 三个训练 seed 比较可重复性。

```bash
export V4_CKPT="$V4_A/model_1000.pt"
LIVESTREAM=0 python -u scripts/rsl_rl/train_locomotion_v4.py \
  --mode eval --stage nominal --checkpoint_path "$V4_CKPT" \
  --episodes 20 --seed 142 --output_dir "$V4_A/eval_nominal" \
  --headless --device cuda:0

export V4_B="logs/rsl_rl/v4_robust_$(date +%Y%m%d_%H%M%S)"
LIVESTREAM=0 python -u scripts/rsl_rl/train_locomotion_v4.py \
  --stage robust --checkpoint_path "$V4_CKPT" \
  --num_envs 1000 --max_iterations 1000 --output_dir "$V4_B" \
  --headless --device cuda:0
```

随后同样以通过验收的 v4 checkpoint 进入 interaction、rough。默认阶段加载权重但重新初始化两个优化器和 iteration；不会继承 v3 权重，也不冻结旧 encoder。

中断续训用 `--resume --checkpoint_path ...`，恢复两个优化器及 iteration，要求物理设置和速度范围一致。`--max_iterations` 是额外轮数。仿真 episode 和 RNG 轨迹重新开始，不保证逐位重现中断前 rollout。修改课程时不要使用 `--resume`。

可用 `--payload_mass_range MIN MAX --force_n F --yaw_torque_nm T --velocity_limits VX VY WZ` 扩展范围。实体刚体质量必须>0，零负载极限用很小正值近似，并明确其仍有负载碰撞几何。若要评估真正无负载机器人，需要额外构造无负载场景，不可把微小质量当完全移除负载。

## 评估与验收

每个案例默认 20 回合：站、退、低速、前进、左右、左右 yaw、起停两方向。逐步写 `evaluation_steps.jsonl`，汇总写 `evaluation.json`。固定 seed 跨案例复位，保留终止前状态。已完成回合不等于通过。

初始平地gate要求完整20s、最大倾角<15°、平面/yaw跟踪RMSE<0.1、站立速度/yaw RMS<0.05；另要求移动稳态有向进展比0.7～1.3，停止必须先观测到实际运动且停止时间<2s。新`acceptance_version=v4_motion_progress_v1`不能与旧pass直接混用。站立/移动各忽略段首1s；漂移报告但尚未作为gate。粗糙地形阈值仅供严格对照，尚未经验证。

必须分开跑：

1. nominal 名义0.5 kg、零外力，作为每个候选的基础回归。
2. 固定质量，如 `--stage nominal --payload_mass_range 0.25 0.25`，分别测0.25/0.5/1.0 kg。
3. interaction 的随机负载及回合内拉力；再固定质量覆盖外力上限。
4. rough 用 `--terrain_level 0`、`5`、`9` 分别评估不同难度。每条命令输出到新目录。

例如：

```bash
LIVESTREAM=0 python -u scripts/rsl_rl/train_locomotion_v4.py \
  --mode eval --stage interaction --checkpoint_path "$V4_CKPT" \
  --payload_mass_range 0.5 0.5 --force_n 15 --yaw_torque_nm 3 \
  --episodes 20 --seed 242 --output_dir logs/rsl_rl/v4_eval_pull \
  --headless --device cuda:0
```

目前外力评估使用固定 seed 的随机方向、持续时间和释放区间；它不是确定性逐方向受力极限扫描，也不能仅凭这组结果宣称全方向上限验收。

每次 run 保存独立 USD、`contract.json`、训练 checkpoint、TensorBoard、组合 `locomotion_v4.pt`，以及 v4 源码快照与 SHA256。checkpoint 包含 v4 schema 和形状，严格加载；导出属于该次训练最终模型，评估候选会另行导出相应模型，不自动标 best。评估汇总逐回合记录实际负载质量、总质量及实际施力峰值。

底层通过原生验证后，才适配 teacher/MuJoCo，先做零力→小拉力→完整交互；高层训练期间冻结底层。固定臂姿训练尚不覆盖高层大幅臂运动，需另做臂姿/运动范围测试或扩展课程。

## 本地验证范围

CPU 测试覆盖真实 PPO 更新、速度估计器、TorchScript 保存/加载一致性、checkpoint 拒绝旧接口、固定环境配额/起停复位、逐物理步外力坐标变换、负载质量/惯量非累乘更新，以及高度扫描参考点。测试不证明机器人已经学会站立或带载行走；正式收敛与稳定性必须由 Isaac Sim 训练及上述评估确认。

2026-10-23 本机验证（RTX 4070 Laptop / Isaac Sim 4.5）：

- v4 新增 9 项测试与 StandWalk 13 项回归测试全部通过。
- nominal、robust、rough 均完成 4 环境、2 次真实 PPO 更新；rough 同时运行地形课程，保存 model_1/model_2 的 iteration 为1/2。
- rough 原生评估完成 stand/walk 两个案例并写出 `evaluation.json`；提前跌倒被记录为失败，未误当 timeout 通过。修复跨回合复位的 inference tensor 问题。
- 最终 smoke checkpoint 的 TorchScript 与原生 actor+encoder 在随机输入上的最大输出差为0。
- 原生训练产物位于 `logs/rsl_rl/v4_native_smoke`、`v4_native_robust_smoke`、`v4_native_final_smoke`；粗糙地形评估位于 `v4_native_rough_eval_smoke`。这些目录仅为本地检查产物，不是已收敛策略，评估 `all_passed=false`。尚未启动正式长训。
