# v4：完整 SFYG + 真实负载的双足学走重建

本设计通过 `--design biped` 启用，从零训练。旧 v4 的 `--design baseline` 默认值保留，供旧模型复查。**启动新一轮时必须显式带上 `--design biped`，旧训练命令不会自动变成这个新任务。**

## 结论与证据边界

采用完整 SFYG、真实机械臂惯性、机械臂 PD 和实体负载训练腿策略；不退回拆臂的 SF，不用基座合成 wrench 抵消真实机械臂。将训练顺序改为：先连续交替迈步，再多方向速度跟踪，最后起停，然后才扩展负载与扰动。

用户提供的 model_5000 播放统计是：vx 指令 0.2 m/s，实际均值约 0.00041 m/s，双脚同时接触 100%，左右脚离地事件均为 0。这证明该测试中的策略没有迈步。它不能独立证明 KL、奖励或双足结构中的某一项是唯一根因。

旧线速度核 std=0.5 时，指令 0.2、实际为零也可获得 `exp(-0.2²/0.5²)=0.852` 的跟踪分数；旧步态的卸载核 `exp(-F²/25)` 在半身重量下近乎为零，小幅卸载很难改变分数。同时，原地双支撑容易获得低动作、低关节速度和稳定存活收益。恢复一个步态奖励权重，并没有解决这些信号与早期探索之间的关系。它们是需要实验证伪的退化机制，不能只靠训练时长判断。

之前的 KL 统计也表明独立训练的速度估计器会改变最终策略：actor 更新后 KL 约 0.0286、估计器引入的策略漂移约 0.0159、最终 KL 约 0.0450；不能把估计器 MSE 小当成策略变化小。本实现保护两段更新，但不把 KL 保护当成“会走”的充分条件。

## 方法来源与双足差异

- [PAINT locomotion guide](https://paint-bot.github.io/docs/locomotion.html) 推荐的替代方案就是完整机器人、臂位置控制、腿 RL、真实负载和外扰。其四足实验参数不应直接移植到 TRON2。
- [Ma 等，2022](https://arxiv.org/abs/2201.03871) 的训练简化包含拆除机械臂、用外部 wrench 表示上肢影响。仓库 v3 的 SF 路线与这种简化更接近；这不等于完整 SFYG 不能训练。
- [HOMIE](https://arxiv.org/abs/2502.13013)、[FALCON](https://arxiv.org/abs/2505.06776) 提供了人形下肢控制结合上肢运动/负载的相关证据。这里不是复现它们的完整算法。

双足需要主动完成侧向载荷转移，才能获得可用的单支撑和摆脚经验。本设计保留明确的左右相位、给出平滑卸载目标，以实际足底净空定义抬脚，并延后低速和反复起停。完整上肢动力学与这些双足任务设计可以同时保留。

## 实体、观测与初始化

| 项目 | 新设计 |
|---|---|
| 机器人 | 完整 SFYG、6 轴机械臂、夹爪，全部参与动力学 |
| 首阶段负载 | 夹爪处实体 0.5 kg，质量和惯量均保留；本地总质量约 40.023 kg |
| 臂控制 | 原有 PD 60/6 保持默认搬运姿态；没有锁死 articulation 或取消臂质量 |
| 腿动作 | 原有顺序的 10 维关节目标增量，`q_default + 0.25*clip(action,-5,5)` |
| actor 输入 | 估计线速度3 + policy135 + command3 = 141 |
| policy135 | 腿 proprio36 + 地形81 + 步态6 + 实测臂 q/dq12 |
| critic | 原特权信息、实测臂 q/dq12、两脚按总重量归一化的接触力；运行时173维，另拼命令3 |
| history | 原腿 proprio 的 10×36，估计器仍预测真实机体系线速度 |
| 控制 | 物理200 Hz，策略50 Hz，episode20 s |

臂 q 是绝对位置，dq 缩放0.1；腿 proprio 仍采用原定义。actor 没有直接读取真实接触力或未来外扰；实际接触力用于训练奖励、critic 和评估。

默认机器人 root z=0.85 m 时，本地 USD 首步测得足底悬空约12 cm。新任务从 URDF 关节变换和足部碰撞几何计算出生高度，名义 root z 约0.73563 m，足底初始净空1 cm。不是强行固定基座高度。训练只给腿加入±0.05 rad 初态扰动，臂和夹爪每次恢复默认姿态；root yaw±0.05 rad，水平速度和 yaw 角速度±0.03。评估关闭这些初态扰动。

足底净空由足部碰撞 box 顶点、sphere 中心及半径经过真实足部姿态变换计算。sphere 半径沿世界 z 减去，不能旋转“球底点”代替球体。平地接触诊断与 USD 相符；复杂地形仍只使用足中心附近的扫描高度，尚未实现逐碰撞点地形查询，因此粗糙地形不能直接视为已验收。

本地零动作 PD 诊断：修正高度后接触足底平均绝对误差约8.7e-6 m；机器人仍在1.16 s 前倾触发摔倒，最大臂位置误差约0.208 rad，踝力矩峰值约其限制的64%。这说明默认 PD 姿态不是无需策略补偿的静力平衡；也没有证据支持仅提高力矩上限就能解决。保持当前增益和限幅，由策略学补偿，并持续记录倾角、全身 COM、臂误差和力矩占比。COM 对足几何包围盒仅作诊断，不是接触稳定性证明。

## 分阶段学技能

技能阶段 `--skill_stage` 和物理阶段 `--stage` 是两个维度。

| 技能阶段 | 环境角色配额：恒站/连续走/站走站/走站走 | 移动命令 |
|---|---|---|
| bootstrap | 5/95/0/0% | 每回合恒定，vx绝对值0.20–0.35 m/s，约75%向前、25%向后；vy/wz=0 |
| omni | 10/80/5/5% | 正负前后、侧移、纯yaw与组合；轴向下限0.15/0.10/0.20，受速度上限裁剪 |
| transitions | 20/40/20/20% | 上述范围，并加入约25%的低速目标，最低0.05/0.05/0.08 |

配额按20个环境一组取整，正式运行4096环境足够覆盖；单环境 probe/评估会覆盖指定角色。bootstrap 不在回合中换目标；后两阶段沿用5–8秒重采样。所有阶段 reset 后速度命令用0.5秒 smoothstep 建立，死区规则不变。站走站/走站走使用原有1秒命令平滑，尚未实现按实际落脚完成信号切换的停止控制器。

bootstrap 强制 nominal、固定负载、无外扰/观测噪声，不能先增加载荷扰动难度。先通过 bootstrap→omni→transitions，再考虑物理阶段 nominal→robust→interaction→rough；后者仍需单独验证，不自动晋级。当前实现只覆盖固定搬运姿态，尚未加入随机臂轨迹或上肢任务动作课程。

## 步态、奖励与 PPO

左右相位差0.5、频率0.8 Hz、支撑占比0.6，名义双支撑占周期20%，摆脚**足底**净空峰值0.08 m。reset 后相位先保持0.25秒，随后连续推进；站立时奖励关闭，时钟继续。观测与奖励使用同一个 episode 时钟。

| 信号 | 权重/定义 |
|---|---|
| 线速度跟踪 | +2；命令方向误差std=0.2，正交方向std=0.3，给侧向平衡留余量 |
| yaw跟踪 | +1；std=0.25 |
| 载荷转移 | -1；`mean((Fz/(Mg)-desired_contact/sum(desired_contact))²)`，实际负载归一化，范围受限 |
| 摆脚净空 | -0.5；按0.08 m归一化的摆动参考误差，单脚误差平方上限4 |
| 移动停滞 | -1；命令方向速度不足；静止为1、匹配为0、反向最多2，纯yaw同理 |
| 有效落脚 | 每次+0.1；实际腾空0.08–1 s、足底曾抬高≥0.015 m、在支撑相位落脚 |
| 双脚腾空 | -1；避免用跳跃代替交替支撑 |
| 姿态/平滑 | flat_orientation=-2.5；bootstrap动作一阶/二阶变化=-0.005/-0.002，后续阶段恢复原平滑权重 |

`--gait_weight` 默认1，只缩放载荷转移、摆脚和有效落脚三项。保留原脚滑动、脚姿态、基座高度、力矩、速度、加速度、关节限位和站立约束。保留失败终止奖励-250、倾角35°/高度0.45 m终止，以及禁止手臂/负载接触地面。上述连续奖励由 RewardManager 乘dt；落脚事件已显式抵消dt，避免控制频率改变事件收益。

新线速度核在0.2指令下静止得分为0.368，配合停滞惩罚拉开走与不走的收益。载荷转移从双脚各50%变为60/40时就有改善，不要求先完全抬脚才出现信号。RL 不通过奖励反向传播物理，这里的“连续信号”指可被采样比较的回报变化。

PPO保留原rollout32步、5轮×4 minibatch、GAE与裁剪结构；初始动作std=0.8、entropy=0.01，actor LR初值/上限1e-4，估计器LR=3e-5。

新任务额外启用更新后检查：在固定的最多2048条rollout探针上，actor每步后总KL不能超过0.015；估计器每步后自身累计策略漂移不能超过0.003，且最终总KL仍不能超过0.015。超限回滚该步参数和Adam动量/计数，再降LR并停止这一段更新。不是简单把KL阈值放宽，也不承诺全状态空间的KL界。actor与critic仍共用优化器，actor段停止也会限制critic更新；若有效更新持续不足，应看日志再决定是否分离优化器。

## 启动、评估与晋级

在服务器正确分支更新后先做短测；这一步只检查资产、接口、更新和保存，不判断收敛。

```bash
cd ~/workspace/isaaclab_tron2/TRON2_WholeBody_LAB
conda activate limx
git switch paint-teacher-student
git pull --ff-only origin paint-teacher-student
python -m pip install -e exts/bipedal_locomotion -e rsl_rl

mkdir -p logs/rsl_rl
STAMP=$(date +%Y%m%d_%H%M%S)
LIVESTREAM=0 python -u scripts/rsl_rl/train_locomotion_v4.py \
  --design biped --skill_stage bootstrap --stage nominal \
  --num_envs 64 --max_iterations 8 --save_interval 4 \
  --output_dir "logs/rsl_rl/v4_biped_smoke_$STAMP" --headless --device cuda:0
```

独立的 PD 诊断，无需 checkpoint，不是正式训练的通过门槛；它会记录前倾、力矩、臂误差和真实足底，发生跌倒就结束并标记失败。

```bash
LIVESTREAM=0 python -u scripts/rsl_rl/train_locomotion_v4.py \
  --design biped --mode probe --stage nominal --probe_seconds 5 \
  --output_dir "logs/rsl_rl/v4_biped_pd_$(date +%Y%m%d_%H%M%S)" --headless --device cuda:0
```

首轮bootstrap建议3000轮，保存间隔250；500/1000轮用于查看是否出现卸载、真实离地和有效更新，不是最终收敛期限。不要续训旧 model_5000/20000；新actor输入已经变化。

```bash
export V4_BIPED_RUN="logs/rsl_rl/v4_biped_bootstrap_$(date +%Y%m%d_%H%M%S)"
nohup env LIVESTREAM=0 python -u scripts/rsl_rl/train_locomotion_v4.py \
  --design biped --skill_stage bootstrap --stage nominal \
  --num_envs 4096 --max_iterations 3000 --save_interval 250 --seed 42 \
  --output_dir "$V4_BIPED_RUN" --headless --device cuda:0 \
  > "${V4_BIPED_RUN}.console.log" 2>&1 < /dev/null &
echo "PID=$!  RUN=$V4_BIPED_RUN"
```

选择多个保存点进行相同条件的评估，例如1000/2000/3000，不默认最后一个最好。先验收bootstrap训练过的前后移动，再扩展其他技能。

```bash
export V4_BIPED_CKPT="$V4_BIPED_RUN/model_3000.pt"
export V4_BIPED_EVAL="logs/rsl_rl/v4_biped_bootstrap_eval_$(date +%Y%m%d_%H%M%S)"
LIVESTREAM=0 python -u scripts/rsl_rl/train_locomotion_v4.py \
  --design biped --skill_stage bootstrap --stage nominal --mode eval \
  --checkpoint_path "$V4_BIPED_CKPT" --episodes 20 --eval_cases walk forward \
  --output_dir "$V4_BIPED_EVAL" --headless --device cuda:0

# 单条向前指令录像，诊断用途；远程桌面可去掉 --headless。
LIVESTREAM=0 python -u scripts/rsl_rl/play_locomotion_v4.py \
  --design biped --skill_stage bootstrap --stage nominal \
  --checkpoint_path "$V4_BIPED_CKPT" --base_command 0.25 0 0 \
  --seconds 20 --video --fast --headless \
  --output_dir "logs/rsl_rl/v4_biped_preview_$(date +%Y%m%d_%H%M%S)" --device cuda:0
```

正式评估除了原20秒生存、倾角<15°、线/yaw跟踪RMSE<0.1，还要求命令方向进度比0.7–1.3，并满足真实迈步条件：左右各≥2次有效落脚、至少3次左右交替、移动期间单支撑占比0.1–0.9、接触脚平均水平速度<0.1 m/s。原地不动、只切换接触标志但无净空、滑行，都不能靠RMSE或存活时间绕过验收。静止案例不要求迈步；起停还要求站立速度RMS<0.05，观察到运动后停止延迟<2秒。原RMSE/倾角阈值保留，是严格候选门槛；若后续出现正常双足摆动却被RMSE拒绝，应先检查完整时序，再另外评审周期平均指标，不能事后悄悄降低门槛。

晋级要求同一 checkpoint 的 SHA256 匹配评估报告；每个要求案例至少10次、通过率≥90%。bootstrap→omni要求walk/forward；omni→transitions还要求left/right/yaw_left/yaw_right。晋级只加载权重，重建优化器；同阶段 `--resume` 才恢复优化器与轮数，物理和目标必须一致。

```bash
export V4_OMNI_RUN="logs/rsl_rl/v4_biped_omni_$(date +%Y%m%d_%H%M%S)"
nohup env LIVESTREAM=0 python -u scripts/rsl_rl/train_locomotion_v4.py \
  --design biped --skill_stage omni --stage nominal \
  --checkpoint_path "$V4_BIPED_CKPT" --promotion_report "$V4_BIPED_EVAL/evaluation.json" \
  --num_envs 4096 --max_iterations 3000 --save_interval 250 --seed 42 \
  --output_dir "$V4_OMNI_RUN" --headless --device cuda:0 \
  > "${V4_OMNI_RUN}.console.log" 2>&1 < /dev/null &
```

先用omni模型评估 `walk forward left right yaw_left yaw_right`，通过后相同方式晋级transitions；其最终评估使用默认完整10个案例。不要把bootstrap的侧移/起停失败直接当成其连续迈步课程失败。

## 训练时判断什么

- `Motion/linear_progress_ratio` 必须与真实播放进度一起看；不能仅依据总体reward。
- `Gait/moving_single_support_fraction`、`Gait/contact_releases_per_env_second` 用于尽早发现一直双支撑；释放次数本身不等于有效迈步，正式评估还检查净空和落脚。
- `Actuator/leg_saturation_fraction`、`Actuator/arm_tracking_rmse` 与终止原因共同判断是否动力学或执行器受限。
- `PPO/optimizer_steps`、`PPO/actor_rejected_updates`、`PPO/encoder_rejected_updates`、`PPO/encoder_policy_probe_kl`、`PPO/post_update_probe_kl` 同时观察。持续几乎没有有效更新需要停下来诊断，不能把保护器频繁触发理解为训练健康。
- 若1000轮仍无卸载改善、无足底抬起、进度约零，优先核对各项回报、初态、力矩/关节时序和探索，不直接加到20000轮。若已出现可重复的有效交替，再继续当前课程并比较候选点。

每个run保存contract、源码快照及哈希、负载USD、参数、checkpoint与TorchScript。新schema为`paint_locomotion_v4_biped_v1`，禁止旧v2/v3/无步态v4/旧恢复步态v4权重混入。部署和上层teacher仍需新的135/360/3接口适配；旧78/420接口不能直接加载。

本地实现验证包括CPU逻辑/回滚测试和短时真实Isaac训练、保存、导出、失败评估链路。它们不构成训练收敛或学会走路的证据；正式4096环境训练结果仍待服务器实验。

2026-10-25 本地验证记录（运行产物在 `logs/rsl_rl`，不提交模型和仿真缓存）：

| 检查 | 结果 |
|---|---|
| 新双足/原v4/PPO回滚逻辑 | 30项通过，包含Adam状态精确回滚、真实净空与伪接触区分、晋级报告校验、TorchScript等价性 |
| 原步态/播放/StandWalk/Transport/teacher动作分布/v2回归 | 48项通过 |
| `v4_biped_final_smoke_20261025` | 原生Isaac、64环境×8轮；policy135/critic173、model_4/model_8及组合TorchScript保存成功 |
| 最终配置PPO探针 | 7轮各接受20次更新、1轮接受10次后拒绝下一步；最终KL范围约0.00613–0.01421；8轮均未拒绝encoder步骤 |
| `v4_biped_final_eval_20261025` | 重新加载model_8，站立与前进均在约1.22秒跌倒，`all_passed=false`；前进有效落脚0/0、`stepping_passed=false` |

短测时的单支撑和进度均夹杂摔倒过程，不能据此宣称步态恢复。可确认的是新任务已能运行，保护器能保留有效更新，评估链路会拒绝未学会走的模型。
