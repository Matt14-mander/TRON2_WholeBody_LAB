# SFYG 源码恢复与分阶段持载行走

当前分支：`codex/sfyg-locomotion-reset`，直接从本仓库初始提交 `f441d67` 建立。原 v2/v3/v4、teacher/transport 试验留在 `paint-teacher-student` 历史中，本分支不导入它们。这里的“源码”明确指本仓库初始快照，不声称它等于当前上游的最新版本。

## 恢复了什么

新入口 `scripts/rsl_rl/train_sfyg_reset.py` 每次启动会比对关键文件与 `f441d67` 的 Git blob；不一致就拒绝将它称为源码基线。PPO、runner、actor、估计器、storage、算法配置、SFYG 资产配置、原始奖励/观测/复位/步态实现均未改动。

| 项目 | 恢复后的数值/行为 |
|---|---|
| 机器人 | 完整 SFYG，包含真实机械臂和夹爪质量 |
| 机械臂锁位 | 原版 `arm_lock`，隐式 PD stiffness800、damping40、effort_limit200 |
| 腿 PD、限幅、延迟 | 全部原值；腿0–4个物理步延迟恢复 |
| 腿动作 | 原始10维位置增量，比例0.25；没有新增tanh或±5裁剪 |
| actor/估计器 | policy42、history10×42=420、command3，actor48→10；原始速度估计器保留 |
| PPO初始LR / 估计器LR | 1e-3 / 1e-3 |
| 自适应LR | 原始KL逻辑，原始范围1e-5–1e-2；不封顶到1e-4或3e-4 |
| 额外更新保护 | 无v4 KL探针、参数/Adam回滚、估计器KL预算或新增梯度限制；原版early_stop默认False |
| rollout / epochs / minibatches | 24 / 5 / 4 |
| 探索 / entropy | std1.0 / 0.01 |
| 步态 | 原版0.8 Hz、相差0.5、支撑占比0.5、foot-link高度参考0.20 m |
| 奖励与终止 | 原版完整奖励、原版机身接触/动作阈值终止；无新增停滞、净空、落脚、失败-250等奖励 |
| 命令与复位 | 原版heading命令、vx±1/vy±0.5/yaw±1.5、10%站立，原版随机复位 |
| 物理 | 原版root z0.85、求解4/0、材质等；不混入v4出生高度与物理参数调整 |

这里的“锁住”沿用源码的高刚度PD锁位，不是把上肢质量删掉或把关节改成无质量固定件。其位置仍可能有误差，应实测。原版自身包含速度估计器与KL自适应LR；恢复源码不等于删除它们。本轮不同时移植Go2真值速度接口，以保持第一阶段可归因。

新入口对source训练只有三个显示/网络适配：关闭速度可视化标记、远程地面视觉材质和天空贴图。它们不改变碰撞材质、摩擦、回弹、奖励、指令和优化器。原始 `train.py` 也仍保留。

## 阶段顺序

1. **source**：原始SFYG臂姿态、无新增负载、无外力，先恢复基础行走。质量约39.523 kg。首次从零训练，不能加载旧v4权重。
2. **carry**：完整SFYG，臂保持 `[0, π/2, -1.4835299, 0, 0, 0]`，夹爪保持默认位置，新增0.5 kg实体负载，无外力。其余训练设置沿用source。默认质量约40.023 kg。
3. **robust阶段**：source和carry的服务器实测已出现前进行为；现新增独立的负载外力、质量与摩擦随机化配置，见[鲁棒性训练说明](sfyg_payload_robust.md)。source/carry保持无外力，不自动晋级，不直接复制四足40 N/10 Nm。

carry只新增物理适配：每个物理子步重发固定臂/夹爪目标，每次reset恢复该姿态；负载为夹爪处固定连接的实体box，尺寸0.20×0.12×0.08 m，偏置(0.15,0,0) m，含质量、惯量、碰撞体。资产输出到run目录，不改源USD。没有基座合成wrench、速度推扰、随机外力或机械臂运动轨迹。

实体负载会让原版critic的逐刚体质量观测多一维：source critic122，carry critic123（另各拼命令3）。source→carry明确只迁移actor、动作std和速度估计器，整个critic重新初始化。加载是严格检查的，不使用旧runner的“尺寸不匹配则跳过”功能。两阶段均不恢复旧优化器；新阶段学习率按源码1e-3重新开始。后续如需同阶段完整续训，另行实现并验证，不能把权重加载称为optimizer resume。

## 参考方法如何使用

[PAINT guide](https://paint-bot.github.io/docs/locomotion.html)允许自训速度跟踪策略，建议完整机器人、搬运臂PD、真实负载，再加入扰动。2026-10-26重新读取了其公开HTML源码。这里采纳完整上肢物理与分阶段负载训练，保留TRON2双足原有步态；不声称复现未公开ANYmal内部backbone。

本地 `Go2_PAINT_Reproduction` 的 `GO2-PIPER-PAINT-Loco-Flat` 成功案例同样采用真实上肢、固定搬运姿态和腿策略。借鉴其“小物理适配叠加成熟locomotion”的结构。本轮不一并复制四足对角镜像、PD数值、质量/外力范围、真值速度输入或奖励权重；先验证源码双足是否能恢复。

## 服务器同步

在旧训练结束后，于现有仓库切换新分支；训练日志不会被删除。若还需运行旧分支的任务，请在独立clone目录操作。

```bash
cd ~/workspace/isaaclab_tron2/TRON2_WholeBody_LAB
conda activate limx
git fetch origin
git switch codex/sfyg-locomotion-reset
git pull --ff-only origin codex/sfyg-locomotion-reset
```

继续使用现有Isaac环境，不升级Isaac/RSL-RL。新入口显式使用本仓库的RSL-RL和扩展目录，不需要为它改Go2可用环境的原生RSL-RL安装。

## 先运行source

```bash
mkdir -p logs/rsl_rl
LIVESTREAM=0 python -u scripts/rsl_rl/train_sfyg_reset.py \
  --phase source --num_envs 64 --max_iterations 8 --save_interval 4 \
  --output_dir "logs/rsl_rl/sfyg_source_smoke_$(date +%Y%m%d_%H%M%S)" \
  --headless --device cuda:0

export SFYG_SOURCE_RUN="logs/rsl_rl/sfyg_source_10000_$(date +%Y%m%d_%H%M%S)"
printf '%s\n' "$SFYG_SOURCE_RUN" > logs/rsl_rl/latest_sfyg_source_run.txt
nohup env LIVESTREAM=0 python -u scripts/rsl_rl/train_sfyg_reset.py \
  --phase source --num_envs 4096 --max_iterations 10000 --save_interval 250 \
  --seed 42 --output_dir "$SFYG_SOURCE_RUN" --headless --device cuda:0 \
  > "${SFYG_SOURCE_RUN}.console.log" 2>&1 < /dev/null &
echo "PID=$! RUN=$SFYG_SOURCE_RUN"
```

原生runner的周期保存编号从0开始，`model_0.pt`实际已经完成首次更新；最终10000轮保存`model_10000.pt`。本轮不修改这个编号行为。先看500/1000/3000等候选的动作与真实速度，再决定后续训练预算；长训轮数不代表通过验收。

固定前进0.25 m/s播放/录像，注意新旧脚本和输入不同：

```bash
export SFYG_SOURCE_CKPT="$SFYG_SOURCE_RUN/model_1000.pt"
LIVESTREAM=0 python -u scripts/rsl_rl/train_sfyg_reset.py \
  --phase source --mode eval --checkpoint_path "$SFYG_SOURCE_CKPT" \
  --command 0.25 0 0 --episodes 1 --video \
  --output_dir "logs/rsl_rl/sfyg_source_play_$(date +%Y%m%d_%H%M%S)" \
  --headless --device cuda:0
```

后退用`--command -0.25 0 0`，左右侧移用`0 ±0.1 0`，转向用`0 0 ±0.2`，站立用`0 0 0`。为减少变量，评估关闭观测噪声并使用固定默认复位，禁用heading转向而改为给定yaw rate；训练依旧保留原版heading与随机复位。输出`preview.mp4`、`evaluation_steps.jsonl`、`evaluation.json`，包含真实速度、倾角、足接触释放和臂误差。抓取自动reset前终止状态，不把复位后的正常状态当成未摔倒。

这些是诊断报告，不会仅凭20秒存活标记“学会走”。source实际能走后，应至少对前后、侧移、转向各做10回合，核对速度、实际位移、真实交替迈步和摔倒率。接触释放计数本身不排除抖动/滑动，需结合视频与逐步数据。

## carry独立从零训练

source和carry作为独立对照。按当前实验约定，carry从零初始化，不加载source权重：

```bash
export SFYG_CARRY_RUN="logs/rsl_rl/sfyg_carry_10000_$(date +%Y%m%d_%H%M%S)"
nohup env LIVESTREAM=0 python -u scripts/rsl_rl/train_sfyg_reset.py \
  --phase carry --payload_mass 0.5 \
  --num_envs 4096 --max_iterations 10000 --save_interval 250 --seed 42 \
  --output_dir "$SFYG_CARRY_RUN" --headless --device cuda:0 \
  > "${SFYG_CARRY_RUN}.console.log" 2>&1 < /dev/null &
echo "PID=$! RUN=$SFYG_CARRY_RUN"
```

carry播放沿用上述评估命令，改为`--phase carry --payload_mass 0.5`并选择carry checkpoint。代码仍支持显式的source→carry迁移作为另一组对照；本次服务器carry实验实际从零训练。下一轮robust则从已经能走的carry权重微调。

## 验证边界

每次run保存核心源码blob/SHA256、启动/物理适配源码快照、完整agent参数、观测形状、实际质量及上肢增益。checkpoint携带独立`sfyg_source_reset_v1`标识，旧v4/v3不接受。

本地只做短时原生Isaac链路验证与CPU检查；source与carry的长期locomotion恢复需要服务器训练结果。不能把8轮短测、原版文件一致或20秒不摔倒当成已恢复行走的证据。

2026-10-26本地验证记录：6项CPU测试通过，覆盖源码文件身份、原始PPO实际更新、跨阶段actor/估计器迁移与critic重建、旧模型拒绝，以及每个子步的臂锁位/复位。原生Isaac source与carry各64环境×8轮均完成训练和保存，实际质量/观测维数与上述记录一致。carry的model_8固定前进诊断在1.62秒机身触地，未完成20秒，报告`completed=false`，不属于有效步态。

原版自适应LR仍会降学习率：这次64环境短测source结束时为1.5e-5，carry为1e-5；初始值均确认是1e-3。恢复的是完整原始规则，不承诺LR保持1e-3。这个现象已记录，未据此重新增加限幅、回滚或修改源码；4096环境正式运行需结合其实际日志判断。

source的model_8录像链路也已验证，生成2.96秒、1280×720、50 fps有效画面，在机身触地时停止并记录`completed=false`。已检查画面中完整SFYG可见；该录像仅验证播放功能。
