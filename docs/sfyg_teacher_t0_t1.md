# SFYG T0/T1：冻结 locomotion，固定臂前向 teacher

本任务位于 `codex/sfyg-locomotion-reset`，采用当前已能行走的 SFYG carry/robust checkpoint。
独立入口是 `scripts/rsl_rl/train_sfyg_teacher.py`；不需要注册/覆盖旧 Gym 任务。
不整体迁入 v3/v4 teacher，不修改原 SFYG 的16个核心源码、PPO、runner、奖励、观测、机器人资产、carry/robust任务或 MuJoCo demo。
每次运行前后执行 `verify_source()`，每次高层保存和任务结束检查冻结底层参数指纹。

参考 [PAINT training](https://paint-bot.github.io/docs/training.html)、[Transport environment](https://paint-bot.github.io/docs/environment.html) 与本机 Go2 工程的固定臂 teacher 课程。
这是 TRON2 前向启动阶段，不宣称完整论文复现、横向搬运或 student 已实现。

## 训练契约

| 项目 | 当前设置 |
|---|---|
| 物理模型 | 完整 SFYG；当前搬运姿态；独立0.5 kg刚性负载；无世界支撑、无臂重力前馈 |
| 机械臂/夹爪 | 沿用800/40等原执行器设置，固定原PD目标；实际关节仍参与动力学 |
| 腿执行 | 原PD、0.25动作比例、无新增腿动作clip/tanh；两组腿执行器固定2×5ms延迟 |
| 冻结 LL | policy42、history10×42、commands3；actor与速度估计器一起冻结 |
| LL观测 | 使用已保存的clip/scale和关节顺序；相位持续推进，站立也不重置；历史首次重复填充 |
| Teacher actor | 147：8×18臂历史＋当前归一化真实外力3 |
| 臂历史 | q−q_nominal、原单位dq、上一执行目标−q_nominal，旧→新、reset清零，每控制步一次 |
| Critic | 177：actor147＋末端/负载各15维位置、线/角速度、线/角加速度；向量在base轴表达 |
| 高层动作 | 原始9维Gaussian；vx按物理m/s使用，裁剪到[0,0.3]，≤0.02归零；vy/wz和臂输出不执行 |
| 高层其余输出 | 仍参与原始动作成本，不会绕过固定臂设置；未训练student |
| 时间 | Isaac physics 0.005s、decimation4、高低层50Hz |
| 回合 | 18s：0–3s落稳，3–5s加力，5–13s保持，13–15s撤力，15–18s零力 |
| 落稳/撤力 | 落稳期间强制底盘零速；撤力后仍由teacher控制，不能强制归零伪造停稳 |
| 交互力 | 单机负载COM、base x方向；F/40是真实teacher输入与速度参考；每个控制区间内幅值保持、每物理步旋转并重施力 |
| 采样 | 20%零力，10%弱力(0,2N)，70%运动力[4,8N]；没有横向力或偏航力矩 |
| 随机化 | 暂停质量/摩擦/reset速度随机化；不叠加原robust随机施力器；保留自碰撞 |

双机物体总8N不等于单机器人8N。当前teacher的真实外加力不等于抓取反力，导出不能直接替换双机FT导纳。
未来teacher双机特权评估和student/抓取反力闭环必须分开定义；本次没有改动现有demo。
MuJoCo四个求解子步也没有被直接移植为Isaac物理频率修改。

## 高层奖励与普通PPO

每项连续成本只由 RewardManager 乘一次0.02s：

- 平动：`-2*(1-exp(-||v_xy-c_xy||²/0.25))`。
- 偏航：`-(1-exp(-(wz-cz)²/0.25))`，T1中cz=0。
- 辅助指令：`-0.5*||executed_command-c||²`，不以此代替实际速度评估。
- 倾斜：`-0.5*||projected_gravity_xy||²`。
- 原始动作：`-0.002*sum((raw-previous_raw)²+0.5*raw²)`，未执行的维度也在内。
- 非timeout失败：每次−50；包括base/大腿接触、倾角>35°、base高度<0.45m、臂/负载非法接触。
- 不复制旧teacher的强全腿运动惩罚，也不使用不适配TRON2的ANYmal高度区间。

沿用仓库原版PPO/runner，teacher配置独立：128×3 ELU、24steps、5epochs、4minibatches、LR初始1e-4、adaptive targetKL=.01、entropy=.001、gamma=.99、lambda=.95、clip=.2、gradnorm=1。
原始PPO已有adaptive学习率范围[1e-5,1e-2]，这里只改变新teacher的初始LR；没有加入新上限、early-stop、encoder KL或蒸馏KL。
原 `ActorCritic` 实际忽略 `init_noise_std` 参数，所以入口对新teacher实例的logstd显式设为log(.1)，末层均值初始化缩小；原类和低层均不改。
高层兼容encoder的输出宽度为0，无估计器优化器；`obsHistory147`只是原runner的占位，不与低层420维历史混用。

## T0通过条件

`smoke`先读回原生物理外力缓冲，检查8N仅作用于负载、无额外力矩、每控制步一次LL/四次写力、固定臂与限幅、部分reset隔离和残留清零。
接口审计为了让未收敛本地权重也能覆盖非零施力，显式将测试episode时钟设置到保持段；报告标为`synthetic_phase_for_force_readback=true, is_acceptance=false`，随后完整reset。
这不用于T0运动验收或训练。

`baseline`运行同seed的hold/reference × 0/2/4/6/8N，以及无外力的0.05/0.1/0.15/0.2m/s探针；每回合18s，自动reset前捕获终止状态，只统计首次终止以前的数据。
不会将跌倒后重置继续运行拼成成功回合，也不会将timeout与失败同帧算通过。

训练gate要求：

- hold零力、reference零力与4/6/8N均完整运行，无非timeout失败；最大倾角<15°，臂最大误差<0.1rad。
- 保持段vx RMSE≤max(0.025,参考速度×0.5)，平均vx在参考0.5–1.5倍；vy RMS<0.2m/s且均值绝对值<0.05，wz RMS<0.15rad/s且均值绝对值<0.05。
- 零参考保持段和末尾1.5s的vx/vy/wz RMS分别<0.03/0.05/0.1。
- 4/6/8N同工况hold/reference都完整运行，reference平均总回报高于hold，确认所有奖励相加后不会更偏好站立。
- 2N和无外力探针仅诊断，不能伪装弱力起步能力已通过。

这是首版工程阈值，不是论文标准。单机SFYG横向步态摆动用RMS和均值共同记录，不能只看正负抵消后的均值。
评估同时输出起步延迟、停稳误差、原始vx越界比例、奖励分项、失败和时间信息。

`baseline`失败会保存报告并非零退出，启动脚本不会继续长训。正式训练还会独立重算gate并核对low SHA和完整契约。
失败时查看哪一力档、哪一阶段失败；不要删除gate或以增加轮数替代物理检查。
唯一开发绕过`--development`仅允许≤16环境、≤2轮更新，保存的产物标为开发用途。

## 服务器同步与整轮启动

沿用已有limx，无需升级包或更换底层策略：

```bash
cd ~/workspace/isaaclab_tron2/TRON2_WholeBody_LAB
conda activate limx
git switch codex/sfyg-locomotion-reset
git pull --ff-only origin codex/sfyg-locomotion-reset
git submodule update --init --recursive
unset CUDA_VISIBLE_DEVICES

export RUN="logs/rsl_rl/sfyg_teacher_t1_$(date +%Y%m%d_%H%M%S)"
export LOW="logs/rsl_rl/sfyg_robust_5000_gpu1_20261027_131013/model_5000.pt"
mkdir -p logs/rsl_rl
printf '%s\n' "$RUN" > logs/rsl_rl/latest_sfyg_teacher_run.txt

nohup env LIVESTREAM=0 DEVICE=cuda:1 NUM_ENVS=512 ITERATIONS=5000 \
  RUN="$RUN" LOW="$LOW" bash scripts/rsl_rl/train_sfyg_teacher_t1.sh \
  > "${RUN}.console.log" 2>&1 < /dev/null &
TEACHER_PID=$!
printf '%s\n' "$TEACHER_PID" > "${RUN}.pid"
echo "PID=$TEACHER_PID"
echo "目录=$RUN"
tail -n 80 -f "${RUN}.console.log"
```

上述命令取消当前shell的CUDA_VISIBLE_DEVICES映射，`DEVICE=cuda:1`使用物理第二张CUDA卡。
Kit参数采用`--kit_args="--/renderer/multiGpu/enabled=false"`的等号形式，避免argparse把参数内容当成新选项。
脚本顺序是interface → baseline → fresh teacher。任一步失败停止，不会停止其他训练进程。
每100轮保存；100/500轮先看实际跟踪与停稳，5000是预算，不是收敛保证。

## 中期teacher评估

```bash
RUN="$(cat logs/rsl_rl/latest_sfyg_teacher_run.txt)"
LOW="logs/rsl_rl/sfyg_robust_5000_gpu1_20261027_131013/model_5000.pt"
for FORCE in 0 4 8; do
  OUT="${RUN}/eval500_${FORCE}N_$(date +%Y%m%d_%H%M%S)"
  LIVESTREAM=0 python -u scripts/rsl_rl/train_sfyg_teacher.py \
    --mode eval --controller teacher --low_checkpoint "$LOW" \
    --teacher_checkpoint "$RUN/train/model_500.pt" \
    --force_n "$FORCE" --num_envs 1 --episodes 3 --delay_steps 2 \
    --headless --device cuda:0 \
    --kit_args="--/renderer/multiGpu/enabled=false" --output_dir "$OUT"
done
```

需要视频时加`--video`；无headless则为窗口模式。输出steps.jsonl/evaluation.json；force_n是单机器人负载施力。
本机Isaac Sim 4.5的离屏视频仍返回黑帧，未通过视觉验证；入口会在预热后检测并报错，不会把黑视频报告为成功。服务器默认无视频的T0/T1流程不依赖该渲染路径。
T1训练总是从零初始化高层，`--teacher_checkpoint`仅用于评估，不会静默warm-start旧v3/v4权重。
当前未提供训练优化器resume；不可把eval checkpoint参数当作resume。

最终保存`train/model_5000.pt`与`train/exported/model_5000/{policy.pt,manifest.json}`。TorchScript只输出原始9维动作，部署必须执行manifest中的固定臂、前向限制、启动时序，不能直接套用Go2或旧teacher解释。

## 本地开发验证（2026-10-28）

- 12项teacher单元检查与7项SFYG回归检查通过；Bash启动脚本语法检查通过。
- 真实原版PPO更新、初始std、奖励方向与越界惩罚、低层42维观测与现有MuJoCo构造一致、history/reset隔离、契约和TorchScript重载检查。
- 原生Isaac四环境接口审计：8N负载外力，其他body外力/力矩/复位残留均为0，一次LL推理、四次重施力；部分reset隔离通过。
- 原生Isaac八环境两轮teacher PPO完成，保存与导出成功，底层参数指纹及16个核心Git blob不变。
- 保存模型严格重载、连续两回合评估完成；本地短训底层的失败均如实记录。完整14工况×2环境的负例T0被拒绝。
- 启动脚本另用不依赖Isaac的产物检查器验证每阶段结果，避免Kit快速退出掩盖异常；已验证失败报告退出码为1。原生入口也显式设置Kit退出码。
- 本地仅有`sfyg_robust_smoke2_20261027/model_8.pt`短训底层。上述结果验证工程链路，不能证明正式模型通过T0或teacher学会搬运；服务器脚本将使用正式model_5000执行完整T0。
