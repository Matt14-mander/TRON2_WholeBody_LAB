# v4 恢复步态后的训练安排（2026-10-25）

目标：判断在修复后的 PPO 上恢复 TRON2 步态，能否恢复按速度命令交替迈步。参考失败运行为用户报告的 `v4_nominal_klfix_4096_to20000_20261024_130411`；本地没有该次正式训练权重和日志，不能宣称已经定位唯一根因。

## 本轮配置

默认恢复相位引导：0.8 Hz，左右相位差0.5，支撑占比0.5，摆脚参考最高0.20 m，奖励权重0.75。恢复接触力、支撑脚速度、摆脚竖直速度与高度参考，移除原单脚支撑奖励，避免重复计算两套支撑目标。腿关节顺序和两只脚的顺序都显式固定为左→右。

观测和奖励直接读取同一 episode clock，站立时时钟继续推进、步态奖励为零；复位立即得到相位0，不依赖 command manager 本步是否已经更新。当前保留原1秒速度指令平滑过渡，没有新增“落脚完成后再停止”的状态机。这一轮检验步态收益，后续若起停仍失败，再独立设计落脚过渡。

摆脚参考采用每只脚自己的 swing phase，三次曲线上升/下降；当前0.5支撑占比对应原周期结构，未来改变支撑占比时不会把接触与抬脚参考错开。高度是 foot link 相对最近扫描点的地形高度，不是已标定的脚底净空。平地沿用旧link高度目标；本轮不进入粗糙地形课程，台阶能力仍需足底几何与局部地形检查。

| 项目 | 本轮主实验 |
|---|---|
| 任务/入口 | `Isaac-Limx-SFYG-TRON2A-Paint-Locomotion-v4` / `train_locomotion_v4.py` |
| gait | `--gait restored --gait_weight 0.75`（默认） |
| schema | `paint_locomotion_v4_gait_v1` |
| policy | 36 proprio + 81 height + 6 gait = 123维 |
| gait顺序 | sin、cos、frequency、offset、duration、swing_height |
| actor | 3 estimated velocity + 123 policy + 3 command = 129维 → 10 action |
| estimator | history仍为10×36=360，监督目标仍是critic前三维真实机体系速度 |
| 命令课程 | `--command_profile mixed`，保持20/40/20/20环境配额 |
| 线速度核 | `--tracking_std 0.5`，保持分母0.25；yaw核保持原配置 |
| 物理 | 完整SFYG、名义0.5kg实体负载、臂PD60/6、200/50Hz、平地 |
| PPO | LR3e-4、自适应范围1e-5～3e-4、desired KL0.01、保留KL修复与early stop |
| 探索/批次 | std0.6、entropy0.005、rollout32、5 epochs/4 minibatches |
| 其他 | 保持腿PD、动作缩放0.25、裁剪±5、姿态/动作惩罚及跌倒终止 |

当前新actor不能加载旧123维actor checkpoint，必须从零训练。旧v4播放/评估用 `--gait legacy`，该模式恢复117/360/3接口与旧single-support奖励。`--gait_weight 0`仍保留新相位观测，适合新接口下的奖励消融。旧transport FrozenBackbone只支持旧117维policy，会严格拒绝新schema；本轮先验收底层，再适配运输与MuJoCo，不能用补零方式绕过接口。

## Linux执行顺序

同步本次源码后，进入训练服务器工程根目录并激活原Isaac Lab环境。所有输出使用新目录。不要加载上述已站立不动的旧checkpoint。

```bash
conda activate limx
python -m pip install -e exts/bipedal_locomotion -e rsl_rl
python -m unittest discover -s tests -p 'test_locomotion_v4*.py' -v

export V4_TAG="$(date +%Y%m%d_%H%M%S)"
export V4_SMOKE="logs/rsl_rl/v4_gait_smoke_${V4_TAG}"
export V4_A="logs/rsl_rl/v4_gait_A_seed42_${V4_TAG}"

LIVESTREAM=0 python -u scripts/rsl_rl/train_locomotion_v4.py \
  --stage nominal --gait restored --gait_weight 0.75 \
  --command_profile mixed --tracking_std 0.5 \
  --num_envs 32 --max_iterations 2 --save_interval 1 --seed 42 \
  --output_dir "$V4_SMOKE" --headless --device cuda:0

# 首个诊断窗口；这是新随机初始化，不是500轮收敛承诺。
LIVESTREAM=0 python -u scripts/rsl_rl/train_locomotion_v4.py \
  --stage nominal --gait restored --gait_weight 0.75 \
  --command_profile mixed --tracking_std 0.5 \
  --num_envs 4096 --max_iterations 500 --save_interval 250 --seed 42 \
  --output_dir "$V4_A" --headless --device cuda:0

# 正式确定性评估，不用quick视频代替验收。起步每项3回合用于筛查。
for ITER in 250 500; do
  LIVESTREAM=0 python -u scripts/rsl_rl/train_locomotion_v4.py \
    --mode eval --stage nominal --gait restored \
    --checkpoint_path "$V4_A/model_${ITER}.pt" \
    --episodes 3 --seed 142 \
    --output_dir "${V4_A}/eval_${ITER}" --headless --device cuda:0
done
```

检查contract：policy=[123]、obsHistory=[10,36]、commands=[3]、action=10、height_width=81、schema含gait。评估JSON的`acceptance_version`应为`v4_motion_progress_v1`。

若更新与动作正常、存活及迈步趋势改善，可以继续到总计3000轮；若500轮尚未会走但指标正在改善，不据此直接判定失败。若仍持续不动且指标平坦，应运行下面的B对照，避免直接无诊断跑到20000轮。

```bash
export V4_A_CONT="logs/rsl_rl/v4_gait_A_to3000_seed42_${V4_TAG}"
LIVESTREAM=0 python -u scripts/rsl_rl/train_locomotion_v4.py \
  --stage nominal --gait restored --gait_weight 0.75 \
  --command_profile mixed --tracking_std 0.5 \
  --checkpoint_path "$V4_A/model_500.pt" --resume \
  --num_envs 4096 --max_iterations 2500 --save_interval 250 --seed 42 \
  --output_dir "$V4_A_CONT" --headless --device cuda:0
```

`--max_iterations`是额外轮数，因此500+2500=3000。续训要求相同物理设置、命令范围、gait、奖励权重、命令课程与tracking_std；修改目标函数应开始新的实验，不能伪装成完全相同的resume。仿真状态/RNG会重新初始化，续训并不逐位等价于连续3000轮。

在1000/2000/3000轮使用同样评估命令并选候选，候选至少20回合/案例；有希望后用训练seed43、44重复。早期只通过一两个方向，仍应分别报告其余方向失败，不能用总体平均掩盖。

## 除步态外，什么时候改其他配置

第一优先级是评估修复（已实现），不改变训练奖励：除原生存/倾角/RMSE条件外，新增每段移动忽略最初1s后的有向速度积分。实际线性进展/命令线性路程，以及实际转向/命令转角，都要求落在0.7～1.3；零速段不计算进展比。停止需要先观测到实际运动，且停下时间<2s。因此不动策略无法通过慢走或转向，也不能取得虚假的停止通过。该指标是机体系有向速度积分，不是完整的世界坐标轨迹误差；JSON同时保存逐步位置供检查绕圈、侧漂等问题。

评估逐步文件新增左/右足竖直接触力、foot link世界高度和gait输入，终止时保留reset前状态。观察确实左右交替并产生目标方向的运动，避免原地踏步、滑行或长期抬同一只脚。

若A恢复后仍不动，按单变量顺序做B/C；都保留同一新观测架构，从零开始、同一seed和训练预算比较，不同时改变PD/std/entropy：

| 实验 | gait | command_profile | tracking_std | 问题 |
|---|---|---|---|---|
| A（本轮） | restored/0.75 | mixed | 0.5 | 恢复步态后是否开始有效迈步 |
| A0（可选） | restored/0 | mixed | 0.5 | 相同新观测下，周期奖励本身的贡献 |
| B | restored/0.75 | walk_first | 0.5 | 站立/微小命令比例是否压制探索 |
| C | restored/0.75 | walk_first | 0.3 | 跟踪奖励过宽是否仍鼓励不动 |

`walk_first`配额10%恒站、80%恒走、5%两种起停各一；非零目标分量初始范围为|vx|∈[.15,.35]、|vy|∈[.10,.20]、|wz|∈[.15,.40]，一半轴向、一半组合，仍覆盖正负方向和纯yaw。命令上限若更小会同步缩小下限。评估固定命令不会被训练最小值重写，慢走0.05仍保留。

例如B从零训练：

```bash
LIVESTREAM=0 python -u scripts/rsl_rl/train_locomotion_v4.py \
  --stage nominal --gait restored --gait_weight 0.75 \
  --command_profile walk_first --tracking_std 0.5 \
  --num_envs 4096 --max_iterations 500 --save_interval 250 --seed 42 \
  --output_dir "logs/rsl_rl/v4_gait_B_seed42_${V4_TAG}" --headless --device cuda:0
```

C仅将B的`--tracking_std`改成0.3，使用新目录。从A到B、B到C按等预算比较；不能将A的500轮与B的3000轮直接作为课程优劣结论。

必须同时看PPO实际optimizer_steps、early_stopped、LR，以及post_actor_probe_kl/encoder_policy_probe_kl/post_update_probe_kl。若更新几乎停止，先查优化；若正常更新但仍学不走，优先上述任务对照。估计器显著改变有效策略时，再做真实速度输入的诊断分支。本轮不改PPO数学实现、估计器学习率或噪声，以避免混杂。

## 验证记录

本地CPU回归覆盖周期交替、相位/高度/速度一致性、可变支撑占比时参考对齐、地形高度平移不变性、站立奖励屏蔽、旧接口拒绝、walk_first配额/最小命令、慢走/纯yaw不动拒绝、真实运动后停止、新129维actor的PPO更新及TorchScript保存加载。

本地两轮Isaac Sim smoke只验证环境、训练、checkpoint和导出接口；不构成已恢复行走或收敛证据。正式效果以训练服务器的上述评估为准。

2026-10-25 本地验证结果：

- v4/步态/preview共19项，PPO KL诊断4项，transport14项，StandWalk13项，共50项通过。
- Isaac Sim nominal、32环境、2轮PPO完成，保存iteration1/2；运行时policy123、critic159、history10×36、actor129→10、gait_reward0.75。产物：`logs/rsl_rl/v4_gait_smoke_native_20261025`。
- 上述实际model_2与导出的TorchScript，在相同随机输入上最大绝对差0。
- stand/walk/slow/yaw_left各1回合原生评估完成并写出逐步足端与相位记录，均因未训练模型跌倒而失败，`all_passed=false`。产物：`logs/rsl_rl/v4_gait_eval_native_20261025`。这不是正式训练效果比较。
