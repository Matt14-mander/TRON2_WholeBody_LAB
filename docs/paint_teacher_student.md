# PAINT Teacher–Student：transport v4 的后续开发

分支：`paint-teacher-student`，基于 `paint-transport-v4`。

依据 [Teacher–Student Training](https://paint-bot.github.io/docs/training.html) 与
[Transport Environment](https://paint-bot.github.io/docs/environment.html)。指南是公开复现说明与骨架，
不是作者原始私有训练代码。本实现延续 Isaac Lab / TRON2 的物理与动作适配。

## 训练顺序与当前边界

1. 训练/验收 locomotion v4，冻结一个明确 checkpoint。
2. 同一个 transport 环境中训练 privileged Teacher PPO，验收零力/固定方向/带载表现。
3. 冻结 Teacher 和低层；Student 在自己的 rollout 上用 PPO + KL(Student || Teacher) 训练，
   estimator 在同一 mini-batch 上监督回归真实归一化交互参考。

现在可以完成代码和开发 smoke，不必等待低层收敛。正式 Teacher/Student 能力训练必须使用验收过的上游模型。
行为克隆、DAgger、离线数据集和预训练 estimator 不是这条主线的前置条件。

## 实现

- `rsl_rl/rsl_rl/modules/intent_estimator.py`：144→128→128→128→3，ELU；部署组合模块只接收 history144。
- `rsl_rl/rsl_rl/storage/paint_storage.py`：每个 transition 保存 pre-action Student144、Teacher147、critic177、
  动作、旧 log-prob 与 latent Normal 参数。打乱索引一致，不丢弃不能整除 mini-batch 的余数。
- `rsl_rl/rsl_rl/algorithm/paint_ppo.py`：独立 Student PPO，Teacher 冻结；estimator 用独立 Adam。
- `rsl_rl/rsl_rl/runner/paint_runner.py`：配对采样、训练日志、严格 checkpoint、优化器恢复和组合 TorchScript 导出。
- `scripts/rsl_rl/train_transport_v4.py`：复用环境创建、冻结低层、固定案例评估和可视化；新增 Student 训练/控制器。

没有修改通用 PPO/storage/runner 的实现。Teacher 与 Student 都沿用当前 v4 的 tanh-Normal 有界分布，
KL 使用变换前 Normal 的解析值，两者共享同一可逆 tanh，因此等价。Teacher KL 是 Student→Teacher，
与 PPO 的 old-policy→new-policy KL 分开记录。不能对变换后动作直接套 Normal KL。

网络输入为 `[student_history144, estimator(history).detach()3]`。Teacher 输入为
`[clean_history144, true_c3]`，Teacher 两阶段的历史/顺序保持一致；Student critic 保留仿真特权177维。
策略、critic、KL 的梯度不能进入 estimator；代码在更新时检查这一条件。

回归目标是底座系 `c=[Fx/40,Fy/40,Mz/10]`，不是物理单位的力，也不是额外的速度指令输入。
历史每帧为 `q-q_nominal(6), dq(6), previous_executed_target-q_nominal(6)`，8帧旧→新，reset零填充。
Student历史有原环境的均匀噪声，Teacher历史干净。

## 时序

每次动作前，从同一 `obs` 同时取 `student`、`teacher`、`critic`，并 clone 后记录。
`env.step()` 返回的新 `extras.teacher_obs` 对应下一个观测，不能用于监督上一个动作。
当前 Teacher 的最后3维直接提供当前 pre-action 的标签；不能误用 `transport_state.reference`，
后者是 preceding interval 的诊断值。自动reset后的下一次动作使用新episode的成对观测。

Student必须从已发生的臂响应推断当前力，无法预知突变；不通过前移标签、暴露真实力给actor来消除这个差异。
当前10/80/10施力曲线缓慢变化；仍需在正式评估中检查渐入、释放和噪声下的误差。

## 运行

以下在服务器原 `limx` 环境运行。Teacher/Student 使用相同低层文件、地形、随机化、受力幅度、负载质量、
臂动作尺度和速度限幅；这些设置不匹配会拒绝启动。Student默认继承Teacher的负载高度奖励区间，避免重新标定差异。

```bash
conda activate limx
cd ~/workspace/isaaclab_tron2/TRON2_WholeBody_LAB
python -m pip install -e exts/bipedal_locomotion -e rsl_rl
python -m unittest discover -s tests -p test_paint_student.py -v

export LOW_CKPT=/absolute/path/to/accepted_locomotion/model_N.pt
export TEACHER_CKPT=/absolute/path/to/accepted_transport_teacher/model_N.pt
export STUDENT_RUN=logs/rsl_rl/paint_student_smoke_01

# 两轮只验证执行链路；默认仍是当前v4的小受力/0.5kg负载配置。
LIVESTREAM=0 python -u scripts/rsl_rl/train_transport_v4.py \
  --mode train --actor_mode student \
  --locomotion_path "$LOW_CKPT" --teacher_checkpoint "$TEACHER_CKPT" \
  --num_envs 16 --max_iterations 2 --save_interval 1 \
  --headless --device cuda:0 --output_dir "$STUDENT_RUN"

# Student实际闭环；不使用真实力替代估计结果。
LIVESTREAM=0 python -u scripts/rsl_rl/train_transport_v4.py \
  --mode eval --actor_mode student --controller student \
  --locomotion_path "$LOW_CKPT" --teacher_checkpoint "$TEACHER_CKPT" \
  --student_checkpoint "$STUDENT_RUN/model_2.pt" \
  --cases zero forward backward left right yaw_left yaw_right --episodes 3 \
  --headless --device cuda:0 --output_dir "${STUDENT_RUN}_eval"

# 在新目录恢复两套优化器与迭代计数；max_iterations是新增轮数。
LIVESTREAM=0 python -u scripts/rsl_rl/train_transport_v4.py \
  --mode train --actor_mode student --resume \
  --locomotion_path "$LOW_CKPT" --teacher_checkpoint "$TEACHER_CKPT" \
  --student_checkpoint "$STUDENT_RUN/model_2.pt" \
  --num_envs 16 --max_iterations 2 --save_interval 1 \
  --headless --device cuda:0 --output_dir "${STUDENT_RUN}_resume"
```

如果 Teacher 训练采用 rough/randomized/disturbance_scale=1，Student启动也必须显式带相同参数。
以上命令默认 flat/nominal/disturbance_scale=0，仅适用于相同设置的Teacher。
`--distill_coef` 默认1e-6，`--est_learning_rate` 默认固定1e-3，与guide一致；没有恢复guide的较大PPO学习率或探索幅度。
PPO沿用TRON2 v4保守设置：32步rollout、初始std0.2、LR1e-4及现有奖励/动作范围；这属于明确的工程适配。

恢复保留actor/critic、estimator、两套优化器、当前LR和iteration；不恢复物理引擎、episode或随机数状态，
因此不是逐轨迹确定性重放。不带 `--resume` 的 `--student_checkpoint` 只做权重warm-start，从iteration0开始。
评估可以改变保存频率、seed、设备和环境数；恢复训练要求policy/algorithm/rollout长度一致。

## 产物与检查

- `contract.json` / `student_contract.json`：环境、源码SHA、低层SHA、Teacher文件SHA和Student I/O。
- `model_N.pt`：Student模型、estimator、两套优化器、配置、iteration；不重复存低层或Teacher权重。
- `student_actor.pt`：组合TorchScript，`history144 -> action9`，已含estimator与tanh；无critic/Teacher/真实力输入。
  低层仍是单独固定的部署产物。导出会重新加载文件并检查数值一致性。
- `student_training.jsonl` / TensorBoard `Student/*`：任务损失、Teacher KL、归一化估计MSE、实际优化步数、
  PPO rollout KL、estimator引起的策略漂移、全部更新后的probe KL、失败/timeout。
- `evaluation.json`：继承原transport跟踪/倾角/完整回合gate，新增估计RMSE（归一化及N/N/Nm）。
  `rollout.jsonl`记录动作前估计与真实c，和前一区间物理诊断分别命名。
- `completion.json`：完成流程、低层fingerprint一致与Teacher冻结核验。完成不等于任务能力通过。

Teacher需要原训练checkpoint的分布参数（包括std）；仅导出的确定性 `teacher_actor.pt` 无法提供KL所需分布。
旧payload-local四帧intent estimator的权重不直接迁入。

正式评估应使用相同seed/受力案例对比hold、reference、Teacher和Student，分别报告估计误差、跟踪、跌倒和完整回合。
当前工程gate没有单独的estimator精度合格线；不能把任意小MSE或短smoke解释为正式Student验收。

## 2026-10-24 开发验证

- 新增12项CPU测试通过：KL方向及tanh等价、真实PPO/回归更新、独立梯度、Teacher冻结、
  pre-action缓冲快照、mini-batch配对及余数、timeout/失败、KL提前停止、CLI、契约、恢复及TorchScript一致性。
- 原transport14项、locomotion9项、PPO KL诊断4项、有界动作分布3项均通过；合计42项。
- 本地Isaac Sim 4.5 / CUDA，4环境完成Teacher两轮smoke：`logs/rsl_rl/student_v4_teacher_smoke_02`。
- 4环境Student两轮训练完成，每轮PPO和estimator均20次更新，保存model_1/model_2和student_actor，
  低层及Teacher冻结核验通过：`logs/rsl_rl/student_v4_native_smoke_01`。
- 保存的Student严格重载后完成zero/forward/yaw_left三个固定case，估计与物理跟踪日志完整：
  `logs/rsl_rl/student_v4_native_eval_01`。三个case均约1.08–1.10秒提前跌倒，`all_passed=false`，
  没有把流程完成解释为搬运成功。
- 原生恢复model_2的模型、estimator及两套优化器，继续至iteration3并再导出：
  `logs/rsl_rl/student_v4_native_resume_01`。

以上均使用已有未收敛的低层开发权重和两轮Teacher，仅验证算法/环境集成。没有启动正式长训，
也没有完成真实机器人部署或有效Teacher/Student策略验收。
