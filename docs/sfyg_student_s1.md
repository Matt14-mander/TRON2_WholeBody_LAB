# SFYG Student 独立训练管线

当前阶段为单机、正向外力、固定机械臂的 S1 工程版本。复用当前 T1 物理环境和奖励，
底层 SFYG、原始 PPO、teacher 训练源码不修改。OCS2 和共享负载抓取反力不进入本管线。
已有 teacher 在 2/4 N 不启动、6 N 跟随不足；用它训练的 student 是诊断实验，不能宣称上游缺陷已解决。

## 数据与优化约定

- 部署只接收 history144：8 帧 × (臂位置误差6、臂速度6、上一已执行目标偏移6)，旧帧在前，reset 清零。
- 当前固定臂目标偏移为零；历史使用 T1 的干净观测，尚未加入传感噪声、随机延迟或新随机化。
- estimator 为 144→128→128→128→3 ELU，估计底座系外加负载质心 `[Fx/40,Fy/40,Mz/10]`。
  目前只有正向 Fx 被激励，不代表已经训练横向/偏航估计能力。
- actor 接收 history144 与估计值3；真实力仅用于 estimator 标签和冻结 teacher。
  critic 仍使用仿真特权177维，部署导出不包含 critic。
- 从 teacher 初始化 actor/critic/std，estimator 新初始化；student 自己执行采样动作。
- 独立 PPO：24 步、5 epochs、4 minibatches、LR=1e-4 固定、entropy=0.001；
  任务奖励加 `0.001 * KL(Student || Teacher)`，系数可配置。这些是当前工程设置。
- KL 在 raw Normal 空间计算，**不是裁剪/死区后执行动作分布的精确 KL**。不使用旧版 tanh 映射，
  不加入旧低层的 KL 提前停止或修改原 PPO。
- 每轮先冻结 estimator 完成 PPO，再用同一轮成对数据监督更新 estimator（独立 Adam，LR=1e-3）。
  actor/critic/KL 梯度不传入 estimator；记录 estimator 更新造成的策略 KL 和完整更新的 rollout KL。
  这种分阶段更新仍可能引起策略漂移，需要监控，不能用小 MSE 代替闭环测试。
- 采样保存动作前的 history/teacher/critic 快照。真实力标签取同一动作前 teacher 观测末3维，
  不取上一区间诊断值或下一时刻标签。突然施力无法从尚未发生的臂响应预测，需要测量响应延迟。
- 超时使用复位前 terminal critic 值 bootstrap；跌倒不 bootstrap；任何 reset 都截断 GAE 递推。

## 服务器运行

当前 teacher 可用于下面的 500 轮诊断；后续改善 teacher 后，通过 TEACHER 指定新 checkpoint，
新建 RUN。每个 student 与其 teacher 文件 SHA、底层 SHA 和执行契约绑定。

```bash
cd ~/workspace/isaaclab_tron2/TRON2_WholeBody_LAB
conda activate limx
export LOW="logs/rsl_rl/sfyg_robust_5000_gpu1_20261027_131013/model_5000.pt"
export TEACHER="logs/rsl_rl/sfyg_teacher_t1_20261028_130353/train/model_5000.pt"
export RUN="logs/rsl_rl/sfyg_student_s1_$(date +%Y%m%d_%H%M%S)"
mkdir -p logs/rsl_rl
nohup env LIVESTREAM=0 DEVICE=cuda:1 NUM_ENVS=256 ITERATIONS=500 \
  LOW="$LOW" TEACHER="$TEACHER" RUN="$RUN" \
  bash scripts/rsl_rl/train_sfyg_student_s1.sh \
  > "${RUN}.console.log" 2>&1 < /dev/null &
echo "PID=$! RUN=$RUN"
```

脚本先跑2轮开发检查，再从 teacher 重新初始化 student 正式执行所指定的诊断轮数，
最后配对评估 student/teacher/reference/hold，0/2/4/6/8 N、3 seeds、每 seed 4环境。
检查器核对完成标记、迭代数、冻结核验和导出 hash；物理失败的完整诊断仍可能正常结束，
必须检查 evaluation.json 的各 case gate。

恢复示例（同一组 LOW/TEACHER，配置参数必须与原训练一致，另建输出目录）：

```bash
LIVESTREAM=0 python -u scripts/rsl_rl/train_sfyg_student.py \
  --mode train --resume --student_checkpoint "$RUN/train/model_500.pt" \
  --low_checkpoint "$LOW" --teacher_checkpoint "$TEACHER" \
  --num_envs 256 --max_iterations 500 --save_interval 100 \
  --headless --device cuda:1 --kit_args="--/renderer/multiGpu/enabled=false" \
  --output_dir "${RUN}_resume"
```

恢复包括两套优化器和迭代计数；这里继续至1000轮。不恢复仿真物理状态或随机数流，
不是逐轨迹确定性重放。省略 --resume 为权重 warm-start，迭代从0开始。
teacher 文件改变会拒绝直接恢复；新 teacher 需要新训练或后续显式迁移工具。

## 产物和边界

- `model_N.pt`：student actor/estimator、critic、std、两套优化器及绑定契约。
- `exported/model_N/policy.pt`：只接收 history144，输出原始动作9；调用方仍需历史维护及 T1 执行动作处理。
  它不等于旧 teacher bundle，不能直接传入 demo 的 --teacher_bundle。
- `training.jsonl` 与 TensorBoard：PPO/蒸馏损失、估计物理 RMSE/偏置、策略漂移和任务统计。
  训练数据上的 estimator RMSE 不算留出集泛化结果。
- `evaluation.json`：逐 case 物理/跟踪/停车门槛和估计误差。
  `required_task_cases_passed` 要求 student 的0/4/6/8N案例齐全且通过，2N仍是诊断。
  `is_acceptance=false` 保留，因为估计器泛化、多机和真机验收尚未建立。
- `steps.jsonl`：当前动作前估计/标签，以及同一状态下 teacher/student 指令，
  与执行后的速度、复位前终止状态分别记录。

本阶段完成后仍需独立的 MuJoCo student 加载器和多机验证；现有 demo 的 student 入口继续明确拒绝未实现的部署。
固定臂 PD 为800/40，弱外力可观测性、步态引起的偏置和仿真差异需要实际测试。

## 2026-10-29 开发验证

- 12项CPU测试通过：实际PPO/回归更新、Teacher冻结、两优化器恢复、契约拒绝、
  原始Normal KL、超时/跌倒GAE、观测快照、history144导出及无特权输入。
- 既有Teacher的12项回归测试通过，共24项自动化检查。
- 本地原生Isaac运行4环境、2轮训练：`logs/rsl_rl/sfyg_student_native_20261029`。
- 原生恢复至第3轮并重新导出：`logs/rsl_rl/sfyg_student_native_resume_20261029`。
- 原生2环境、0/8N、student/teacher配对评估：`logs/rsl_rl/sfyg_student_native_eval_20261029`。
  本地未收敛开发底层在施力前即跌倒，所有case失败；只证明执行链路和失败记录有效，
  不证明外力估计或搬运能力。服务器正式权重不在本地。
- 三次原生运行均完成冻结权重核验及16个SFYG源码文件一致性核验；Bash启动脚本语法检查通过。
