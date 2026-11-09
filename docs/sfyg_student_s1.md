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

## PIPER X 真机只读推理

`scripts/hardware/shadow_sfyg_student.py` 在连接松灵 PIPER X 的 Linux 主机上监听已有 SocketCAN 接口。
外部控制器保持与 checkpoint 一致的搬运目标。脚本将 J1～J6 依次对应 arm1～arm6，按训练时的关节零位和正方向解释反馈；
现场需核对安装和关节定义。启动时打印 checkpoint 的目标角。默认单轴偏离目标超过 0.10 rad 即结束记录。
这是采集判据，脚本没有设置姿态、使能或下发控制指令的入口。

依赖 NumPy、CPU 可用的 PyTorch（支持 `torch.load(weights_only=True)`）及 `python-can>=4.3,<5`。
默认 CAN ID 为 0x2A5～0x2A7（角度，0.001 度）和 0x251～0x256（高速反馈速度，0.001 rad/s），
分别转为 rad 和 rad/s。采用同一批次齐全的九个反馈报文，记录报文时间跨度，并在源时间戳上以 50 Hz 取样。
若接口没有完整反馈或丢失一个控制周期，保留已记录的行并在报告中写明原因。
字段定义见[松灵协议解码实现](https://github.com/agilexrobotics/piper_sdk/blob/master/piper_sdk/protocol/protocol_v2/piper_protocol_v2.py)和[松灵关节反馈节点](https://github.com/agilexrobotics/piper_ros/blob/humble/src/piper/piper/piper_ctrl_single_node.py)。

```bash
python scripts/hardware/shadow_sfyg_student.py --can can0 \
  --checkpoint ~/sfyg-shadow/model_500.pt --calibration ~/sfyg-shadow/handoff6.json \
  --seconds 30 --output-dir "$RUN"
```

校正文件使用 `student500_6N_handoff_20261109_143239_149054/summary.json`，脚本核对原 student 及绑定的底层 SHA。
`steps.jsonl` 保存同帧的关节角、关节速度、CAN 时间戳、两个版本各自的 history144、Fx、raw_vx 和 cmd_vx。
`native` 使用原始输入，`mujoco_cal` 按报告加入六轴 q 偏置，同时以 `min(8, (Fx-zero_n)*1.5444)` 替换 actor 的 Fx 输入。
两组都执行 checkpoint 的启动屏蔽、裁剪和死区。回放时用 `--replay "$RUN/steps.jsonl"` 代替 `--can can0`，
并指定新输出目录。两组估计都没有使用真机力传感器数据，测力计读数需按时间另行记录。
- `steps.jsonl`：当前动作前估计/标签，以及同一状态下 teacher/student 指令，
  与执行后的速度、复位前终止状态分别记录。

单机 MuJoCo 使用下方的 S1 入口。多机 student 的接入与验证仍待完成。
固定臂 PD 为800/40，弱外力可观测性、步态引起的偏置和仿真差异需要实际测试。

## 单机 MuJoCo 回放原版 500 轮 student

在仓库根目录运行。脚本固定读取
`logs/rsl_rl/sfyg_student_s1_gpu0_500_20261107_141132/train/model_500.pt`。
底层使用同一 robust checkpoint 的 MuJoCo 导出，并在加载时核对绑定的 SHA256。

```bash
LOW=logs/rsl_rl/sfyg_robust_5000_gpu1_20261027_131013/model_5000.pt
BUNDLE="${LOW%/*}/exported/model_5000"
if [ ! -f "$BUNDLE/deployment.json" ]; then
  python scripts/mujoco/export_sfyg.py --checkpoint "$LOW" --output_dir "$BUNDLE"
fi
python scripts/mujoco/run_sfyg_student.py --force_n 8
```

切换 `--force_n 4` 可观察临界起步和停步。无桌面时加 `--headless --video`，
在 Linux 可用 `MUJOCO_GL=egl` 运行。每次回放在 `logs/mujoco/sfyg_student_*`
下写入独立的 `interface.json`、`steps.jsonl` 和 `evaluation.json`，开启录像时写入 `preview.mp4`。
外力施加在实体负载质心。每个控制周期只向 student 输入 144 维机械臂历史。
`steps.jsonl` 记录动作前时刻、真实施力、估计的底座系 Fx/Fy (N) 与 Mz (N·m)、raw 动作、执行指令和步进后的实测速度。

### 固定 S1 权重的因果对照

在同一环境运行 `python -u scripts/mujoco/diagnose_sfyg_student.py`。默认使用上述 student 和底层权重，CPU 无窗口运行。每个工况从同一初始条件重置物理状态、关节目标、PD 队列与策略历史。

- `student_8N` 执行原生 student。每一帧使用同一份动作前历史，额外计算仅替换 Fx 以及替换全部三个分量（Fx、Fy、Mz）后的 raw/执行速度。替换值取该帧实际设定的外载，并按训练单位归一化。逐帧结果在 `shadow_raw_cmd_vx`，每对数字依次为前向 raw 和映射后的指令，单位 m/s。两种替换都只用于计算指令。
- `low_*` 绕过 student，给底层直接发送前向速度。0、4、8 N 各测 0、0.10、0.15、0.20 m/s，另有 16 N 零指令与 `student_16N` 原生轨迹。直接速度与拉力使用相同时间包络；零拉力下速度也按独立的时间包络给定。
- 每个工况写一个逐帧 JSONL 文件。`summary.json` 记录权重 SHA、终止原因、恒力段帧数、平均实际基座系 vx、世界系前向位移及左右脚接触地面后释放的次数。名义恒力段取动作前时间 `[5,13)` 秒、400 帧，实测速度是该动作执行后的值。若提前终止，汇总显示实际帧数。

### 8 N 真值 Fx 闭环与跨仿真器锁臂对照

在仓库根目录执行 `DEVICE=cuda:0 bash scripts/rsl_rl/run_sfyg_arm_transfer.sh`。脚本沿用上面的原始 500 轮 student 与 robust 底层。MuJoCo 从相同初态分别执行原生 student、只替换 Fx 的闭环。后者在每帧动作前用实际外力 Fx/40 替换估计的第一维；Fy、Mz 和历史均来自这条轨迹自身。撤力段也运行 actor。

`[MOTION]` 中时刻从回合开始计。统计从施力开始到恒力段末尾的持续运动/停步事件：0.5 秒窗口前向均速至少 0.05 m/s 判为起步，已起步后窗口速度 RMS 至多 0.03 m/s 判为停步。完整回合末 1.5 秒的 vx、vy、wz RMS 分别低于 0.03、0.05、0.10 才记为停稳。

两个仿真器各自执行 0、4、8 N 零速度指令，记录同名六关节在动作前的 `q-q_nominal`、`dq` 和 Fx 估计。汇总取同一恒力窗口；`arm_comparison.csv` 给出每个仿真器的均值、相对本仿真器 0 N 的增量、角度标准差、角速度均值及 RMS。CSV 中关节量单位为 rad、rad/s，终端表格角度为 mrad。原始数据在 `isaac/steps.jsonl` 与 `mujoco/low_*_vx0.00.jsonl`；单次配对诊断以同样的编号、seed 和外力时序运行。

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
