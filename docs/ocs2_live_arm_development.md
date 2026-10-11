# 在线机械臂开发与验收

当前项目适合按“Isaac Lab 在线机械臂 → 求解性能 → 小规模诊断 → 采集/训练”推进。
现有异步 TCP、独立机械臂执行器、WholeBody policy 和离线采集结构可以复用。
本次实现的是第一轮在线诊断基础；尚未在 Isaac Lab 主机完成无摔倒或实时性验收。
不需要重训 locomotion policy，也没有修改 intent estimator、部署端或 MuJoCo。

## 本次实现

- 实时机械臂目标由 `Ocs2PlayBridge.set_target()` 或热更新 JSON 文件输入，不依赖轨迹 CSV。
- 目标从当前末端附近开始：更新位置距测量末端不超过 0.15 m，姿态为 WXYZ，arrival 为 0–1 s。
  该距离检查只是初期幅度限制，不能证明可达或无碰撞；关节边界另由执行端检查。
- 底盘始终使用已验证指令 `-0.29 -0.08 0`。求解器通过线性等式约束固定这三个输入，
  保留 18-state/9-input 数据契约；执行端拒绝底盘输出不一致的解。
- 等解不暂停物理仿真。过期、错误、越界或旧目标结果不执行；保持上次有界机械臂位置目标，
  速度、feed-forward effort 和策略中的预测 wrench 清零，行走指令继续保持。
- 位置目标默认限速 0.4 rad/s、限加速度 1 rad/s²，使用 Isaac 资产的 soft joint position limits。
  第一轮关闭 effort feed-forward；预测 wrench 来自优化计划，经过目标平滑后并不代表实际作用力。
- 重置后改为新的测量末端保持目标，清除旧解；新的 JSON id 才会再次接收外部目标。
- JSONL 记录每次求解开始/完成、目标接受、控制前状态/动作/实际下发目标和下一帧响应；
  记录求解耗时、排队时间、仿真/墙钟解年龄、末端与关节误差、倾角、双脚接触和失败/重置原因。
  自动重置返回的 next state 标记 `response_observation_valid=false`。

## Linux 主机启动

在仓库目录，先按现有 OCS2 构建流程重新编译，再使用新的固定底盘配置启动服务：

```bash
source ocs2_ws/install/setup.bash
ros2 run tron2_ocs2 tron2_ocs2_bridge \
  "$PWD/ocs2_ws/src/tron2_ocs2/config/task_arm_live.info" \
  "$PWD/robot_description/tron2/SFYG_TRON2A/urdf/robot.urdf" \
  /tmp/tron2_ocs2_codegen 5555
```

在线配置保留数值稳定性、自碰撞、1 s horizon 和原有容差，只关闭每次启动强制重编译代码生成库。
首次缺少缓存仍会生成/编译库。服务打印初始化耗时及每次 `mpc_ms`、
`validate_sample_wrench_ms`、轨迹 knot 数和总 `elapsed_ms`，用来区分启动成本与连续求解瓶颈。
尚未证实原先 15–18 s 的求解耗时已下降；不能据此宣称实时控制。

在 Isaac Lab 环境启动（CHECKPOINT 替换为现有 WholeBody checkpoint）：

```bash
LIVESTREAM=2 python scripts/rsl_rl/play.py \
  --task Isaac-Limx-SFYG-TRON2A-WholeBody-Flat-Play-v0 \
  --checkpoint_path CHECKPOINT --headless --ocs2 --ocs2_deterministic_reset \
  --ocs2_target_file /tmp/tron2_arm_target.json \
  --ocs2_diagnostic_log logs/ocs2_live/diagnostic.jsonl \
  --ocs2_live_duration 20
```

文件可以在运行时创建；不存在时保持初始目标。根据日志的实际末端坐标填写一个新目标，
建议先改变 0.01–0.02 m，保留测量姿态。每次更新使用新的 `id`，以临时文件原子替换：

```json
{
  "id": "reach_001",
  "position_world": [0.35, 0.0, 0.85],
  "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
  "arrival_s": 1.0
}
```

这些坐标仅展示格式，必须替换为本次实际末端附近的坐标和姿态。
position 为世界系；arrival 是目标接受时的仿真时间 deadline，而非每次请求重新计时。
超过 deadline 后发送 remaining arrival=0。JSON 解析/范围检查失败保留上次目标，且不确认该 id。

相同诊断控制路径也可由采集器启动：

```bash
python scripts/rsl_rl/collect_ocs2_rollouts.py \
  --live --checkpoint_path CHECKPOINT --headless \
  --live_target_file /tmp/tron2_arm_target.json --live_duration 20 \
  --output_dir logs/ocs2_live
```

此模式只记录单次有界诊断到 `live_diagnostics.jsonl`，可包含多个重置 episode；
原有 `--trajectory_manifest` 离线 NPZ/配对外力采集继续保留。
在线模式暂不生成训练划分或接受/拒绝 NPZ 数据集，也不施加外力；外力字段明确为零命令基线，非实测。
预测未来 wrench 始终标记为 OCS2 plan，不能作为 intent 外力标签。

## 性能测量与进入下一阶段的条件

### 第一轮 10 Hz 优化与 Linux 对照测试

目标是连续重规划 10 Hz，热求解 P95 优先争取 50–80 ms，且持续匹配 100 ms 截止。
`mpcDesiredFrequency=10` 只是 OCS2 配置元数据；TCP 服务由请求驱动，真正提交周期仍由客户端控制，
它不会自动把 15 s 求解变成 100 ms。控制执行仍沿用 policy 的频率。

本轮增加两项可回退优化：`model.analyticDynamics` 使用同一模型的解析流映射/精确 Jacobian；
`model.consistentInitializer` 用固定底盘输入、实测臂速度和模型精确积分初始化冷启动及延展窗口，
避免原来的零底盘输入违反 arm-only 等式约束。在线 SLQ 配置开启这两项；原 AD/默认初始化仍可选择。
碰撞、关节/输入限制、数值检查、1 s horizon、2 次最大迭代及旧解执行截止均保留。

`[OCS2_PROFILE]` 新增 reference/optimization/extraction 耗时、冷/热标记、迭代次数、代价和等式 SSE。
`profiling.internal=true` 打印 Initialization/LQ Approximation/Backward Pass/Compute Controller/Search Strategy 等
OCS2 内部统计。内部计时是**自最近一次 solver reset 起的累计统计**；外层耗时是单次请求。
尤其检查原来约 15 s 是否主要落在 SLQ 的 Backward Pass。取得日志后，再决定是否采用 iLQR。

同步代码后，在 `tron2_ocs2` 环境重编译并测试解析模型：

```bash
cd ~/workspace/isaaclab_tron2/TRON2_WholeBody_LAB
conda activate tron2_ocs2
export OCS2_CODEGEN_COMPILER="$(command -v x86_64-conda-linux-gnu-gcc)"
bash ocs2_ws/scripts/build_tron2_ocs2.sh
source ocs2_ws/install/setup.bash
colcon test --build-base ocs2_ws/build --install-base ocs2_ws/install --packages-select tron2_ocs2 --event-handlers console_direct+
colcon test-result --test-result-base ocs2_ws/build/tron2_ocs2 --verbose

# 先少量检查编译、模型和两次热求解；3 次结果不能当作可靠 P95。
MPC_BENCH_SAMPLES=3 bash scripts/ocs2/run_mpc_benchmarks.sh logs/ocs2_live/hold_single_gpu.jsonl

# 再运行每个配置 20 次，获取初步统计；慢的 SLQ 基线可能耗时数分钟。
bash scripts/ocs2/run_mpc_benchmarks.sh logs/ocs2_live/hold_single_gpu.jsonl
```

基准不需要启动 Isaac Lab 或 TCP bridge；参数 JSONL 取既有日志中 step>=100 的第一条实际状态与目标。
可用 `MPC_BENCH_STEP=0` 选初始状态。不传日志则使用名义状态和当前末端保持目标。
每个配置复用同一个 solver，冻结同一测量状态，只让 MPC 时间按 0.1 s 递增。
这是计算/热启动对照，**不是闭环仿真或实时回放**，既不会驱动机器人，也不会证明跟踪成功。
`solve_duration_s` 不含模型构造及首次代码生成，第一条 MPC 冷启动独立报告。

依次比较六个配置：原 AD+SLQ+默认初始化、解析 SLQ+默认初始化、解析 SLQ+一致初始化、
解析 iLQR、iLQR+固定 RK4 rollout、再改为单线程。各步只改相应因素，保留其他安全条件。
iLQR 使用离散 Riccati 递推，是规避连续 Riccati 积分开销的候选方案；RK4 rollout 为 10 ms 固定网格。
这些生成的 `.info` 仅供对照，尚未选为在线默认配置。

输出在 `logs/ocs2_benchmark/`：`summary.txt`、各配置的 `.jsonl` 和 `.profile.log`。
汇总包含热 p50/p95/max、deadline miss、缺失/错误、冷启动数、代价、等式 SSE、全部 knot 的最小
URDF 位置余量、最大臂速度/加速度以及输出固定底盘命令误差。
负位置余量表示计划越界。速度/加速度应分别对照配置 5 rad/s、20 rad/s²，
等式误差/代价用于对照解质量；这些统计不能证明自碰撞安全或实际末端跟踪。
`latency_gate_passed` 只表示该次基准的完整热请求均在 100 ms 内，不能代替在线验收。

选定候选配置后还需在 `LIVESTREAM=0` 下重测 TCP 端到端延迟及连续行走，
核查多个目标的新解实际被执行、状态时间/解年龄、末端误差、接触、倾角及重置。
当前在线 TCP 仍返回单个提前 20 ms 的命令样本；完整轨迹按当前控制时间采样是后续执行接口工作，
不能将该单样本接口视作论文中的完整轨迹执行。

`scripts/ocs2/benchmark_live_bridge.py` 使用符合 `Ocs2MpcObservation` 字段名的测量状态 JSON，
连续请求同一状态并递增仿真时间；输出冷/热求解耗时及 deadline miss。
它仅测试接口延迟，不能证明动态跟踪或行走稳定。

```bash
python scripts/ocs2/benchmark_live_bridge.py \
  --observation observation.json --output benchmark.jsonl --samples 20 --period 0.1
```

先比较服务端 MPC 与 wrench 时间，确认 Release 构建和缓存有效，再根据实际瓶颈调整
积分/离散配置或实现更轻量的求解结构；任何调整都需重新检查约束和跟踪。
socket timeout 为墙钟通信截止，不能中断服务端正在执行的 MPC；max solution age 是独立的执行截止，
默认仿真年龄和含排队的墙钟年龄均不得超过 0.5 s。不能提高年龄阈值来掩盖慢求解。

只有满足以下条件，才进入在线批量采集：

1. 连续求解耗时可持续匹配 0.1 s 更新周期，日志中不出现周期性过期/错误回退。
2. 连续行走期间多次接受不同目标，多个目标均有新解被实际执行，不能只记录目标输入。
3. 实际关节/末端跟踪、倾角和双脚接触可测，诊断期间无非超时重置/摔倒。
4. 每个控制间隔和下一帧响应按时间戳可对齐；自动重置响应被剔除。
5. 数据契约明确区分优化计划、实际响应和外力命令/测量，随后再扩展外力及训练格式。

本机验证：42 项纯 Python OCS2/基准单元测试通过；新增解析 Jacobian 对 AD/有限差分、
一致初始化对独立 RK4 积分的 C++ 测试，待 Linux 编译执行。未具备 Isaac Lab/PyTorch 或可用 WSL，
因此 C++ 编译、SLQ 固定输入约束收敛、求解性能和模拟器闭环仍待 Linux 主机验证。

## 重置选项与 RTX 启动崩溃

`--ocs2_deterministic_reset` 是可选的名义初始状态重置。开启后显式重置所有 robot 关节
（含机械臂和夹爪），取消 root/joint 的随机偏移；不改变资产初始姿态、GPU、驱动、RTX、
观测噪声或执行器延迟，也不保证跨驱动的数值一致性。
在线采集器不再自动开启该选项；需要时显式传入同名参数。
离线采集仍固定名义重置，所有 episode 均重置机械臂与夹爪；配对外力模式另行关闭噪声/随机延迟。

若日志在约 3 s 出现 `librtx.scenedb.plugin.so`、`libcarb.scenerenderer-rtx.plugin.so`、
`libomni.hydra.rtx.plugin.so` 的 Fatal backtrace，属于原生渲染路径崩溃。
仅凭这些帧不能确定具体驱动版本或证明重置选项是原因。
启动日志按顺序输出 `app_launcher_start`、`app_launcher_ready`、`environment_create_start`、
`environment_create_ready`：若缺少 `app_launcher_ready`，名义重置配置尚未执行。
保持同一驱动/环境/启动命令仅切换重置选项进行对照，并保存首次 Fatal 前的完整日志。
