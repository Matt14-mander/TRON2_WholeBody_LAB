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

每轮输出在独立的 `logs/ocs2_benchmark/run_<UTC时间>_<PID>/`：`summary.txt`、各配置的 `.jsonl` 和 `.profile.log`。
启动及结束时打印实际路径。可用 `MPC_BENCH_OUTPUT` 指定目录；脚本拒绝覆盖所选配置已有的结果。
stdout JSON 与 stderr profile 同时显示到终端并分别记录；脚本直接运行 overlay 中解析出的二进制，
避免将 `[ros2run]: Interrupt` 写入 JSONL。中断立即停止后续配置，`run.status` 标明整轮状态。
只有 `run.status=complete` 才是完整运行；这不表示性能或安全验收通过。
需要补取内部统计时先只跑 iLQR 一组，不必重跑六组或重新编译：

```bash
MPC_BENCH_OUTPUT="logs/ocs2_benchmark/ilqr_$(date +%Y%m%d_%H%M%S)" \
MPC_BENCH_VARIANTS=analytic_ilqr MPC_BENCH_SAMPLES=3 \
bash scripts/ocs2/run_mpc_benchmarks.sh logs/ocs2_live/hold_single_gpu.jsonl
```

汇总包含热 p50/p95/max、deadline miss、缺失/错误、冷启动数、代价、等式 SSE、全部 knot 的最小
URDF 位置余量、最大臂速度/加速度以及输出固定底盘命令误差。
负位置余量表示计划越界。速度/加速度应分别对照配置 5 rad/s、20 rad/s²，
等式误差/代价用于对照解质量；这些统计不能证明自碰撞安全或实际末端跟踪。
`latency_gate_passed` 只表示该次基准的完整热请求均在 100 ms 内，不能代替在线验收。

### 网格碰撞距离的逐项计时与保守几何对照

初步 iLQR 日志显示 Backward Pass 仅约 1–2 ms，初始化/LQ/线搜索占主要耗时。
原 URDF 碰撞网格中 base_Link 有 142490 个三角面，相关臂部 link 约 2–4 万面；
它们是距离查询的候选瓶颈，尚需逐项数据确认。
`profiling.internal=true` 现在额外输出 `[OCS2_TERM_PROFILE]`，分别计时 self_collision 与
end_effector 的 value/quadratic 调用。统计对 solver 克隆及线程共享，**每次求解重新计数、累加各线程工作时间**，
不能当作不重叠的墙钟耗时相加，也不能与累计 DDP 平均计时直接相减。

重编译后，用同一观测分别测试原网格与包围盒（各 3 次足够先定位数量级）：

```bash
BASE_OUT="logs/ocs2_benchmark/mesh_$(date +%Y%m%d_%H%M%S)"
MPC_BENCH_OUTPUT="$BASE_OUT" MPC_BENCH_VARIANTS=analytic_ilqr MPC_BENCH_SAMPLES=3 \
  bash scripts/ocs2/run_mpc_benchmarks.sh logs/ocs2_live/hold_single_gpu.jsonl

BOX_OUT="logs/ocs2_benchmark/aabb_$(date +%Y%m%d_%H%M%S)"
MPC_BENCH_OUTPUT="$BOX_OUT" MPC_BENCH_VARIANTS=analytic_ilqr MPC_BENCH_SAMPLES=3 \
  MPC_BENCH_COLLISION_PROXY=aabb \
  bash scripts/ocs2/run_mpc_benchmarks.sh logs/ocs2_live/hold_single_gpu.jsonl
```

包围盒候选由 `make_collision_proxy_urdf.py` 生成在独立结果目录中。每个碰撞 mesh 的全部三角形
由局部 AABB 包含，每侧增加 5 mm；正确应用 mesh scale、旋转 origin 和中心偏移。
保留关节/惯量、现有原始 primitive，视觉 mesh 路径转为绝对 URI。
继续使用同一组自碰撞 link 对、0.04 m 安全距离及原惩罚参数；原 robot.urdf 和在线默认路径不改。
独立 codegen 目录避免混用原网格的几何绑定。
`collision_proxy.report.json` 记录各 link 的三角面数和盒尺寸。
包围盒几何保守，可能产生原 mesh 没有的碰撞，因此它是计算对照及优化候选，
不能仅凭速度就用于在线控制。仍需核对初始接触/间距、计划限位、可达性和实际跟踪。

检查两组 summary 与 term profile：如果 mesh 的 self_collision 调用占主导且盒模型明显下降，
下一步优化碰撞几何/距离查询；若不是，则继续按实际逐项计时定位其他热点。

2026-10-11 Linux 对照：原网格 ILQR 两次热求解约 7.4–7.8 s，AABB 候选约
26.7–26.9 ms。碰撞项线程累计计时也大幅下降，确认网格距离查询是主要热点。
但候选计划仍出现约 0.69 rad 越界，最大速度 5.44 rad/s、加速度 31.03 rad/s²，
超过当前 5/20 限制。短样本时延通过不能作为下发依据。

summary 独立输出 `joint_input_gate_passed`，检查含冷启动的每次计划的关节位置余量、
最大速度、最大加速度及固定底盘指令；缺失指标为 null，不表示通过。
该检查限值对应 `task_arm_live.info`，位置容差为 1e-6 rad；它不验证碰撞间距或实际跟踪。
为了区分旧世界坐标目标与模型/约束问题，可保持测量 q/dq 不变，改用该帧实测末端姿态
作为新目标并给出 1 s 到达时间：

```bash
CHECK_OUT="logs/ocs2_benchmark/current_ee_$(date +%Y%m%d_%H%M%S)"
MPC_BENCH_OUTPUT="$CHECK_OUT" MPC_BENCH_VARIANTS=analytic_ilqr MPC_BENCH_SAMPLES=20 \
  MPC_BENCH_COLLISION_PROXY=aabb MPC_BENCH_TARGET_MODE=current_ee \
  bash scripts/ocs2/run_mpc_benchmarks.sh logs/ocs2_live/hold_single_gpu.jsonl
cat "$CHECK_OUT/observation_metadata.json"
cat "$CHECK_OUT/summary.txt"
```

`observation_metadata.json` 记录原目标/新目标距实测末端的位置距离、选中帧与到达时间。
20 次 current_ee 对照中，热求解 P95 为 17.38 ms，最大 18.89 ms；但全计划
最低关节余量仍为 -0.917 rad、最大加速度 32.97 rad/s²。目标替换未消除约束问题。
关节/输入限位是 RelaxedBarrierPenalty 软盒约束，`equality_sse` 只反映等式约束，
不能据此认定关节限位满足。新增 `analytic_ilqr_4iter` / `analytic_ilqr_8iter`
仅提高最大迭代数，保持目标、碰撞、限位和惩罚不变，用来测量收敛与耗时的关系。
这两个候选不加入默认全套测试；显式设置
`MPC_BENCH_VARIANTS="analytic_ilqr analytic_ilqr_4iter analytic_ilqr_8iter"` 进行对照。
Linux 迭代对照中，2/4/8 轮的最大越界约 0.917/0.760/0.522 rad，最大加速度
32.97/28.45/26.30 rad/s²，均未通过关节/输入验收。接下来固定 8 轮，只比较
`limits.boxConstraintMu`：原值 1e-3、10 倍 1e-2、100 倍 1e-1。
`boxConstraintDelta` 保持 1e-3，所有物理限位、碰撞、目标与固定底盘指令不变。
新增候选 `analytic_ilqr_8iter_mu10` / `analytic_ilqr_8iter_mu100`，不加入默认全套测试。
SolverCore 从配置读取并检查这两个正有限惩罚参数，缺省保持旧行为，启动时打印
`[OCS2_LIMITS] box_mu=... box_delta=...`。必须重新编译 tron2_ocs2；旧二进制会忽略新配置。
提高软惩罚不能保证严格可行，若仍越界需进一步定位具体关节/时刻及几何/模型问题。
不同 mu 下总 cost 口径改变，不据此横向判断跟踪优劣，应比较实际约束指标和时延。
惩罚对照的 mu=0.01/0.1 候选最大位置越界分别约 0.138/0.277 rad，最大加速度
23.63/20.23 rad/s²。加速度改善但位置非单调，暂停继续增大参数，先定位越界。
native benchmark 新增 `initial_arm_position_margin`、`worst_arm_position_index`
（六维 arm state 的零基索引）、`worst_arm_position_time`（绝对仿真时刻）、
`worst_arm_position_value`、`arm_lower_position_limits` / `arm_upper_position_limits`。
summary 分别报告冷/热求解最低位置余量、越界记录数及对应原始指标；旧 JSONL 仍可汇总，
但没有具体关节信息。这些诊断检查轨迹 knot，不能证明 knot 之间无越界。
再次编译后仅重跑 mu10/mu100 两个候选即可，物理限位和验收容差保持不变。
具体诊断发现两组最大越界均为六维 arm state 索引 4，且均在预测终点 t+1 s；
mu10 热求解该关节约 -1.692 rad，mu100 约 -1.830 rad，下限为 -1.553343 rad。
代码原先只把 limits 放在运行阶段 `softConstraintPtr`，终点末端 cost 有权重但没有
对应关节限位项。补充 `finalSoftConstraintPtr/arm_limits`，使用与运行阶段相同的
URDF 位置边界、速度限值及 mu/delta；终点没有输入，不添加加速度项。
启动打印 `[OCS2_LIMITS] terminal_arm_position_velocity=enabled`。
新增 `test_arm_state_limits` 检查全部关节上下边界、速度边界、梯度方向/有限差分及 clone。
需要在 Linux 编译并执行原生测试，之后保持同目标/几何/8 轮迭代重新比较 mu10/mu100。
这项修复补齐了目标函数的终点约束，但软惩罚仍不能保证严格可行；不能提前判定通过。
捕获初始状态另有约 2.2e-5 rad 的位置越界，与 0.1–0.3 rad 的终点越界不同，
诊断保留原测量值，不静默裁剪，也不改变 1e-6 rad 验收容差。
终点限位修复后的 mu100 对照：最低余量为原输入状态的 -2.2e-5 rad，最差点在
观测起点，原先 arm state 索引 4 的大幅终点越界已不再是最差点；加速度最大
20.0825 rad/s²、热求解 P95 25.15 ms。这不是完整可行性或闭环验收通过。
新增 `MPC_BENCH_MIN_ARM_MARGIN=0.000001` 筛选至少有该位置余量的真实诊断帧，
保留该帧全部测量状态，不裁剪/重写 q 或 dq。默认不筛选。找不到时明确报错，
不能用修改 observation 冒充实测。metadata 记录实际余量、要求余量和跳过帧数。
新增 `analytic_ilqr_8iter_mu100_acc19` 仅把优化内部加速度上限从 20 收紧为 19 rad/s²，
留出软约束余量；summary 仍按原物理验收上限 20 rad/s² 判断。这两个限值含义不同。
用同一选定有效帧比较 mu100 与 acc19，其他目标/几何/迭代设置保持一致。
Linux acc19 在 step=116 的限位内测量帧（余量约 7.84e-6 rad）上首次同时通过
时延及关节/输入门槛：20 次成功，热求解 P95 25.95 ms、最大 25.97 ms，最大速度
3.65 rad/s、加速度 19.19 rad/s²。这仍是冻结状态计算对照，不是在线闭环或碰撞验收。
下一步在 `profiling.internal=true` 且自碰撞开启时，独立 clone 碰撞距离约束，
求解结束后检查原始初始状态和所有预测 knot（含终点）。不依赖碰撞 penalty 的总 cost。
记录 `minimum_collision_margin`（距离减去配置最小距离，本配置为 0.04 m）、
`initial_collision_margin`、最差时刻/对象对索引、`collision_validation_ms`；检查耗时
计入 native benchmark 的整体 solve_duration_s。不开启诊断/自碰撞时不声明通过。
summary 增加独立 `collision_gate_passed`，缺失完整有效距离检查为 null。
这个门槛只检查当前几何和配置碰撞对的离散 knot，不能证明 knot 之间或其他 link 对无碰撞，
也不能代替实际全身姿态/接触与模型一致性验证。原生改动需在 Linux 编译执行。
20 次碰撞诊断中，初始 AABB 距离约 1.61 cm、最差预测距离约 1.05 cm，均不足
配置的 4 cm；正距离不足安全间距不能据此宣称实体相撞。终点碰撞项原先也缺失，
现补上与运行阶段相同的自碰撞软约束。初始不满足条件无法靠优化终点修复。
新入口 `TRON2_OCS2_COLLISION_ONLY=1` 对输入状态只计算原始碰撞距离，不运行 MPC。
`probe_collision_geometry.sh BENCH_RESULT_DIR` 用相同 observation/config 比较原网格与
该结果目录已有的 AABB，独立 codegen 和日志，不覆盖原结果。
启动输出 `[OCS2_COLLISION_PAIR] index=... first=... second=...`，可将对象对索引映射
到几何名称。探测 JSON 提供每对 margin、最差对象对及最小距离；先查原网格与盒模型
是否都不满足间距，再判断需处理初始姿态还是改进保守碰撞几何。不能只降低安全距离让 gate 通过。
默认 `MPC_BENCH_TARGET_MODE=recorded` 保持原对照行为。当前末端目标仍是固定世界坐标目标，
预测期间需要补偿底盘运动；该冻结观测测试不能证明走动时的实时跟踪。

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
