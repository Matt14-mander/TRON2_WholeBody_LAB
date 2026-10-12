# PAINT 意图估计：PD＋负载交互采集与 MuJoCo 部署

本分支 `codex/paint-intent-pd` 从 `ocs2-core-stage1` 的 `82ef92c` 创建。
第一目标是让意图估计器从机械臂观测历史恢复交互力，并在 MuJoCo 在线验证。
现有 OCS2 控制流程保留；新增流程不连接 OCS2、不读取其轨迹、不需要 ROS 2。

## 方法与当前边界

参考 [PAINT v2 §III-C](https://arxiv.org/html/2604.12852v2#S3.SS3)：
输入为机械臂 q、dq、已经执行的位置指令历史，输出 Fx/Fy/Mz；用已知仿真交互 wrench 回归。
本实现先独立训练估计器，还没有实现论文完整的高层教师—学生 PPO/KL 控制器。
使用 TRON2/Isaac Lab 迁移原文的 ANYmal/Isaac Gym 方法。
网络隐藏层为 128/128/128，ELU；4 帧、50 Hz 为默认输入契约。
额外将 q 与位置目标的差作为内部特征，并保留绝对目标；预处理和归一化均封装到模型。

标签约定：**外部作用于负载质心的力/力矩，以负载自身坐标表示**。
这不是机械臂对底座的反作用，不是期望末端位置，也不是换算到底座原点后的力矩。
这里选择负载局部轴作为明确的迁移契约；负载倾斜时不能称为世界水平平面力。
每个样本的状态在 t+dt 测量，标签是 [t,t+dt) 实际施加的 wrench，避免提前泄漏未知外力。
模型部署时只能估计已发生的交互，不能预测伙伴突然改变的下一步外力。

负载为 0.5 kg、0.20×0.12×0.08 m 箱体，质心在夹爪局部 +x 0.15 m，带明确惯量和刚性连接。
Isaac 入口在输出目录生成独立 `robot_payload.usdc`；MuJoCo 入口生成独立 MJCF。
原 `robot_description` 文件不改。
初始实验机械臂保持名义姿态，Kp=60、Kd=6，速度目标/前馈力矩为零。
每个姿态组跑 zero、±Fx、±Fy、±Mz 七个回合；5 N/0.5 Nm 为默认诊断幅值。
外力用平滑梯形（10% 上升、80% 保持、10% 下降）或半正弦，期间以仿真时间更新。
相同种子与位置目标属于同一组；训练/验证/测试按组划分，不能拆相邻帧或正负配对回合。

## Linux 同步

本次使用本地 Git bundle，不推送 GitHub。包包含本分支的 Git 历史、代码和初步模型；
不包含 logs 下的仿真数据、Conda 环境或 robot_description 子模块实体文件。
本机交付包路径为 `logs/paint/sync/paint-intent-pd.bundle`。

先在 **Windows PowerShell** 上传（若服务器名称不能解析，将目标替换成实际 IP）：

```powershell
Set-Location 'E:\Robot\Sustech-多人形协作\TRON2_WholeBody_LAB'
scp '.\logs\paint\sync\paint-intent-pd.bundle' matt14@server-Super-Server:~/paint-intent-pd.bundle
```

先确认 Linux 工作区干净。若有未提交代码，先自行保存，不要用 reset 覆盖。

```bash
cd ~/workspace/isaaclab_tron2/TRON2_WholeBody_LAB
git status --short
git bundle verify "$HOME/paint-intent-pd.bundle"
git fetch "$HOME/paint-intent-pd.bundle" \
  refs/heads/codex/paint-intent-pd:refs/remotes/paint-bundle/paint-intent-pd
if git show-ref --verify --quiet refs/heads/codex/paint-intent-pd; then
  git switch codex/paint-intent-pd
else
  git switch -c codex/paint-intent-pd refs/remotes/paint-bundle/paint-intent-pd
fi
git merge --ff-only refs/remotes/paint-bundle/paint-intent-pd
git submodule update --init --recursive robot_description
git branch --show-current
git log -1 --oneline
```

导入、合并时若有本地不同提交，ff-only 会停止；保留现有代码再检查差异。
如果 robot_description 已完整且对应提交一致，子模块命令无需额外下载。

激活你原来能运行 WholeBody PLAY 的 Isaac Lab 环境；当前未提供该环境名，因此交互输入。
这里不是 OCS2 的 Conda 环境。

```bash
conda env list
read -r -p 'Isaac Lab Conda 环境名: ' PAINT_ENV
conda activate "$PAINT_ENV"
python -m pip install -e exts/bipedal_locomotion -e rsl_rl
python -m pip install 'mujoco==3.2.7'
export PAINT_CHECKPOINT="$PWD/logs/rsl_rl/sfyg_tron_2a_wholebody_flat/2026-09-10_22-16-15/model_20000.pt"
test -f "$PAINT_CHECKPOINT"
export PAINT_RUN="$(date +%Y%m%d_%H%M%S)"
python -m unittest discover -s tests -p test_paint_intent.py -v
```

后续命令在同一个终端执行，变量沿用上述定义。输出目录必须为空；重跑时更新 PAINT_RUN。

## 先看 MuJoCo 初始效果

仓库附带 `examples/paint/fixed_base_demo` 初步模型，直接部署到固定底座 MuJoCo。
这是可运行的估计器观测演示，**估计结果不输出机器人动作**。

```bash
python scripts/paint/run_mujoco.py \
  --model_dir examples/paint/fixed_base_demo \
  --output_dir "logs/paint/mujoco_demo_$PAINT_RUN" \
  --seed 142 --warmup 0.5 --duration 1 --recovery 0.5
python scripts/paint/analyze_pd_pairs.py "logs/paint/mujoco_demo_$PAINT_RUN" \
  > "logs/paint/mujoco_demo_$PAINT_RUN/analysis_stdout.txt"
cat "logs/paint/mujoco_demo_$PAINT_RUN/run_status.json"
cat "logs/paint/mujoco_demo_$PAINT_RUN/pair_analysis.json"
```

有桌面显示时可加 `--viewer`，通过 SSH 无桌面时不加。
输出 NPZ 同时含已知交互力与在线估计值，可在 `pair_analysis.json` 查看误差和配对关节响应。

本机初步结果（2026-10-12）：6 个姿态组、42 个回合，全组完成；底座固定，负载 0.5 kg，
姿态目标仅 ±0.02 rad 变化，时序为 0.5 s/1 s/0.5 s。
按姿态组留出的测试 MAE 为 Fx=0.287 N、Fy=0.100 N、Mz=0.00547 Nm。
这是一组很窄的验证，不能代表跨负载、行走或真机泛化；示例模型不是可闭环控制的正式模型。

### 已复现的版本差异

2026-10-12 Linux MuJoCo 3.13.0 的 seed=142 零外力估计均值为
[-0.787 N, -4.911 N, 0.650 Nm]。本机保持脚本、模型、种子和配置不变，仅切换至 3.13.0，
复现了相同结果：第一关节响应变化即可引发明显偏置。3.2.7 下对应均值为
[-0.186 N, -0.050 N, 0.012 Nm]。这是跨运行版本泛化失败，不是模型文件损坏。
模型 SHA256 保持 adb89e4d705470a233fd310c62f1f0f757adf0c22407f5bddc3119f125b0a736。

不改动现有 limx 环境中的 MuJoCo，可将 3.2.7 安装到独立目录做对照：

```bash
python -m pip install --target logs/paint/runtime/mujoco327 --no-deps 'mujoco==3.2.7'
paint327() {
  PYTHONPATH="$PWD/logs/paint/runtime/mujoco327${PYTHONPATH:+:$PYTHONPATH}" python "$@"
}
paint327 -c 'import mujoco; print(mujoco.__version__, mujoco.__file__)'
PAINT_DEMO327="logs/paint/demo327_$(date +%Y%m%d_%H%M%S)"
paint327 scripts/paint/run_mujoco.py --model_dir examples/paint/fixed_base_demo \
  --output_dir "$PAINT_DEMO327" --seed 142 --warmup 0.5 --duration 1 --recovery 0.5
paint327 scripts/paint/analyze_pd_pairs.py "$PAINT_DEMO327"
```

此独立目录方案使用现有环境的 NumPy/Torch/GLFW 等依赖，仅覆盖 MuJoCo。
新采集记录版本信息，训练模型保留来源版本，运行不匹配时打印警告。
版本一致只用于复现初步基线，不能代替更大范围泛化验证。

## Isaac Lab：采集初始行走/受力数据

先跑零底座速度指令对照：底座仍由行走策略支撑，**不等于物理固定底座**。
入口检查模型 actor 与 encoder 完整匹配，避免仓库 runner 静默跳过不匹配权重。
新增负载会改变 critic 的质量观测；采集只执行已验证匹配的 actor 和 encoder。

```bash
export PAINT_DIAG="logs/paint/isaac_pd_diag_$PAINT_RUN"
LIVESTREAM=0 python scripts/rsl_rl/collect_paint_pd.py \
  --checkpoint_path "$PAINT_CHECKPOINT" --headless \
  --output_dir "$PAINT_DIAG" --repeats 3 --base_command 0 0 0
python scripts/paint/analyze_pd_pairs.py "$PAINT_DIAG" > "$PAINT_DIAG/analysis_stdout.txt"
cat "$PAINT_DIAG/run_status.json"
```

检查所有 21 回合都 completed，施力前配对差异接近零，施力后关节响应可测，无限位/倾角终止。
若 `paint_payload_Link` 未进入 articulation、导入质量不匹配或 actor 维度不匹配，入口直接报错。
负载偏移须在可视化中核查；不要靠增大外力解决负载导入或接触问题。
数据配对默认关闭观测噪声、执行器随机延迟和已知随机化事件，用于先定位信号。

### 首回合退出时的诊断

如果所有条件都在 2 s 外力开始前触发 tilt_limit，应先验证基础支撑，不扩大全部七条件采集。
`--payload_mode none --conditions zero` 使用原始机器人的展平副本，不增加负载，仅记录零外力支撑；
这些单条件数据不满足训练器的完整配对要求，不能作为正式意图训练组。
统一设置 `--pose_jitter 0`，依次比较无负载零指令、无负载原项目行走指令、加负载同一行走指令。
保持 PD、初始姿态、倾角限制和 checkpoint 相同。

```bash
PAINT_BASELINE_STAMP="$(date +%Y%m%d_%H%M%S)"
for CASE in no_load_zero no_load_walk load_walk; do
  MODE=none
  BASE_CMD=(0 0 0)
  if [ "$CASE" != no_load_zero ]; then BASE_CMD=(-0.29 -0.08 0); fi
  if [ "$CASE" = load_walk ]; then MODE=attached; fi
  DIR="logs/paint/${CASE}_$PAINT_BASELINE_STAMP"
  LIVESTREAM=0 python -u scripts/rsl_rl/collect_paint_pd.py \
    --checkpoint_path "$PAINT_CHECKPOINT" --output_dir "$DIR" --headless --repeats 1 \
    --payload_mode "$MODE" --conditions zero --pose_jitter 0 --base_command "${BASE_CMD[@]}"
  cat "$DIR/run_status.json"
  cat "$DIR/manifest.json"
done
```

无负载也失败时先检查支撑策略与初始状态；同一指令无负载通过、加负载失败时，再定位负载惯量和接触。
新增 NPZ 记录实际腿部动作、底座位置/姿态；manifest 记录时长、倾角和关节余量。
不得提高 max_tilt_deg 或把 rejected 改成 accepted 来消除失败。

无负载/有负载现在都先展平原始 USD 再加载，避免把引用解析路径的变化混入负载对照。
展平无负载资产保留原机器人的物理属性和关节拓扑；输出 `robot_unloaded.usdc`，不改子模块文件。
`[ASSET]` 记录打开、展平和导出阶段；环境创建超过 60 秒自动打印堆栈。
这消除了已识别的加载方式差异，仍需 Linux 验证是否消除启动停顿。

2026-10-12 已修复回合重置报错 `Inplace update to inference tensor outside InferenceMode`：
仿真步和策略执行使用 `torch.no_grad()`，不再在 inference_mode 中生成需要跨回合原地更新的步态状态。
这不改变策略权重、PD 参数或物理限制。纯模型导出仍可使用 inference_mode。
重置报错修复后仍需核查 manifest 中首回合的 termination reason；回合提前结束不等于完成。

采集器在进入环境构造前写入 `run_status.json`，默认 complete=false。
首 3 步及之后每 50 步记录 `[PAINT_STAGE]`，区分准备观测、推理、physics_step、
read_response、保存回合。Python 异常在环境清理前打印完整堆栈，并将状态设为 failed。
原生崩溃不能由 Python 捕获，但上次写入的 stage 可定位到最后一个检查点。
退出码 0 不能代替 collection_finished/complete 与回合数量检查。
如果单独重敲 `PAINT_RC=${PIPESTATUS[0]}`，可能取得后一个命令的状态；必须紧接采集流水线赋值。

```bash
PAINT_STAMP="$(date +%Y%m%d_%H%M%S)"
PAINT_DEBUG="logs/paint/isaac_debug_$PAINT_STAMP"
PAINT_LOG="logs/paint/isaac_debug_$PAINT_STAMP.log"
set -o pipefail
PYTHONFAULTHANDLER=1 LIVESTREAM=0 python -u scripts/rsl_rl/collect_paint_pd.py \
  --checkpoint_path "$PAINT_CHECKPOINT" --output_dir "$PAINT_DEBUG" \
  --headless --repeats 1 --base_command 0 0 0 \
  2>&1 | tee "$PAINT_LOG"
PAINT_RC=${PIPESTATUS[0]}
echo "exit=$PAINT_RC"
cat "$PAINT_DEBUG/run_status.json"
tail -n 120 "$PAINT_LOG"
```

日志位于采集目录之外，避免 tee 创建的文件触发“输出目录必须为空”检查。

诊断通过后采 30 个姿态组，再采慢速行走条件：

```bash
export PAINT_STAND="logs/paint/isaac_pd_stand_$PAINT_RUN"
export PAINT_WALK="logs/paint/isaac_pd_walk_$PAINT_RUN"
LIVESTREAM=0 python scripts/rsl_rl/collect_paint_pd.py \
  --checkpoint_path "$PAINT_CHECKPOINT" --headless \
  --output_dir "$PAINT_STAND" --repeats 30 --base_command 0 0 0
LIVESTREAM=0 python scripts/rsl_rl/collect_paint_pd.py \
  --checkpoint_path "$PAINT_CHECKPOINT" --headless \
  --output_dir "$PAINT_WALK" --repeats 30 --base_command -0.10 0 0
python scripts/paint/analyze_pd_pairs.py "$PAINT_STAND" > "$PAINT_STAND/analysis_stdout.txt"
python scripts/paint/analyze_pd_pairs.py "$PAINT_WALK" > "$PAINT_WALK/analysis_stdout.txt"
```

负底座 x 速度沿用现有项目的行走方向约定。训练读取全部成功的完整七回合组；
失败组保留在 rejected，不能用幸存回合替代完整配对。
当前没有自动化大范围臂动作、混合多轴外力或全套域随机化，需在初步结果后扩展。

## 训练与在线验证

```bash
export PAINT_MODEL="logs/paint/intent_model_$PAINT_RUN"
python scripts/paint/train_intent_estimator.py \
  --datasets "$PAINT_STAND" "$PAINT_WALK" \
  --output_dir "$PAINT_MODEL" --history 4 --epochs 100 --device cpu
cat "$PAINT_MODEL/metrics.json"
cat "$PAINT_MODEL/splits.json"
```

输出包含 `intent_estimator.ts`、嵌入式归一化、metadata、数据来源和组级划分。
同时报告三个分量的 MAE/RMSE、激励轴误差、零外力误报，以及常零预测器基线。
`offline_baseline_gate_passed` 只表示三个总体 MAE 优于常零基线，不代表部署验收。
参数只按验证组选择，测试组只用于最终评估。

新姿态种子下在 Isaac Lab 在线观测：

```bash
export PAINT_SHADOW="logs/paint/isaac_pd_shadow_$PAINT_RUN"
LIVESTREAM=0 python scripts/rsl_rl/collect_paint_pd.py \
  --checkpoint_path "$PAINT_CHECKPOINT" --headless \
  --output_dir "$PAINT_SHADOW" --repeats 3 --seed 142 \
  --base_command -0.10 0 0 --intent_model_dir "$PAINT_MODEL"
python scripts/paint/analyze_pd_pairs.py "$PAINT_SHADOW" > "$PAINT_SHADOW/analysis_stdout.txt"
```

## MuJoCo：训练模型迁移和行走策略接入

2026-10-12 初步结果：Isaac 行走在线观测 21/21 完成，外力回合全时段 MAE 约
0.066 N/0.065 N/0.0079 Nm；同一模型转到固定底座 MuJoCo 后仍有 21/21 回合完成，
但 MAE 升至 2.49 N/0.81 N/0.079 Nm。这是估计迁移未通过，不能用 accepted 回合数代替精度验收。
固定底座同时改变了运动分布；后续应先验证 MuJoCo 行走零外力基线，再讨论模型适配。
不要覆盖已有 Isaac 模型。

`evaluate_intent_rollouts.py` 汇总全部有效估计，包括 zero 回合、施力前偏置、激励轴误差；
接受/拒绝回合分别报告，可用于单独的 zero 条件，不要求完整七条件配对：

完整报告保存为 `intent_evaluation.json`，终端仅打印 summary。

先将 Isaac 训练的估计器放到固定底座 MuJoCo 中检查估计误差：

```bash
export PAINT_MJ_FIXED="logs/paint/mujoco_fixed_$PAINT_RUN"
python scripts/paint/run_mujoco.py --model_dir "$PAINT_MODEL" \
  --output_dir "$PAINT_MJ_FIXED" --seed 142
python scripts/paint/analyze_pd_pairs.py "$PAINT_MJ_FIXED" > "$PAINT_MJ_FIXED/analysis_stdout.txt"
python scripts/paint/evaluate_intent_rollouts.py "$PAINT_MJ_FIXED"
```

导出你给出的行走 checkpoint，再开启自由底座的行走验证：

```bash
export PAINT_LOCO="logs/paint/loco_export_$PAINT_RUN"
LIVESTREAM=0 python scripts/rsl_rl/collect_paint_pd.py \
  --checkpoint_path "$PAINT_CHECKPOINT" --headless \
  --output_dir "$PAINT_LOCO" --export_locomotion_only
export PAINT_MJ_WALK="logs/paint/mujoco_walk_$PAINT_RUN"
python scripts/paint/run_mujoco.py \
  --model_dir "$PAINT_MODEL" --locomotion_dir "$PAINT_LOCO" \
  --base_command -0.25 0 0 --output_dir "$PAINT_MJ_WALK" --seed 142 --conditions zero
cat "$PAINT_MJ_WALK/run_status.json"
cat "$PAINT_MJ_WALK/manifest.json"
python scripts/paint/evaluate_intent_rollouts.py "$PAINT_MJ_WALK"
```

此处先做 zero 单条件诊断：检查 max_tilt_deg、实际底座平均速度和位移，再检查零力误报。
只有行走稳定且零力偏置可接受，才去掉 --conditions zero 运行七条件估计对照。
单条件不要运行 analyze_pd_pairs，它会按设计报告没有完整配对组。

MuJoCo adapter 对齐名义关节顺序、左右 yaw 零点、50 Hz 控制、10 帧腿部历史、
观测缩放、ELU 导出的 actor/encoder、足式 PD 参数及零 wrench 预测输入。
目前仅支持本项目 flat WholeBody 的 420→3 encoder 和 78→10 actor 契约；不支持 rough 高度扫描。
原 MJCF 与 USD 的接触/惯量/执行器模型不同，因此带行走的 sim-to-sim 效果仍须 Linux 实测，
本机没有用户的 checkpoint，不能声称已验证行走。
仅调估计器时不要通过它改变行走命令；先测量误差与稳定性，再开发高层控制闭环。

## 真机后续接口

真机未到，当前不写硬件 SDK 控制命令。可复用 `observe_intent.py` 的 JSONL/标准输入入口：

```bash
python scripts/paint/observe_intent.py --model_dir "$PAINT_MODEL" --input telemetry.jsonl
```

输入每行 `{ "time": 0.02, "q": [六个弧度值], "dq": [六个弧度每秒值],
"previous_applied_position": [上个控制间隔实际位置目标], "reset": false }`。
上述中文数组说明需替换为真实数值，不能直接作为 JSON 执行。
每次 reset 或采样时间跳变清空历史；历史不足时 valid=false。
实际部署前确认关节顺序、负载坐标、PD 增益、采样周期和有效的力参考。
不自动把输出解释成底座速度，也不将它当成力传感器真值。
