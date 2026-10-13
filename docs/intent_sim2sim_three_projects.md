# Intent estimator：三个改造项目的 MuJoCo sim2sim 操作指南

核对日期：2026-10-12。针对工作区根目录下的 `TRON2_WholeBody_LAB`、`tron2_mujoco_sim`、`tron2_rl_deploy_python`；不针对 `TRON2-Sim2sim/` 或其他 third_party 副本。下文运行命令以 Linux Bash 为主，适合已跑通 Isaac Lab 的机器；路径按实际位置修改。

## 1. 当前能做什么

推荐顺序：**同一 checkpoint 导出 → 无负载行走 → 带负载行走 → estimator 旁路观测 → 七条件误差评估 → SDK 双进程旁路接入 → 高层闭环**。

“旁路观测”表示模型在线推理并记录结果，但不修改行走速度、机械臂目标或 policy wrench。

| 项目 | 当前职责和入口 | 已有能力 | 当前缺口 |
|---|---|---|---|
| TRON2_WholeBody_LAB | `collect_paint_pd.py`、`scripts/paint/run_mujoco.py` | Isaac 数据采集与训练；TorchScript estimator；直接 MuJoCo 固定/自由底座验证、负载生成与误差记录 | 迁移精度和行走仍需按实际 checkpoint 验收；没有 estimator 高层闭环 |
| tron2_mujoco_sim | `simulator.py`、`tron2_sim/variants/sfyg.py` | SFYG 18 关节、SDK 状态/IMU、1 ms 物理步长、已知外力验证 | 原始 SFYG 模型未自动生成 PAINT 的负载；没有 PAINT 配对采集与 estimator 评价链 |
| tron2_rl_deploy_python | `main.py`、`controllers/SFYGController.py` | 500 Hz 发布、50 Hz ONNX 推理、WholeBody 行走和 OCS2 CSV 回放 | 没有 `--intent-model-dir` 或 estimator 在线调用、日志接口 |

因此现在可以直接完成第一条 MuJoCo 验证链；**不能仅把 `intent_estimator.ts` 复制到 deploy 项目就认为 SDK 部署已完成**。第 6 节列出第二条链路所需改造，本指南不修改控制代码。

## 2. 先锁定版本、模型和接口

本地核对时三个 HEAD 分别为 Lab `abcb974`、sim `a78a1d7`、deploy `61cdf72`。HEAD 不是完整交付状态：Lab 中 `scripts/paint/`、`collect_paint_pd.py`、`paint_intent.py`、`paint_payload.py`、`paint_run_status.py`、`ocs2_reset.py`、相关测试和文档目前是未跟踪文件。同步时包含工作区改造文件，并检查资产子模块；仅 pull HEAD 会遗漏 PAINT 功能。Linux 如有自己的更新，以 Linux 实际状态为准。

```bash
# 修改为三个项目共同的父目录，不要混用另外一套副本
export TRON_ROOT="$HOME/workspace/isaaclab_tron2"
export LAB="$TRON_ROOT/TRON2_WholeBody_LAB"
export SIM="$TRON_ROOT/tron2_mujoco_sim"
export DEPLOY="$TRON_ROOT/tron2_rl_deploy_python"
for REPO in "$LAB" "$SIM" "$DEPLOY"; do
  git -C "$REPO" status --short
  git -C "$REPO" rev-parse HEAD
  git -C "$REPO" submodule status
done
test -f "$LAB/scripts/paint/run_mujoco.py"
test -f "$LAB/exts/bipedal_locomotion/bipedal_locomotion/controllers/paint_intent.py"
test -f "$SIM/robot-description/tron2a/SFYG_TRON2A/xml/robot.xml"
test -f "$LAB/robot_description/tron2/SFYG_TRON2A/xml/robot.xml"
```

保存正式 Isaac 模型目录的完整内容：`intent_estimator.ts`、`metadata.json`、`metrics.json`、`splits.json`、`source_contracts.json`。`weights.pt` 用于后续训练，不代替部署的 `.ts`。`examples/paint/fixed_base_demo` 是窄范围 MuJoCo 示例，不代替自己的 Isaac 模型。

| 接口 | 固定契约 |
|---|---|
| intent 输入 | 每帧 q(6)、dq(6)、上个已执行控制间隔的位置目标(6)，按 arm1→arm6；默认 4 帧，最旧到最新，共 72 维 |
| intent 周期 | 0.02 s / 50 Hz；实际 history 以 metadata 为准 |
| intent 预处理 | tracking error 与归一化已嵌入 TorchScript；不要再手动缩放 |
| intent 输出 | `[Fx, Fy, Mz]`，N、N、N·m；外部作用于负载 COM，以负载局部坐标表示；估计前一控制间隔 |
| 腿部 encoder | 420→3，10 帧 × 42 维本体观测 |
| 腿部 policy | 78→10：latent(3) + proprio(42) + future wrench(30) + base command(3) |
| policy wrench | 5 个时间点 0/0.2/0.4/0.6/0.8 s 的六维预测；与 intent 的三维历史交互力不是同一接口 |
| SDK 线序 | 腿 0–9、臂 10–15、夹爪 16–17，完整带名 18 维命令 |
| 默认机械臂/负载 | Kp=60、Kd=6；0.5 kg、0.20×0.12×0.08 m，夹爪局部 +x 0.15 m |

不要将 intent 的三维输出重复五遍填入 30 维 future wrench。即便旋转/平移坐标，也不能自动得到机械臂对底座的反作用和未来预测。现阶段保持原来的零 wrench 或 OCS2 输出，先评估 estimator。

## 3. 在 Isaac 环境导出同一套行走模型

激活已经能运行 Isaac Lab 的环境。将下列两条路径替换为**自己已成功验证的 checkpoint 与模型目录**，不是示例模型。

```bash
cd "$LAB"
export PAINT_CHECKPOINT="/absolute/path/to/model_20000.pt"
export PAINT_MODEL="/absolute/path/to/isaac_intent_model"
export PAINT_RUN="$(date +%Y%m%d_%H%M%S)"
export PAINT_LOCO="$LAB/logs/paint/loco_export_$PAINT_RUN"
test -f "$PAINT_CHECKPOINT"
test -f "$PAINT_MODEL/intent_estimator.ts"
test -f "$PAINT_MODEL/metadata.json"
cat "$PAINT_MODEL/metadata.json"
LIVESTREAM=0 python scripts/rsl_rl/collect_paint_pd.py \
  --checkpoint_path "$PAINT_CHECKPOINT" --headless \
  --output_dir "$PAINT_LOCO" --export_locomotion_only
cat "$PAINT_LOCO/export.json"
```

产物是 `policy.pt`、`encoder.pt` 和 `export.json`。入口严格检查 checkpoint 的 actor/encoder 匹配。后续 Lab MuJoCo 直接加载这些 TorchScript 文件，不能把 ONNX 文件改后缀代替。

所有输出目录使用新名字；脚本拒绝非空目录。切换终端后重新设置这些环境变量。

## 4. 先跑 Lab 自带的直接 MuJoCo 链路

### 4.1 环境

可以在独立 MuJoCo 环境运行这一节，不需要启动 Isaac Sim、ROS、OCS2 或 LimX SDK。安装与自己模型兼容的 Torch；其余依赖为 NumPy、MuJoCo。已有可运行环境直接沿用。

```bash
python -m pip install 'mujoco==3.2.7' numpy torch
python -c 'import mujoco, torch, numpy; print(mujoco.__version__, torch.__version__, numpy.__version__)'
cd "$LAB"
python -m unittest discover -s tests -p test_paint_intent.py -v
```

将 MuJoCo 3.2.7 作为初始对照版本，是因为已有项目文档记录了 3.13.0 的显著零力偏置；这不是保证 Isaac→MuJoCo 成功。单独建环境可避免改动现有 SDK 环境。Torch 版本也应记录。

默认 `--arm_pd_backend native` 使用原生位置/速度反馈和 implicitfast，缓解低惯量腕部数值不稳定。它与旧 explicit 数据的动态响应不同，必须重新评价误差。若复现旧 fixed_base_demo 的旧指标，使用 explicit；不要把旧误差指标套到 native。

### 4.2 无负载、无 estimator：先确认行走策略迁移

以下 -0.25 是现有指南中的诊断值，并非对所有 checkpoint 已验证的稳定速度。若自己在 Isaac 的成功配置不同，先用相同速度、姿态和时长做对照。

```bash
export PAINT_MJ_NOLOAD="$LAB/logs/paint/mj_noload_$PAINT_RUN"
python scripts/paint/run_mujoco.py \
  --locomotion_dir "$PAINT_LOCO" --payload_mode none --conditions zero \
  --base_command -0.25 0 0 --pose_jitter 0 --seed 142 --repeats 1 \
  --output_dir "$PAINT_MJ_NOLOAD"
cat "$PAINT_MJ_NOLOAD/run_status.json"
cat "$PAINT_MJ_NOLOAD/manifest.json"
```

有桌面可加 `--viewer`。未提供 locomotion_dir 时底座是固定的；提供后才是自由底座行走。无负载只允许 zero 条件。

### 4.3 带负载、无 estimator：隔离负载影响

```bash
export PAINT_MJ_LOAD="$LAB/logs/paint/mj_load_$PAINT_RUN"
python scripts/paint/run_mujoco.py \
  --locomotion_dir "$PAINT_LOCO" --payload_mode attached --conditions zero \
  --base_command -0.25 0 0 --pose_jitter 0 --seed 142 --repeats 1 \
  --output_dir "$PAINT_MJ_LOAD"
cat "$PAINT_MJ_LOAD/manifest.json"
```

如果自己的训练负载不是默认值，在所有相关命令里显式传相同的 `--payload_mass`、`--payload_dimensions`、`--payload_offset`，PD 和时序也保持一致。每次保存的 `contract.json` 是本次实际配置证据。

无负载失败：先查行走模型、关节映射、初始姿态、观测和动力学。无负载通过但带负载失败：查负载惯量、位置、机械臂响应和接触。此时还没有 estimator 参与，不能把摔倒归因于它。

### 4.4 接入自己的 estimator，先测零力误报

```bash
export PAINT_MJ_ZERO="$LAB/logs/paint/mj_intent_zero_$PAINT_RUN"
python scripts/paint/run_mujoco.py \
  --model_dir "$PAINT_MODEL" --locomotion_dir "$PAINT_LOCO" \
  --payload_mode attached --conditions zero --base_command -0.25 0 0 \
  --pose_jitter 0 --seed 142 --repeats 3 --output_dir "$PAINT_MJ_ZERO"
python scripts/paint/evaluate_intent_rollouts.py "$PAINT_MJ_ZERO"
```

查看 `intent_evaluation.json` 的有效样本、zero 误报与施力前偏置。zero 单条件不要调用 `analyze_pd_pairs.py`，它要求完整七条件配对。

可补充固定底座定位，但固定底座改变了训练运动分布，不应当作最终行走迁移验收：

```bash
export PAINT_MJ_FIXED="$LAB/logs/paint/mj_intent_fixed_$PAINT_RUN"
python scripts/paint/run_mujoco.py --model_dir "$PAINT_MODEL" \
  --seed 142 --repeats 3 --output_dir "$PAINT_MJ_FIXED"
python scripts/paint/evaluate_intent_rollouts.py "$PAINT_MJ_FIXED"
```

### 4.5 稳定后运行七条件

```bash
export PAINT_MJ_PAIRS="$LAB/logs/paint/mj_intent_pairs_$PAINT_RUN"
python scripts/paint/run_mujoco.py \
  --model_dir "$PAINT_MODEL" --locomotion_dir "$PAINT_LOCO" \
  --base_command -0.25 0 0 --seed 142 --repeats 3 \
  --output_dir "$PAINT_MJ_PAIRS"
python scripts/paint/analyze_pd_pairs.py "$PAINT_MJ_PAIRS"
python scripts/paint/evaluate_intent_rollouts.py "$PAINT_MJ_PAIRS"
```

默认每组 zero、±Fx、±Fy、±Mz，共 7 回合；3 组共 21 回合。默认幅值 5 N/0.5 N·m，warmup/duration/recovery 各 2 s。若原 Isaac 验证配置不同，传相同参数。

完成回合数与误差是两个条件。既要检查 run_status/manifest 的完成、终止原因、倾角、关节限位、实际速度和位移，也要分别评价 Fx/Fy/Mz 的 MAE/RMSE、符号、动态滞后、零力偏置。具体容许误差由后续控制需求决定；总体平均值不能掩盖某一轴失效。

已有文档曾记录 Isaac 在线低误差、MuJoCo 固定底座误差明显升高。这是历史诊断，不是你当前模型的测量结果。本指南没有运行你的 Linux checkpoint，也不声称迁移已经通过。

## 5. 跑通 sim + deploy 的 SDK 行走基线

这一节验证另两项目的现有链路，**尚不含 estimator**。Lab 直接 MuJoCo 的通过结果不能自动代表 SDK 桥通过。

### 5.1 用第 3 节的同一网络生成 ONNX

Lab 的 play.py 已有 ONNX 导出；为了确保与第 3 节完全相同，也可在 Torch 环境对已导出的 TorchScript 做以下转换。保存到新目录，检查后才替换 deploy 的文件。

```bash
python -m pip install onnx onnxruntime
export PAINT_ONNX="$PAINT_LOCO/onnx"
mkdir -p "$PAINT_ONNX"
python - <<'PY'
import os
from pathlib import Path
import numpy as np
import torch
import onnx
import onnxruntime as ort
source = Path(os.environ['PAINT_LOCO'])
dest = Path(os.environ['PAINT_ONNX'])
torch.manual_seed(0)
for name, width in [('encoder', 420), ('policy', 78)]:
    net = torch.jit.load(str(source / f'{name}.pt'), map_location='cpu').eval()
    sample = torch.zeros(1, width)
    path = dest / f'{name}.onnx'
    torch.onnx.export(net, sample, str(path), input_names=['input'],
                      output_names=['output'], opset_version=17, dynamo=False)
    onnx.checker.check_model(onnx.load(str(path)))
    session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    with torch.no_grad():
        for x in [sample, torch.randn(1, width) * 0.1]:
            actual = session.run(None, {'input': x.numpy()})[0]
            np.testing.assert_allclose(actual, net(x).numpy(), rtol=1e-4, atol=1e-5)
    print(name, session.get_inputs()[0].shape, session.get_outputs()[0].shape)
PY
```

如果 Torch/ONNX 导出器报兼容问题，先解决导出，不换用不同来源的现成网络。两次数值对照只检查导出一致性，不能证明行走稳定。

在替换前备份现有 policy.onnx/encoder.onnx；保持 params.yaml。确认配置的动作缩放、PD、默认姿态和观测缩放与训练一致。然后：

```bash
cp "$PAINT_ONNX/policy.onnx" "$DEPLOY/controllers/model/SFYG_TRON2A/policy.onnx"
cp "$PAINT_ONNX/encoder.onnx" "$DEPLOY/controllers/model/SFYG_TRON2A/encoder.onnx"
```

目录原有 SHA256SUMS 若仍指向旧模型，应在记录新来源和校验值时一并更新；不要沿用旧 checksum 宣称新模型已校验。

### 5.2 SDK 环境

在独立部署环境安装，不需要完整 Isaac Lab：

```bash
python -m pip install 'mujoco==3.2.7' numpy scipy pyyaml onnxruntime pygame
python -m pip install --no-deps "$SIM/limxsdk-lowlevel/python3/amd64/limxsdk-4.0.2-py3-none-any.whl"
python -c 'import mujoco, limxsdk, onnxruntime; print(mujoco.__version__, limxsdk.__file__)'
```

以上 wheel 是当前本地 pyproject 指定的 Linux x86_64 版本；ARM64 用相应 aarch64 wheel。若 Linux 子模块版本不同，按实际 wheel 安装并核查 SFYG 所需接口。README 中 SDK 版本文字和当前 pin 存在差异，以实际源码、wheel 和导入结果为依据。

### 5.3 先启动仿真，再启动控制器

终端 A（有桌面去掉 headless）：

```bash
cd "$SIM"
export ROBOT_TYPE=SFYG_TRON2A
export ROBOT_IP=127.0.0.1
python simulator.py --headless
```

终端 B（激活同样的部署环境）：

```bash
cd "$DEPLOY"
export ROBOT_TYPE=SFYG_TRON2A
python main.py 127.0.0.1 --start-controller \
  --base-command -0.25 0 0 --duration 20
```

确认日志 `encoder 420->3, WholeBody policy 78->10`，持续收到 state/IMU，18 维命令按名匹配，速度/倾角稳定。这里不加 OCS2 参数，policy future wrench 保持零。普通 48→10 SF policy 不兼容。

不要在控制器退出发送最终阻尼命令之后继续用机器人姿态评价行走。若运行结束且 teardown 阶段发生原生 SDK 崩溃，现有 controller 的 `--sim-sdk-exit-workaround` 可作有界本地仿真诊断；它不修复 SDK。

不依赖 SDK 的另一条诊断入口，可在 SIM 目录运行：

```bash
python tools/diagnose_sfyg_policy.py --deploy-repo "$DEPLOY" \
  --command -0.25 0 0 --duration 10
```

这个工具直接运行 ONNX 和 MuJoCo，用于隔离通信问题。`simulator.py --no-sdk` 只做物理烟雾测试，不会连接外部 controller。

## 6. 要把 estimator 接入 SDK 双进程，还需这些改造

以下是实现规格，当前 CLI 没有这些 estimator 参数。先完成第 4、5 节，再按规格实施。

### deploy 项目

1. 新增独立 intent adapter：加载 `.ts` 和 metadata，CPU eval/no_grad；核查 schema、关节顺序、周期、输入输出维度、单位和坐标语义。新增模型路径和日志路径 CLI，并传入 SFYGController。
2. 从已收到的 RobotState 按关节名读取 arm1→arm6 的 q/dq，显式映射 SDK 名称和 `arm*_Joint` 名称；不得假定回调顺序恒定。
3. 50 Hz 取一帧，配对“产生当前响应的前一控制间隔实际发送目标”。需要跟踪 500 Hz 发布的臂目标和时间；不能拿本轮新计划目标配当前状态。初始 4 帧未满输出 invalid。
4. 处理异步时间：检查 state 新鲜度、单调时间、采样间隔和指令对应关系；丢包、暂停、复位、时间跳变、模式切换清历史，不能把旧帧当连续新帧。历史窗口约 80 ms 是数据窗口，不等于可以直接等同的输出延迟。
5. 日志记录 time、q/dq、previous_applied_position、prediction、valid/reset reason、policy 输入/动作、已发命令；初版仅旁路观测，保留原有控制路径。

可以先复用 Lab 的 `observe_intent.py` JSONL 入口：deploy 增加上述时间对齐后的遥测输出，再将其输入 observer。这仍需要补遥测代码，不是当前 main.py 自带功能。

### sim 项目

1. 生成独立带负载 MJCF，复用/移植 Lab 的 `paint_payload.payload_xml` 思路，资产网格路径正确解析；不覆盖 robot-description 原模型。使 SFYG variant 能选中这个生成模型。
2. 将负载质量、惯量、尺寸、连接点及 PD backend 对齐。Lab 当前 native arm PD 与 SDK JointChannel 的显式 `kp*(q_des-q)+kd*(dq_des-dq)+tau` 不同；需明确选一种，并以该执行方式重新评估/必要时适配训练。物理步长保持 1 ms。
3. 已有 external_wrench 可旋转施力，但默认施加对象是 base_Link、标签诊断转换到底座契约。PAINT 必须施加并记录负载 COM 的局部 wrench，保留该三维标签，不能直接拿现有底座六维标签评价。
4. 统一 PAINT 的七条件、梯形/半正弦时序、回合复位和仿真时钟；SDK 命令与状态的时序也须验证。额外保存仅用于评价的真值日志，不发送给 estimator 或 policy。
5. 记录 sim time、实际力和状态时间，以回合 ID 合并 controller 预测与仿真真值，输出与 Lab 一致的误差评价。

现有 wrench-profile `ramp/step/sine` 不自动等于 PAINT 的 trapezoid/sine。对 base 或 gripper 施力也不是默认的 0.5 kg payload 实验，必须单独标记实验语义。

最后才开发高层 intent→速度/任务控制。先明确其控制目标、输入坐标与滤波/限幅/失效处理，并按该接口训练或验证；当前估计器本身不是速度策略。

## 7. 迁移失败时按这个顺序定位

| 现象 | 优先检查 |
|---|---|
| 78/420 维度报错 | 是否用了普通 SF/48维 policy；actor 与 encoder 是否来自同一 checkpoint |
| 无负载也摔倒 | yaw 零点 ±π、腿关节顺序、动作缩放/限幅、步态相位、初始历史、PD/armature/接触 |
| SDK 等不到状态 | 仿真是否先启动、同一 ROBOT_TYPE/IP、SDK 接口、是否误用了 no-sdk |
| 腕部高速振荡 | 1 ms 步长、低惯量腕部、explicit/native PD 差异和积分器；不要仅提高倾角/限位阈值 |
| zero 时估计很大 | MuJoCo 版本、负载/PD/时序分布、绝对目标、历史 reset、是否重复归一化 |
| Fx/Fy/Mz 符号或耦合异常 | 负载局部坐标与 COM、施力体、旋转/力矩参考点、是否误用了底座 reaction 标签 |
| 离线好、在线差 | 新目标泄漏、状态与上一目标错配、重复旧 state、控制周期变化和历史跨回合 |
| 回合全完成但 MAE 大 | 物理稳定通过但估计迁移失败；保留原 Isaac 模型，另建 MuJoCo 适配模型 |

若需要核查网络导出，用同一 Isaac 无负载输入采集与 MuJoCo 无负载运行做重放对照：

```bash
# 在 Isaac 环境，保持与 MuJoCo 相同 seed、速度、pose_jitter
export PAINT_ISAAC_INPUT="$LAB/logs/paint/isaac_noload_$PAINT_RUN"
LIVESTREAM=0 python scripts/rsl_rl/collect_paint_pd.py \
  --checkpoint_path "$PAINT_CHECKPOINT" --headless \
  --payload_mode none --conditions zero --pose_jitter 0 --seed 142 \
  --base_command -0.25 0 0 --repeats 1 --output_dir "$PAINT_ISAAC_INPUT"
# 在 MuJoCo/Torch 环境
python scripts/paint/compare_locomotion_inputs.py \
  --isaac_dir "$PAINT_ISAAC_INPUT" --mujoco_dir "$PAINT_MJ_NOLOAD" \
  --locomotion_dir "$PAINT_LOCO"
```

读取 `locomotion_input_comparison.json`：同一记录输入重放的 action/latent 应接近浮点误差；首帧项差异用于定位初始化与观测接口。不同物理后端的后续闭环状态本来会分化，不能把逐帧轨迹差异直接认定为网络导出错误。

## 8. 每轮实验保存什么

保存三项目实际 commit 和本地差异、模型 SHA256、metadata/来源契约、运行命令与版本、生成 MJCF、contract.json、run_status.json、manifest.json、NPZ、pair_analysis.json 和 intent_evaluation.json。候选模型用独立目录，不覆盖已在 Isaac 跑通的模型。

建议本轮先完成第 3–4 节，拿到“无负载/带负载行走 + zero误报 + 七条件误差”的完整证据；同时完成第 5 节的 SDK 行走基线。两条基线通过后，再进行第 6 节的旁路接入。
