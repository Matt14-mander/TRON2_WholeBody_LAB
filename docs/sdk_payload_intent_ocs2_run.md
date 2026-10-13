# SDK：附加负载 + intent observer + OCS2 future wrench

2026-10-12 新增实现。负载与 estimator 在本地源码中已接入；服务器需同步代码后才能使用新参数。future wrench 复用已有 OCS2 CSV 接口，不来自 estimator。estimator 仅观测，不改变任何策略输入或动作。

## 同步

交付包 `logs/paint/sdk_payload_intent_update.zip` 仅含本次所需代码、测试和本说明，不含策略、SDK、机器人资产或正式 estimator。包内路径以三个项目的父目录为起点；simulator.py 和 SFYG variant 包括当前工作区已有改造，保留 SFYG 原有 1 ms 显式 MIT PD。

Windows PowerShell 上传：

```powershell
scp 'E:\Robot\Sustech-多人形协作\TRON2_WholeBody_LAB\logs\paint\sdk_payload_intent_update.zip' matt14@server-Super-Server:~/sdk_payload_intent_update.zip
```

Linux 在安装前停掉旧仿真/控制器。备份包内涉及的服务器文件，再覆盖；不要覆盖服务器的 ONNX、params.yaml、子模块或模型目录：

```bash
export UPDATE_DIR="$HOME/sdk_payload_intent_update_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$UPDATE_DIR"
unzip "$HOME/sdk_payload_intent_update.zip" -d "$UPDATE_DIR"
export UPDATE_DIR
python - <<'PY'
import os, shutil
from pathlib import Path
source = Path(os.environ['UPDATE_DIR'])
workspace = Path.home() / 'workspace/limx_deploy'
backup = source / 'server_backup'
for repo in ['tron2_mujoco_sim', 'tron2_rl_deploy_python']:
    if not (workspace / repo).is_dir():
        raise SystemExit(f'Missing project: {workspace / repo}')
for repo in ['tron2_mujoco_sim', 'tron2_rl_deploy_python']:
    for file in (source / repo).rglob('*'):
        if not file.is_file():
            continue
        relative = file.relative_to(source)
        target = workspace / relative
        if target.exists():
            old = backup / relative
            old.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, old)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, target)
print('Installed; original files saved at', backup)
PY
```

如果服务器包内这些文件有额外改造，在 source/server_backup 中与新版比较合并后运行。

## 查找模型和 OCS2 CSV

```bash
find "$HOME/workspace" -maxdepth 9 -type f -name intent_estimator.ts -print 2>/dev/null
find "$HOME/workspace" -maxdepth 9 -type f -name '*.csv' \
  -exec grep -l -m 1 '^time,arm_q1,arm_q2,arm_q3,arm_q4,arm_q5,arm_q6,' {} + \
  2>/dev/null
```

使用正式 Isaac 训练模型目录，避免选择 fixed_base_demo。CSV 必须是现有 52 列契约，臂 q/dq/tau、速度与 joint margin 校验沿用原 loader。不能仅因为 CSV 列名正确就认为轨迹适合带负载：需确认轨迹假设的负载与仿真一致；旧无负载 OCS2 预测在加负载后可能不准确。

## 终端 A：带负载仿真

```bash
conda activate limx
cd "$HOME/workspace/limx_deploy/tron2_mujoco_sim"
export ROBOT_TYPE=SFYG_TRON2A
export ROBOT_IP=127.0.0.1
export PAINT_MJ_RUNTIME="$HOME/workspace/isaaclab_tron2/TRON2_WholeBody_LAB/logs/paint/runtime/mujoco327"
PYTHONPATH="$PAINT_MJ_RUNTIME${PYTHONPATH:+:$PYTHONPATH}" python -u simulator.py \
  --payload-mass 0.5 --payload-dimensions 0.20 0.12 0.08 \
  --payload-offset 0.15 0 0
```

有桌面用上述命令观察稳定性；无桌面加 --headless。载荷在 gripper_base_Link 局部 +x 0.15 m 刚性连接，自带质量和箱体惯量，屏蔽与夹爪本体/手指的碰撞；不修改资产子模块。显示默认 0.5 kg 负载参数；若训练负载不同，显式传入真实值。

## 终端 B：estimator + OCS2

激活 limx 后将两个变量替换成找到的真实绝对路径，模型变量是目录，CSV 变量是文件：

```bash
conda activate limx
cd "$HOME/workspace/limx_deploy/tron2_rl_deploy_python"
export ROBOT_TYPE=SFYG_TRON2A
export INTENT_MODEL='/absolute/path/to/formal_isaac_model'
export OCS2_CSV='/absolute/path/to/trajectory.csv'
export INTENT_LOG="$PWD/logs/intent_$(date +%Y%m%d_%H%M%S).jsonl"
test -f "$INTENT_MODEL/metadata.json"
test -f "$INTENT_MODEL/intent_estimator.ts"
test -f "$OCS2_CSV"
```

不要带占位路径执行以下命令；先打印 metadata 和 CSV 开头确认来源。检查 CSV 的 wrench 是否实际非零并计算覆盖完整轨迹的运行时长：

```bash
export RUN_DURATION="$(python - <<'PY'
import os
import numpy as np
data = np.loadtxt(os.environ['OCS2_CSV'], delimiter=',', skiprows=1, ndmin=2)
if data.shape[1] != 52 or len(data) < 2 or not np.isfinite(data).all():
    raise SystemExit('Invalid 52-column CSV')
print(float(data[-1, 0]) + 8.0)
PY
)"
python - <<'PY'
import os, numpy as np
data = np.loadtxt(os.environ['OCS2_CSV'], delimiter=',', skiprows=1, ndmin=2)
print('future wrench max(abs):', np.max(np.abs(data[:, 22:52])))
print('run duration:', os.environ['RUN_DURATION'])
PY
python -u main.py 127.0.0.1 --start-controller \
  --ocs2-trajectory "$OCS2_CSV" --ocs2-start-delay 3 \
  --safe-walk-command -0.25 0 0 --ocs2-terminal-command -0.25 0 0 \
  --intent-model-dir "$INTENT_MODEL" --intent-log "$INTENT_LOG" \
  --duration "$RUN_DURATION"
```

不要同时传 --base-command，已有轨迹接口会拒绝。safe-walk-command 用刚验证过的低速步态，替代 CSV 的零速度片段；OCS2 臂部轨迹和 future wrench 仍取自 CSV。没有非零 wrench 的 CSV 只验证接口，不构成受力补偿验证。warmup 时 controller 按既有逻辑把 wrench 置零。

## 检查结果与限制

```bash
python - <<'PY'
import os, json
from collections import Counter
rows = [json.loads(line) for line in open(os.environ['INTENT_LOG'])]
samples = [r for r in rows if r.get('type') == 'sample']
valid = [r for r in samples if r['valid']]
print('samples / valid:', len(samples), '/', len(valid))
print('history resets:', Counter(r['history_reset_reason'] for r in samples if r['history_reset_reason']))
print('last valid estimate:', valid[-1]['interaction_wrench_payload_local'] if valid else None)
print('non-PD-only intervals:', sum(not r['pd_only_training_contract'] for r in samples))
print('nonzero recorded OCS2 wrench:', any(any(abs(v)>1e-9 for frame in r['preceding_future_wrench'] for v in frame) for r in samples))
PY
```

valid 仅指输入历史和数值有效，不代表估计准确。SDK 无执行确认和统一仿真时钟，日志采用主机接收时间配对前一发送目标，记录 previous_sent_position 而非宣称真实已执行目标；真实端到端延迟仍需评价。采样间隔偏离 18–22 ms、旧/重复状态时清历史；如果有效样本很少，先排查 CPU 推理/调度/状态流。JSONL 使用独占创建，避免覆盖旧实验。

运行时不暂停或复位仿真；目前 SDK 回调不提供 reset epoch，暂停状态心跳可能被当作新状态。需要复位时停止两个进程并重新启动。控制器终止后仿真改为阻尼状态，不再算行走验收。

estimator 训练为纯位置 PD，OCS2 回放可能有非零 dq/tau，pd_only_training_contract=false 会标明分布变化。并且当前显式 MIT PD 与 Lab 的 native PD 不是同一响应。本轮先看有负载是否稳定、estimator 是否正常输出和 OCS2 wrench 是否进入 policy；没有仿真真值日志时不能报告 MAE 或宣称 estimator 精度通过。

后续量化验证需要另加负载 COM 已知扰动和真值日志；不能以 OCS2 future wrench 作为 estimator 的负载交互真值。对照可使用 --ocs2-zero-wrench 保留相同臂部/速度轨迹，将 policy wrench 清零，日志字段对应原 OCS2 preceding solution 的实际选项结果。
