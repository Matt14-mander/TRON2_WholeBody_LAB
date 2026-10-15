# TRON2 WholeBody 论文对齐训练 v3

v3 修复了 v2 在约 12k iteration 后可能发生策略崩塌的训练链路问题。任务入口仍是
`Isaac-Limx-SFYG-TRON2A-WholeBody-Flat-v0`，但必须从零训练，不能从 v2 checkpoint 续训。

## 修复内容

- actor 均值最后经过 `5*tanh(x/5)`，在零点附近保持单位斜率；WholeBody 环境实际执行的动作再限制到
  `[-5, 5]`。配合
  `action_scale=0.25 rad`，仍保留最多约 `1.25 rad` 的关节目标偏移，同时阻止动作发散到 80～100。
- TensorBoard 新增 `Policy/raw_action_abs_max`、`Policy/mean_action_abs_max` 和
  `Policy/action_saturation_fraction`，用于发现动作分布持续撞限。
- PPO 自适应学习率限制在 `1e-5 ~ 3e-4`，并启用 KL early stop，避免旧配置把学习率升到
  `1e-2` 后破坏已经稳定的策略。
- history encoder 使用独立梯度裁剪，并把其 optimizer state 写入 checkpoint，保证续训语义完整。
- 25% 的训练 episode 使用严格零 wrench，覆盖 PLAY 和 MuJoCo 启动时的零预测工况。
- checkpoint 中的 iteration 在保存前更新，不再把中间模型错误记录为 iteration 0。

Blind/Base 任务保留原配置；这些保护只在 WholeBody v3 runner 中启用。

## 启动检查

先运行小规模检查：

```bash
cd ~/workspace/isaaclab_tron2/TRON2_WholeBody_LAB
conda activate limx

python scripts/rsl_rl/train.py \
  --task Isaac-Limx-SFYG-TRON2A-WholeBody-Flat-v0 \
  --num_envs 256 \
  --headless \
  --seed 42 \
  --max_iterations 300
```

新日志目录为：

```text
logs/rsl_rl/sfyg_tron_2a_wholebody_paper_v3_flat/<timestamp>/
```

启动检查通过后，用 4096 环境从零运行完整训练。不要传 `--resume`：

```bash
python scripts/rsl_rl/train.py \
  --task Isaac-Limx-SFYG-TRON2A-WholeBody-Flat-v0 \
  --num_envs 4096 \
  --headless \
  --seed 42
```

## 训练验收

不能只选最后一个 checkpoint。每 500 iteration 对保存模型做固定 seed、固定命令的 PLAY 验证，至少覆盖
站立、前进、后退、横移和转向，并按 timeout 比例、base contact、动作饱和率和速度误差选择模型。

重点告警条件：

- `Policy/mean_action_abs_max` 长期接近 5；
- `Policy/action_saturation_fraction` 持续上升；
- timeout 比例下降，同时 base contact 或 action-out termination 上升；
- 学习率没有落在 `1e-5 ~ 3e-4`；
- 某一运动方向稳定性明显差于其他方向。

导出的 v3 actor 自带 `5*tanh(x/5)`，因此 ONNX 确定性输出也满足 `[-5, 5]`。进入 MuJoCo 前仍应核对关节顺序、
默认姿态、action scale、控制频率和 PD 参数，并优先测试验证最好的 checkpoint，而不是机械选择 `model_20000.pt`。
