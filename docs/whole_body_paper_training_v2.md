# TRON2 WholeBody 论文对齐训练 v2

## 本次修复的训练语义

`Isaac-Limx-SFYG-TRON2A-WholeBody-Flat-v0` 现在按参考论文的解耦方式训练：

- PPO 只控制 10 个腿部关节，动作维数不变。
- 训练场景使用不含机械臂刚体的 SF TRON2A；机械臂对底座的作用由预测 wrench 代替。
- wrench 的二次序列在世界坐标系生成，策略收到的是按当前底座姿态转换后的 5 个机体系 wrench，预测时刻为 `0.0/0.2/0.4/0.6/0.8 s`。
- critic 额外看到干净预测和包含未观测扰动的当前 wrench；actor 看到带噪声的预测序列。
- gait phase 观测与 gait reward 共用同一个相位状态，站立时固定在 stance phase。
- Play 环境仍加载完整 SFYG 机械臂；OCS2 负责机械臂，向 locomotion policy 注入同一机体系约定的未来 wrench。

训练时不运行 OCS2。旧实现同时保留真实机械臂质量/重力和完整合成 wrench，会重复建模机械臂对底座的作用，本版本已消除这一问题。

当前 actor 仍采用显式 30 维未来 wrench 输入，而没有复刻论文的 teacher/student wrench RNN。论文的消融实验表明显式预测输入可以作为有效基线；RNN 蒸馏应作为独立的下一阶段，避免和物理配置修复同时改变。

## 从零开始训练

不要传 `--resume` 或旧 checkpoint。新实验目录为：

```text
logs/rsl_rl/sfyg_tron_2a_wholebody_paper_v2_flat/<timestamp>/
```

服务器命令：

```bash
cd ~/workspace/isaaclab_tron2/TRON2_WholeBody_LAB
conda activate limx

python scripts/rsl_rl/train.py \
  --task Isaac-Limx-SFYG-TRON2A-WholeBody-Flat-v0 \
  --num_envs 4096 \
  --headless \
  --seed 42
```

建议先用 256 个环境做 100～300 iteration 启动检查，确认 observation/action 维数和 reward 无异常，再运行 4096 环境的完整 20000 iteration。

## 验收顺序

1. TensorBoard 同时检查 episode length、fall/base-contact、速度跟踪和总 reward，而不是只看总 reward。
2. 分别固定测试站立、前进、后退、横移和转向；不接受“平均回报上升但某一方向必摔”。
3. 首先在训练环境验证合成 wrench；随后在完整机械臂 Play 环境由 OCS2 注入未来 wrench。
4. 比较 OCS2 当前预测 wrench 与仿真中机械臂实际反力的坐标系、符号、作用点和时间戳。
5. 只有以上通过后再进入 MuJoCo sim2sim，避免把 Isaac 内部配置错误误判为迁移误差。

旧 checkpoint 的 actor 输入维数仍兼容 Play，但其训练物理语义与 v2 不同，不应作为 v2 的续训起点。
