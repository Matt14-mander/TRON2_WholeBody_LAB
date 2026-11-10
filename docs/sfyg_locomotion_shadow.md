# 冻结 SFYG 真机只读推理

SFYG 随车部署主机为 Jetson NX（本项目现场确认）。已能通过 LimX SDK 订阅本体的扩展电脑也可用于台架采集。运行环境需要能在同一个 Python 中导入 NumPy、PyTorch 和 TRON2 SDK。模型使用训练服务器 robust run 的 `exported/model_5000/{policy.pt,deployment.json}`。

```bash
cd ~/workspace/TRON2_WholeBody_LAB
OUT="logs/hardware/sfyg_shadow_$(date +%Y%m%d_%H%M%S)"
python3 scripts/hardware/shadow_sfyg_locomotion.py \
  --bundle_dir ~/models/sfyg_robust5000 --robot_ip 10.192.1.2 \
  --command 0 0 0 --duration 10 --output_dir "$OUT"
cat "$OUT/summary.json"
```

脚本只订阅状态，全部动作写入日志。输入为 42 维观测、从旧到新的 10×42 历史和 3 维虚拟速度指令。首帧历史重复填充。上一动作初始为零，之后取上一帧模型预测，真机姿态由实际物理状态决定。可以用 steps.jsonl 的 observation、history、command 和 action 复算每个预测。

默认假设 SDK 十关节按左腿到右腿、每侧 pitch/roll/yaw/knee/ankle 顺序，与训练零位和方向一致；按 wxyz 解释 IMU，默认 IMU 轴与 base 轴对齐。identity_unverified 表示这些安装和标定假设尚待现场确认。核准后可用 `--coordinates my_robot.json` 指定四项：

- sdk_names 按 SDK 数组下标列出十个完整关节名，名称取自 deployment.json 的 leg_names。
- zero_rad 和 sign 均按策略关节顺序排列。重排后 q_policy = sign × (q_sdk − zero_rad)，dq_policy = sign × dq_sdk。训练默认姿态由观测函数另行扣除。
- imu_to_base 是把向量从 IMU 轴旋转到策略 base 轴的 3×3 正交旋转矩阵，用于陀螺仪及投影重力。

按导出配置的控制周期采样（当前 20 ms）。SDK 时间戳按流分别检查，主机单调时钟用于检查接收新鲜度和配对。接收时刻接近不代表传感器硬件同步。到期仍无新的有效数据对会结束，已写入的轨迹保留。调度迟到整周期但仍有新鲜数据时，重新填充历史并重置上一动作和步态相位，从新的 segment 开始。检查 summary.json 的 completed、reason、frames 和 timing_gaps。模型哈希、标定映射和时序阈值在 interface.json。这些数据用于验证观测与推理，闭环步行需另行测试。
