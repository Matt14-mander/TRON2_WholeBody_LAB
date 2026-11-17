# robust_5000 的独立机械臂 carry 保持

新增 `scripts/hardware/hold_sfyg_arm.py`，用于安装 Piper/PIPER X 且协议已核对的
SFYG。原 `shadow_sfyg_student.py` 只读 CAN，不保持机械臂；训练/仿真中的 800/40
锁位 PD 也不会自动转移到实机。新入口使用 Piper V2 SDK 的 CAN MOVE J 位置控制，
不将仿真 PD 增益复制给机械臂。该控制器尚未完成实机验证。

## 适配范围与必须确认的条件

- 固定使用正式 robust_5000 `deployment.json` 的 arm1..6 默认姿态，并验证文件哈希。
  carry 仿真角约为 `[0, 1.5707963, -1.4835299, 0, 0, 0]`。
- 映射定义：`q_sdk[j] = sign[j] * q_sim[sdk_to_sim[j]] + zero_rad[j]`。
  配置的限位也是 SDK 弧度。模板映射与限位是待核对候选，不能直接当校准结果。
- 使用 `piper_sdk.C_PiperInterface_V2`；要求支持 `ConnectPort(piper_init=False)`、
  `JointCtrl`、`MotionCtrl_2`、`EnableArm`、`EmergencyStop`、`DisconnectPort`。
  固件必须兼容对应的反馈帧；不兼容应停止，不要盲目升级固件或重置零点。
- Linux 上已有正确激活的 SocketCAN（通常 can0、1 Mbit/s）。脚本不改 CAN 配置。
  读反馈需要 `python-can`、`numpy`；只有执行时才导入 `piper_sdk`。
- 必须确认官方机械臂发布器/VR/其他 ROS 或 CAN 控制器已停止写入。切到腿部
  Low-Level Dev **不等于已证明机械臂 CAN 控制权释放**。本程序不自动杀官方服务；
  配置中的 exclusive_arm_control 是人工核对项。文件锁只防止本脚本同总线重复运行。
- 夹爪和 0.5 kg 负载须另外可靠固定。本脚本不发夹爪命令、不控制夹持力；
  不能保证官方夹爪控制停掉之后仍夹紧。未确认固定前不接管腿策略。
- 吊架承住底盘；机械臂也必须有可用的独立支撑。机械臂初始角距 carry 每关节
  不超过 0.6 rad，速度不超过 0.1 rad/s。没有任意姿态避障规划；大幅折叠起点
  需在官方控制/已验证流程下先移到 carry 附近，不要提高阈值绕过检查。

API/协议依据（2026-11-17 查阅）：
[Piper V2 接口](https://github.com/agilexrobotics/piper_sdk/blob/master/piper_sdk/interface/piper_interface_v2.py)、
[驱动反馈](https://github.com/agilexrobotics/piper_sdk/blob/master/piper_sdk/piper_msgs/msg_v2/feedback/arm_feedback_low_spd.py)、
[控制器反馈](https://github.com/agilexrobotics/piper_sdk/blob/master/piper_sdk/piper_msgs/msg_v2/feedback/arm_feedback_status.py)。

## 操作顺序

以下为 Jetson 上两个交互终端，必须在同一台主机运行（心跳使用本机 monotonic 时钟）。
部署目录沿用现有 `~/models/sfyg_robust5000`，仓库路径按实际安装位置调整。

1. 官方站立，机械臂先到 carry 附近；吊架可靠承重并安排机械臂支撑。
2. 官方遥控 `L1+□` 进入 Idle，再切 Low-Level Dev。确认机械臂状态和控制权；
   不要假设切换过程仍有原先保持力矩。
3. 复制并填写机械臂配置。每项实际核对后才置 true，填写 operator 和实机映射。

```bash
cd ~/workspace/TRON2_WholeBody_LAB
mkdir -p ~/sfyg_shadow/config ~/sfyg_shadow/run
cp -n scripts/hardware/sfyg_arm_hold.example.json ~/sfyg_shadow/config/arm_hold.json
nano ~/sfyg_shadow/config/arm_hold.json
```

4. 先只读核对映射后的目标和反馈，不带 `--execute` 不会实例化电机发送器。

```bash
python3 -u scripts/hardware/hold_sfyg_arm.py \
  --bundle_dir "$HOME/models/sfyg_robust5000" \
  --profile "$HOME/sfyg_shadow/config/arm_hold.json" \
  --can can0 --status_file "$HOME/sfyg_shadow/run/arm_hold.json"
```

终端显示 SDK 弧度目标和实测角；状态文件持续更新 q_sdk_rad。
只读检查完输入 `quit supported`，再运行同一命令加 `--execute`。
进入 WAIT 后逐条输入（不要批量粘贴）：

```text
prepare supported
```

这时才捕获当前关节角、预置该目标、请求 CAN MOVE J 并使能六个臂关节。
前两秒保持捕获目标、等待使能和模式反馈，随后至少八秒五次曲线过渡到 carry；
规划最大速度 0.08 rad/s，SDK 速度比例 10%。ready 要求角误差 <0.04 rad、
速度 <0.05 rad/s 连续一秒且驱动已使能、模式已确认。
等待 `[ARM] HOLD ready=True`；保持这个终端运行。

5. 第二终端按原腿部预检填写 robust5000.json，然后启动腿状态机，新增必填参数：

```bash
cd ~/workspace/TRON2_WholeBody_LAB
python3 -u scripts/hardware/run_sfyg_lifecycle.py \
  --bundle_dir "$HOME/models/sfyg_robust5000" \
  --preflight "$HOME/sfyg_shadow/config/robust5000.json" \
  --arm_status_file "$HOME/sfyg_shadow/run/arm_hold.json" \
  --robot_ip 10.192.1.2 --execute \
  --output_dir "$HOME/sfyg_shadow/logs/lifecycle_$(date +%Y%m%d_%H%M%S)"
```

腿终端仍按 `prepare supported` → 等 READY/settled → 确认双脚接地 →
`start grounded`，首次仅做零速接管。机械臂位置 HOLD 不等于整机平衡通过。

6. 退出顺序：**先腿，后臂**。腿终端 `stop` → ZERO 且现场确认停止 → 吊架承重 →
   `release supported` → 等 DONE/进程退出。然后把机械臂和负载可靠支撑好，
   在机械臂终端输入 `quit supported`。退出会发送机械臂快速停止，之后断开；
   不自动回零、复位、失能或松夹爪。退出后的支撑不能依赖本进程。

## 故障处理与限制

机械臂逐一检查三组角度、六个速度、六个驱动状态和控制器状态帧，新鲜度上限
150 ms。缺帧、故障位、失能/模式丢失、过速、跟踪误差或 HOLD 偏离会停止执行，
发布非 ready 状态并尝试快速停止；不会自动恢复。Ctrl+C/EOF/abort 也是停止。
默认只读不会发送快速停止或使能。正常退出同样要在腿退出和机械臂有支撑之后。

腿启动和 prepare/start 都检查实时 HOLD；200 Hz 电机发布线程也检查心跳。
超过 250 ms、只读状态、非 HOLD、控制器重启或文件读取失败，会锁存腿部故障并
走现有阻尼退出。因此机械臂故障可能撤掉腿部主动支撑，首轮必须全程保留吊架。
机械臂进程独立于腿进程；腿停止/异常不会自动让机械臂松力，需按上述顺序结束。

心跳是同主机进程间状态，不是硬件安全认证，也不监测负载脱落或夹爪力。
SDK 部分发送错误仅记录日志；实际反馈用于发现未到位，不代表每个 CAN 包有确认。
进程挂死、主机掉电、总线断开时不能保证快速停止到达，需验证机端超时和现场急停。
不应将模板检查项或单元测试通过当作实机可直接无吊架行走的证明。

## 本地验证

2026-11-17：在已有 isaaclab Python 环境运行
`python -m pytest tests/test_sfyg_arm_hold.py tests/test_sfyg_lifecycle.py tests/test_sfyg_locomotion_shadow.py -q`，
27 项通过。覆盖只读无发送器、映射与限位、准备轨迹连续性、折叠起点拒绝、
完整反馈/过期/故障、HOLD 丢失、会话重启拒绝、使能前目标预置、腿发布器故障阻尼及原流程回归。
SDK 调用使用替身验证；未连接 SocketCAN 实机，未验证实际保持力矩、碰撞、负载和固件交接。
