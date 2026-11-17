# PIPER X 直接使能和 carry 调试

入口 `scripts/hardware/piper_x_carry.py` 使用官方 pyAgxArm 的
`ArmModel.PIPER_X`、`PiperFW.DEFAULT`、`move_j`（关节位置速度模式，弧度）。
参考 https://github.com/agilexrobotics/pyAgxArm/blob/master/docs/piper/piper_api.md 。
不使用无平滑的 `move_js`，不 reset、不校零、不失能、不初始化或控制夹爪。
底层冻结 policy 不变。该入口是单臂台架调试，不发布腿部所需 HOLD 心跳；不要与腿部接管同时运行。

区别：旧 prepare 入口要求整个测量起点在合法范围内。这里查询控制器实际限位，
对距离边界≤7度的越界起点，显式展示向边界内部的恢复目标，其余轴保持。
7度只约束一次恢复资格，不修改 SDK/驱动器正常限位，不证明零位/方向正确。
超出该窗口拒绝；无错误状态不是任意运动安全证明。使能可能产生重力负载，须机械臂独立支撑。
不能将自然下垂单帧当成标定，不向外移动，不禁用 SDK 限位。

当前 SDK 角例：[-.17,-5.786,2.368,34.178,.655,0]度；入口恢复候选
[-.17,0,0,34.178,.655,0]度，后续 carry 候选[0,90,-85,0,0,0]度。
carry 必须由操作者通过显式 --carry_sdk_deg 提供并现场核对 SDK 坐标、安装方向及整条轨迹。
不能因它等于仿真默认角就认定实机映射已验证。没有自动避障，不设置安装方向/载荷参数以免覆盖现场配置。

Jetson：只保留本程序写机械臂，确认官方/VR/CAN 控制器已释放；吊架承重，腿程序停止，夹爪负载可靠固定。
pyAgxArm 使用现场已有版本，以下 PYTHONPATH 用于使用已有源码而不升级安装。

```bash
cd ~/workspace/TRON2_WholeBody_LAB
export PYTHONPATH="$HOME/arm/pyAgxArm${PYTHONPATH:+:$PYTHONPATH}"
python3 scripts/hardware/piper_x_carry.py --can can0 --carry_sdk_deg 0 90 -85 0 0 0
```

默认只查询/读取，不发送使能、模式、速度或关节控制。打印当前、恢复和 carry 的 SDK 角。
核对路径后运行同一命令加 --execute。

```bash
python3 -u scripts/hardware/piper_x_carry.py --can can0 --carry_sdk_deg 0 90 -85 0 0 0 --execute
```

输入 ENABLE：确认支撑、独占总线、恢复路径，按5%速度预置已打印的合法恢复目标，然后请求使能。
不把初始范围外角直接传给 SDK，避免 SDK 静默裁剪。当前角相对预览变化>0.02rad拒绝。
10秒使能/恢复超时；确认 CAN/J 模式并连续到位1秒后 READY。
输入 CARRY：以同一5%速度比发送明确指定的目标，120秒内到位后打印 HOLD。
5%是驱动器速度比例，不是已验证0.05rad/s限制。
程序持续监测角度和控制器状态新鲜度≤300ms、错误状态、模式及保持误差。
不调用旧入口的全套速度/驱动帧150ms门槛；这不是旧超时已经排除。
不独立验证六轴使能丢失或实测速度，本入口不用于整机行走验收。

正常结束：机械臂与负载可靠支撑后输入 STOP SUPPORTED。
Ctrl+C、终端EOF、反馈异常也尝试电子快速停止；不能保证断线/主机掉电时停止送达。
退出不自动失能，不保证断开后持续保持，支撑不能依赖已退出进程。

软件测试使用 SDK 替身，验证默认无动作、恢复窗口、合法恢复目标先于使能、独立 carry 确认、异常停止。
未连接实机，未验证载荷、碰撞或速度比的物理效果。
