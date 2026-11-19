# Carry保持心跳与坐标核对（2026-11-19）

本次不改变权重、训练配置、观测维度、PD增益或MuJoCo策略。

## 保持方式

`monitor_sfyg_arm.py`仅接收SocketCAN：监测已经到位、使能并由机械臂内部J模式控制器保持的carry。
不实例化Piper SDK、不发送目标、不切模式、不复位、不失能，也不会在退出时急停机械臂。
因此不会与成功的V189运动入口争夺控制权。必须先用已验证的入口到位并确认使能。
这不是新的主动机械臂伺服器；如果硬件控制器不能自行维持位置，它会报错而不是接管补偿。

每20ms刷新原子JSON。所有角度、速度、驱动和状态帧都必须在150ms内；六关节使能、CAN/J、
无故障、目标误差<0.04rad且速度<0.05rad/s连续2秒，才进入HOLD。
之后任一失效锁存FAULT并退出；进程被强杀则腿部ArmGuard在250ms心跳过期后拒绝继续。
腿部原有看门狗在故障时尝试阻尼，不保证继续站立，必须保留吊架保护。
官方arm-mros必须inactive且桥接进程不存在；每秒复查，检查阻塞也可能使心跳过期。
不自动重启服务。没有伪造execute=true：新增独立`sfyg_arm_monitor_v1`协议，明确motor_output=false。
退出顺序：腿部stop→ZERO→支撑→release supported→腿部退出，再Ctrl+C结束监测器。
监测器退出不会让机械臂失能；机械臂退出按已验证的控制流程另行处理。

## 源码核对与边界

核对官方文件（2026-11-19读取master，链接可能随后更新）：
- https://github.com/limxdynamics/tron2_rl_deploy_python/blob/master/controllers/SolefootController.py
- https://github.com/limxdynamics/tron2_rl_deploy_python/blob/master/controllers/model/SF_TRON2A/params.yaml

官方关节顺序：左髋pitch、roll、yaw、膝、踝pitch，再右侧同序。
与`sfyg_policy.LEG_NAMES`一致（本项目名称多`_Joint`后缀）。官方直接使用SDK q/dq、没有符号翻转或零点偏移。
准备角[0,0,-pi,0,0,0,0,+pi,0,0]，动作转换default+0.25*action，与本策略导出一致。
本项目zero/sign以**policy顺序**存储；sdk_names是**SDK数组顺序**。

官方通过SciPy `Rotation.from_quat`默认xyzw解析SDK姿态，然后将世界重力逆旋转到机身；gyro直接使用，
SF配置安装旋转为零。原入口固定按wxyz，存在适配差异。本次增加quaternion_order，模板显式xyzw。
不带配置的旧只读行为仍保持wxyz以复现旧日志；旧日志sdk_quat_wxyz实际上是原始数组，不可相信字段名。
新日志同时记录sdk_quat_raw及quaternion_order。真实电机执行要求显式四元数顺序，旧配置不再默认通过。

这是**源码一致性核对，不是实机映射验收**。不同SDK版本、安装和实机标定仍须核对。
静止直立时两种顺序可能都给出近似向下重力，不能自动判断。检查已知姿态/安全的小幅机身倾斜日志：
按右手系，正roll投影Y为负，正pitch投影X为正；gyro分别对应X/Y/Z，需核对实际机身轴。
只使用官方允许的支撑/姿态调整流程，不用本工具下发测试动作，也不要求手掰使能关节。
十关节顺序/方向需结合厂家轴定义和已知单关节变化日志验证；不能靠单张静态姿态自动签字。
绕Z的安装偏差不能通过静态重力验证。准备窗口差值亦不代表映射正确。
preflight中的joint_mapping_checked及负载/低层模式/支撑检查仍由现场实际核对，不自动填true。

## Jetson流程

```bash
cd ~/workspace/TRON2_WholeBody_LAB
git pull --ff-only origin sim2real
mkdir -p ~/sfyg_shadow/config ~/sfyg_shadow/run ~/sfyg_shadow/logs
cp -n scripts/hardware/sfyg_arm_monitor.example.json ~/sfyg_shadow/config/arm_monitor.json
nano ~/sfyg_shadow/config/arm_monitor.json
```

填写operator；核实carry SDK角与仿真关节映射后将mapping_checked设true。
模板映射为单位映射，不是硬件自动标定。不要改限位以容纳错误读数。
官方arm-mros保持停止；不要在已有保持控制时重复执行旧hold_sfyg_arm.py。

终端A保持运行：
```bash
python3 -u scripts/hardware/monitor_sfyg_arm.py --bundle_dir "$HOME/models/sfyg_robust5000" --profile "$HOME/sfyg_shadow/config/arm_monitor.json" --can can0 --status_file "$HOME/sfyg_shadow/run/arm_monitor.json"
```

等到HOLD，终端B读取心跳验证（不输出电机）：
```bash
cd ~/workspace/TRON2_WholeBody_LAB
PYTHONPATH=scripts/hardware python3 -c 'from pathlib import Path; from sfyg_arm_guard import ArmGuard; from hold_sfyg_arm import POLICY_SHA; ArmGuard(Path.home()/"sfyg_shadow/run/arm_monitor.json", POLICY_SHA).check(); print("HOLD heartbeat accepted")'
```

另存映射候选文件，避免覆盖已有审核结果：
```bash
cp -n scripts/hardware/sfyg_robust5000_preflight.example.json ~/sfyg_shadow/config/robust5000_xyzw_candidate.json
OUT="$HOME/sfyg_shadow/logs/xyzw_shadow_$(date +%Y%m%d_%H%M%S)"
python3 -u scripts/hardware/shadow_sfyg_locomotion.py --bundle_dir "$HOME/models/sfyg_robust5000" --robot_ip 10.192.1.2 --coordinates "$HOME/sfyg_shadow/config/robust5000_xyzw_candidate.json" --command 0 0 0 --duration 10 --output_dir "$OUT"
python3 scripts/hardware/audit_sfyg_coordinates.py --steps "$OUT/steps.jsonl" --coordinates "$HOME/sfyg_shadow/config/robust5000_xyzw_candidate.json" --bundle_dir "$HOME/models/sfyg_robust5000" > "$OUT/coordinate_audit.json"
cat "$OUT/coordinate_audit.json"
```

审核工具永远返回hardware_mapping_verified=false，不会用准备窗口/重力方向冒充实机验收。
修正四元数顺序后需重新检查动作，不能直接沿用旧wxyz日志结论。

实际安装0.5kg负载、Low-Level Dev、映射和支撑全部核实后，使用人工审核过的preflight：
```bash
python3 -u scripts/hardware/run_sfyg_lifecycle.py --bundle_dir "$HOME/models/sfyg_robust5000" --preflight "$HOME/sfyg_shadow/config/robust5000.json" --arm_status_file "$HOME/sfyg_shadow/run/arm_monitor.json" --robot_ip 10.192.1.2 --execute --output_dir "$HOME/sfyg_shadow/logs/lifecycle_$(date +%Y%m%d_%H%M%S)"
```
逐条输入prepare supported→READY→双脚承重后start grounded；零速观察，stop→ZERO→支撑→release supported。
没有0.5kg负载或仅脚尖触地时，不运行该执行命令。

## 验证范围

单元测试覆盖持续到位、使能/模式/位置/速度/NaN失败、心跳超时与重启、四元数roll/pitch、
逆关节映射；复跑原机械臂、腿部状态机和只读推理回归。不具备本地实机，未宣称完成真机承重验收。
