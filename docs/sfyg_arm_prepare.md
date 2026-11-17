# PIPER X 分段搬运准备

`prepare_sfyg_arm.py` 复用独立机械臂保持入口的 CAN 反馈、总线互斥锁、使能前目标预置、
故障快速停止及腿部 HOLD 心跳。底层 policy、训练及腿部控制源码不变。
默认只读，收到完整反馈后生成一次计划并继续被动读取。不会自动校零、复位、打开夹爪、
升级固件、修改总线或结束官方进程。软件测试不能代替实机轨迹、保持力矩及碰撞验证。

## 映射与限位审计（2026-11-17）

核对来源：[AgileX 官方 PIPER X 预设](https://github.com/agilexrobotics/pyAgxArm/blob/master/pyAgxArm/api/constants.py)，
以及本仓库 `robot_description/tron2/SFYG_TRON2A/urdf/robot.urdf`。
注意官方当前 pyAgxArm 和本脚本使用的 Piper V2 SDK 是不同 API；不因读取参数就迁移硬件 SDK。

| 顺序 | URDF 名称 | 官方 PIPER X 预设角范围(度) | SFYG URDF 范围(度) |
|---|---|---|---|
| 1 | arm1_Joint | -150..150 | -150..150 |
| 2 | arm2_Joint | 0..180 | 0..180 |
| 3 | arm3_Joint | -170..0 | -170..0 |
| 4 | arm4_Joint | -89..89 | -89..89 |
| 5 | arm5_Joint | -89..89 | -89..89 |
| 6 | arm6_Joint | -170..170 | -120..120 |

新模板按两者保守交集提供候选限位；旧模板 wrist4/5 是混用的普通 Piper 范围，不再用于这条入口。
新模板不覆盖旧配置。映射定义仍为 `q_sdk[j]=sign[j]*q_sim[sdk_to_sim[j]]+zero_rad[j]`。
同名和相近范围不能证明实机零位与正方向一致；需要现场型号/固件资料、夹具标定或已验证
官方控制记录。不要依据自然下垂单帧设置零位，更不能使用 MuJoCo estimator 的 qzero 校正作为电机零点。

现场记录 `[-.025,-5.652,2.562,33.993,.349,0]` 度在第2、3轴超出官方候选范围。
该冲突尚未解决。新脚本可预览这些数据，但执行仍拒绝。核对协议和实际零位后记录证据，
如需调整物理限位，须有硬件资料支持，不能为通过检查单纯放宽。

## 计划和执行

计划在 SDK 关节空间生成，先经过配置 `sdk_waypoints_rad`（六轴弧度列表），最后到模型映射的 carry。
无中间点时仅生成直线插值候选，不是折叠姿态避障规划。每段最大单轴变化0.2 rad，
五次曲线规划速度≤0.05 rad/s、每段≥4秒。直线段位于关节限位内不代表不会撞机身/腿/吊架。
执行前必须审核整个扫掠空间。可先编辑显式中间点，再生成新的预览；没有通用自动安全展开次序。

Jetson 先保留吊架和机械臂独立支撑；腿部发送程序不运行。
确认是 Low-Level Dev，官方机械臂 CAN 发送器已停止。必须使用 Piper V2 兼容固件。

```bash
cd ~/workspace/TRON2_WholeBody_LAB
mkdir -p ~/sfyg_shadow/config ~/sfyg_shadow/run ~/sfyg_shadow/logs
cp -n scripts/hardware/sfyg_piper_x_prepare.example.json ~/sfyg_shadow/config/piper_x_prepare.json
nano ~/sfyg_shadow/config/piper_x_prepare.json
export ARM_PLAN="$HOME/sfyg_shadow/logs/arm_plan_$(date +%Y%m%d_%H%M%S).json"
python3 -u scripts/hardware/prepare_sfyg_arm.py --bundle_dir "$HOME/models/sfyg_robust5000" --profile "$HOME/sfyg_shadow/config/piper_x_prepare.json" --can can0 --plan_file "$ARM_PLAN" --status_file "$HOME/sfyg_shadow/run/arm_hold.json"
```

预览显示 SDK 实测角、由候选映射反算的仿真角、每段目标及全部越界点，不在越界起点退出。
反馈仍必须完整新鲜且无驱动器故障。已有计划文件禁止覆盖。输入 `quit supported` 退出只读。
打印和保存的计划 SHA256 绑定实际审核文件，不能用另一个计划代替。

审核映射/限位、型号固件、独占控制、支撑、路径、负载夹持并填 operator 和原七个检查项。
新增 `mapping_evidence`、`limits_evidence` 填真实证据，`staged_path_reviewed=true` 仅在路径实际审核后填写。
预览有任何越界时先解决映射/标定，不执行。映射/限位/中间点变化后重新生成审核计划。

```bash
export ARM_PLAN_SHA=$(sha256sum "$ARM_PLAN" | cut -d ' ' -f 1)
python3 -u scripts/hardware/prepare_sfyg_arm.py --bundle_dir "$HOME/models/sfyg_robust5000" --profile "$HOME/sfyg_shadow/config/piper_x_prepare.json" --can can0 --plan_file "$ARM_PLAN" --plan_sha256 "$ARM_PLAN_SHA" --status_file "$HOME/sfyg_shadow/run/arm_hold.json" --execute
```

执行启动仍不发电机命令；输入 `next 1 supported` 才捕获当前位置、预置目标、请求 CAN MOVE J 并使能。
先保持2秒等待确认，再走第1段。与预览起点差>0.02 rad拒绝执行。
到位且稳定1秒进入 PAUSED，持续发位置保持。现场检查后输入 `next 2 supported`，依次类推。
MOVING 期间的下一段命令直接拒绝，不会排队等待到位后自动执行；不要批量粘贴确认。
每段跟踪误差、速度、反馈新鲜度、使能/模式和超时均检查。
最后显示 `[ARM] HOLD ready=True` 后保持该进程运行，直接使用原腿部程序的
`--arm_status_file "$HOME/sfyg_shadow/run/arm_hold.json"`。不要退出后再启动另一个 arm hold。
PAUSED、WAIT 不满足腿部 HOLD 条件，禁止腿部接管。腿先准备，接地后才 start grounded。

结束先退出腿程序，再给机械臂与负载支撑，输入 `quit supported`；执行模式退出尝试快速停止，
不自动失能、不夹爪操作、不回零。异常 abort/Ctrl+C 同样尝试停止，不保证断线情况下指令送达。

## 验证边界

本地测试涵盖真实越界输入只读预览、映射绑定和SHA、路径篡改、逐段确认、速度连续性、
未到位禁止继续、起点改变拒绝、模式丢失、过速/跟踪/到位超时和原腿部心跳回归。
没有连接实机 CAN，不宣称已验证实际展开路径或真实负载能力；正式 policy 权重未修改。
