# 吊架承重、双脚离地的独立准备动作

只读预览使用 `scripts/hardware/prepare_sfyg_legs.py`，仅校验正式 robust5000 bundle 并输出计划。
实机执行统一使用 `scripts/hardware/run_sfyg_lifecycle.py --suspended-prepare`。吊架插值、保持和策略接管共用一个腿部发布线程。

## 工况边界

用吊架把机器人吊起，让双腿自然下垂、停止摆动，双脚全程离地。控制器在收到准备命令时读取当帧实测腿角，沿五次曲线移到正式 bundle 的站立准备角，并在到位后保持。无需先把机器人摆成蹲姿。当前执行范围是 2026-11-19 实测起点附近（每关节±0.08 rad）。
吊架须承重并保证机身姿态；双脚在整段路径中有充分离地间隙。脚尖触地也不属于本测试工况。
脚接触无法从现有SDK自动确认，必须现场核实，不能把preflight布尔值当作传感器。
现有SDK没有明确逐电机使能位；`drive_enable_semantics_confirmed`要求核实厂家的低层控制语义，
不是假装读取到了使能。程序不发送另行的使能/复位命令。
准备可以在 0 kg 或 0.5 kg 负载下执行。实机策略接管只允许使用已核对的 0.5 kg 负载。

所有preflight默认false。现场确认关节顺序/方向、wxyz与IMU安装、独占控制权、急停及机器人端
通信超时行为后才可执行。实机入口持有同一用户下的本机发布锁，阻止重复启动统一入口。现场仍须确认官方控制器或其他进程已释放腿部控制权。

## 状态与退出

- 默认只读 10 秒，输出 `preview.json` 和 `steps.jsonl`。预览记录实测起点、最大关节速度、机身倾角和起点拒绝原因。通过起点检查时，另给出 MOVE 阶段的时长和五个插值采样点。预览通过仅代表该帧通过起点检查，执行时会从收到命令的最新反馈重新检查并生成轨迹。WAIT 不发送电机指令。
- `--execute`仍停在WAIT，输入`prepare suspended`后才开始。
- CAPTURE：目标从实测q开始，Kp在0.25秒平滑建立，保持起点至1秒。
- MOVE：至少16秒五次平滑插值；参考速度<=0.1rad/s、参考加速度<=0.1rad/s²。
  实际关节捕获速度不等于参考速度，实机必须监测反馈。
- SETTLE：位置误差<0.08rad、速度<0.1rad/s连续2秒后进入HOLD，之后持续保持。
- 吊架准备和 HOLD 阶段由位置控制保持。若准备中输入 `stop`，控制器冻结当前参考并等待 PAUSED。PAUSED 只允许受支撑释放。
  停止参考速度会变为0，故不是带加速度约束的重新规划；位置目标连续。
- HOLD/PAUSED下，确认支撑后`release supported`：2秒卸载Kp，1秒阻尼后退出；腿可能下垂。
- `abort`、Ctrl+C、EOF、远端L1+square/cross或心跳/通信故障触发尽力阻尼，不能保证姿态保持。
  保留吊架保护；退出时不自动恢复官方控制器。

运动中倾斜>=12°、反馈速度>1rad/s、超出窄路径走廊、跟踪误差>0.25rad或>0.12rad持续0.25秒，
均故障退出。SETTLE超过12秒未完成也退出。卸载阶段允许重力下垂，不再要求跟踪固定目标或原路径走廊；
倾斜/速度、反馈新鲜度、遥控器及机械臂HOLD检查仍保留。沿用原MotorLease的200Hz发布、60ms租约、
40ms反馈新鲜度及15ms关节/IMU时间差检查。它不替代机器人端看门狗。

## 官方参考及IMU纠正

官方SolefootController从准备按钮按下时的实测q插值到配置目标，单独按钮才进入policy。
此处借鉴流程，不照搬官方1秒准备动作，也不使用其policy。
- https://github.com/limxdynamics/tron2_rl_deploy_python/blob/master/controllers/SolefootController.py
- https://github.com/limxdynamics/tron2_rl_deploy_python/blob/master/controllers/model/SF_TRON2A/params.yaml

官方imu_data_callback将原始wxyz重排成xyzw才传给SciPy。此前文档漏看回调的xyzw候选已纠正。
原SDK字段顺序wxyz也与现场加速度一致性试验吻合；这不等于腿部方向已完成实机验收。

## Jetson

```bash
cd ~/workspace/TRON2_WholeBody_LAB
git pull --ff-only origin sim2real
cp -n scripts/hardware/sfyg_supported_prepare.example.json ~/sfyg_shadow/config/legs_prepare.json
python3 -u scripts/hardware/prepare_sfyg_legs.py \
  --bundle_dir "$HOME/models/sfyg_robust5000" \
  --profile "$HOME/sfyg_shadow/config/legs_prepare.json" \
  --output_dir "$HOME/sfyg_shadow/logs/legs_preview_$(date +%Y%m%d_%H%M%S)"
```

机械臂monitor持续运行。核对预览和现场条件后编辑配置operator、实际payload_mass_kg和所有检查。
不要一键把所有检查改true；尚未确认项会阻止执行，这是预期行为。

```bash
nano ~/sfyg_shadow/config/legs_prepare.json
python3 -u scripts/hardware/run_sfyg_lifecycle.py \
  --bundle_dir "$HOME/models/sfyg_robust5000" \
  --preflight "$HOME/sfyg_shadow/config/legs_prepare.json" \
  --arm_status_file "$HOME/sfyg_shadow/run/arm_monitor.json" \
  --output_dir "$HOME/sfyg_shadow/logs/legs_prepare_$(date +%Y%m%d_%H%M%S)" \
  --suspended-prepare --execute
```

输入 `prepare suspended`，等待 `HOLD preparation_settled=True`。接下来可以在同一终端选择：

- 只验收准备时，吊架继续承重，在 HOLD 下输入 `release supported`。
- 接入零速策略时，先现场确认双脚已承重、吊架保留防跌、机械臂 HOLD 有效且安装了 0.5 kg 负载，再输入 `start grounded`。SDK 没有接地判断，命令表示操作者已确认。控制器延续最后一帧保持指令，随后用约 300 ms 混入零速策略。正常退出依次为 `stop`、等待 ZERO、吊架重新承重、`release supported`。

## MuJoCo与测试

```bash
python scripts/mujoco/test_sfyg_supported_prepare.py \
  --bundle_dir logs/hardware_bundles/sfyg_robust5000 \
  --output_dir logs/mujoco/supported_prepare_check
```

使用原资产生成模型，在1.6m高度以mocap焊接约束支撑机身，保留自由基座和原碰撞。
不以删除自由关节模拟吊架：那会改变父子碰撞过滤，导致假髋关节碰撞。
8组覆盖负载0/0.5kg、延迟0/4物理步、完整动作/7秒中途stop，全部到保持并受支撑释放，policy调用0。
实测起点完整动作19秒进入HOLD，保持3秒后测试释放；机器人脚与环境无接触。
这不是弹性吊绳模型、不是双脚承重，也不是实机验收。
夹具在准备时抬高机身，在 HOLD 后下降，模拟双脚接地后才触发零速接管。测试核对 HOLD→BLEND 的位置目标和增益连续，并检查准备阶段双脚离地。

```bash
python scripts/mujoco/test_sfyg_lifecycle.py \
  --bundle_dir logs/hardware_bundles/sfyg_robust5000 \
  --initial_pose suspended --delay_steps 4 --vx 0 \
  --output_dir logs/mujoco/suspended_lifecycle_zero
```
