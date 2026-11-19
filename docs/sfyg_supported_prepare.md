# 吊架承重、双脚离地的独立准备动作

入口：`scripts/hardware/prepare_sfyg_legs.py`。不加载或调用actor，只读取并校验正式robust5000
导出中的目标、PD增益、力矩上限和文件哈希。policy.pt、训练源码、原Lifecycle的0.6rad检查不变。

## 工况边界

仅覆盖2026-11-19实测起点附近（每关节±0.08rad），不做地面蹲起、跌倒恢复或脚顶地伸腿。
吊架须承重并保证机身姿态；双脚在整段路径中有充分离地间隙。脚尖触地也不属于本测试工况。
脚接触无法从现有SDK自动确认，必须现场核实，不能把preflight布尔值当作传感器。
现有SDK没有明确逐电机使能位；`drive_enable_semantics_confirmed`要求核实厂家的低层控制语义，
不是假装读取到了使能。程序不发送另行的使能/复位命令。
本脚本可记录0kg或0.5kg负载工况，但这不取消后续policy的0.5kg负载要求。

所有preflight默认false。现场确认关节顺序/方向、wxyz与IMU安装、独占控制权、急停及机器人端
通信超时行为后才可执行。不得与run_sfyg_lifecycle、官方部署控制器等同时发布腿部命令。
进程锁只排除本脚本的第二个实例，不能证明其他控制器不存在。

## 状态与退出

- 默认只读10秒，输出preview.json、steps.jsonl；WAIT不发送任何电机指令。
- `--execute`仍停在WAIT，输入`prepare suspended`后才开始。
- CAPTURE：目标从实测q开始，Kp在0.25秒平滑建立，保持起点至1秒。
- MOVE：至少16秒五次平滑插值；参考速度<=0.1rad/s、参考加速度<=0.1rad/s²。
  实际关节捕获速度不等于参考速度，实机必须监测反馈。
- SETTLE：位置误差<0.08rad、速度<0.1rad/s连续2秒后进入HOLD，之后持续保持。
- 无policy/start命令、无自动接管。`stop`冻结当前参考并等待PAUSED，不回零，不自动恢复。
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
  --arm_status_file "$HOME/sfyg_shadow/run/arm_monitor.json" \
  --output_dir "$HOME/sfyg_shadow/logs/legs_preview_$(date +%Y%m%d_%H%M%S)"
```

机械臂monitor持续运行。核对预览和现场条件后编辑配置operator、实际payload_mass_kg和所有检查。
不要一键把所有检查改true；尚未确认项会阻止执行，这是预期行为。

```bash
nano ~/sfyg_shadow/config/legs_prepare.json
python3 -u scripts/hardware/prepare_sfyg_legs.py \
  --bundle_dir "$HOME/models/sfyg_robust5000" \
  --profile "$HOME/sfyg_shadow/config/legs_prepare.json" \
  --arm_status_file "$HOME/sfyg_shadow/run/arm_monitor.json" \
  --output_dir "$HOME/sfyg_shadow/logs/legs_prepare_$(date +%Y%m%d_%H%M%S)" \
  --execute
```

输入`prepare suspended`。HOLD后保持本终端，不启动另一个腿部控制器。
本次没有实现跨进程无缝交接；退出后不能假定腿仍然在准备角。后续policy交接需另行验证。

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
单元测试覆盖拒绝超范围起点、持续保持不进入policy、旧0.6rad窗口仍拒绝该起点、故障阻尼和预览不创建发送器。
