# robust_5000 部署状态机（2026-11-13）

这是部署外围控制器，不重新训练、不修改冻结策略。正式权重来自
`sfyg_robust_5000_gpu1_20261027_131013/model_5000.pt`，与 student 的
`model_500.pt` 无关。默认不下发电机；`--execute` 也只在明确输入准备命令后开始下发。

## 官方参考与适配范围

参考 [LimX SolefootController.py](https://github.com/limxdynamics/tron2_rl_deploy_python/blob/master/controllers/SolefootController.py)
（2026-11-13 检查 master）：`IDLE → STAND → WALK`、准备时捕获当前关节角并插值、
重置历史、位置 PD、轻重关节分组、力矩可行目标投影及阻尼退出。
当前 master 的遥控器处理为 L1+○ Prepare、L1+△ Start、L1+× Stop；
本实现**不接管这三个开始操作的遥控映射**，改用明确的终端命令。
L1+□ / L1+× 在本进程均作为中止信号，不作为普通减速停止。
机器人固件的模式切换和急停仍由官方遥控器负责，本程序不切换 Drive/Idle/Low-Level Dev。

与官方示例不同：保留本模型的 TorchScript encoder+actor、42 维观测、10×42 历史、
3 维速度指令、50 Hz 推理、零指令时继续推进步态时钟、0.25 rad 动作比例和导出增益。
不复制官方 ONNX 模型、其归一化或零速冻结步态逻辑。

`last_action` 在正常策略运行中仍为上一帧原始 actor action，保持训练语义；
只有接管过渡期使用混合后的请求动作。力矩投影只作用于电机目标，不回写为 actor 历史。
权重、训练配置、`sfyg_policy.py`、原 MuJoCo bridge 和原只读脚本均未修改。

固定文件哈希：

* policy.pt：`7e349aa6b3fe2cc62a77e874a26bb4c5742bd774d669d2120bab162d9d17809b`
* deployment.json：`3695b77527f248933aa7578154326bb394e1f601b937e2156660a81a93bd5c7e`
* 原 checkpoint：`09874b232d6f0b5834724129b24b4a307b84d9395aa408b6440b702ac9266114`

## 状态与边界

| 状态 | 行为 |
|---|---|
| WAIT | 仅收状态，不发布电机指令；等待 `prepare supported` |
| PREPARE | 捕获实测角；250 ms 平滑增加 Kp，保持目标到第 1 秒，再用五次曲线到准备姿态；轨迹至少 6 秒，最大规划速度 0.35 rad/s |
| READY | 位置 PD 保持；每关节误差 <0.12 rad、速度 <0.25 rad/s、倾角 <12° 连续 1 秒才允许开始 |
| BLEND | `start grounded` 后重置历史，零速度，300 ms 将策略动作从零平滑混入 |
| POLICY | 策略运行；初始命令零，允许 `vx 0..0.20`，命令变化率 0.1 m/s²；本入口暂不开放侧移/转向 |
| STOPPING | `stop`：速度指令降到零，继续用策略维持支撑，至少 4 秒后进入 ZERO |
| ZERO | 保持零速策略，**不意味着已测得底盘完全静止**；现场确认停止、吊架重新承重 |
| RELEASE | `release supported`：保持最后目标，2 秒平滑卸去 Kp，保留阻尼 |
| DAMP / DONE | 阻尼保持 1 秒，随后清理时再尝试发送 1 秒阻尼，再退出 |
| FAULT | 锁存故障，尽力发送 Kp=0 的阻尼；不会自动恢复或重新接管 |

准备目标来自导出文件，十关节为：
`[0, 0, -3.14159, 0, 0, 0, 0, 3.14159, 0, 0]`。
这不是把 SDK 所有角度设成零；坐标映射会进行逆转换。

**当前只验证了近站立姿态的受支撑准备。** Prepare 要求机身竖直（倾角≤20°）、
关节静止（≤0.5 rad/s），且每关节距离准备目标≤0.6 rad。
深折叠状态拒绝启动，不允许用此程序从地面深蹲自主起立。
之前实机遥控站立数据（膝约 -0.30/-0.27 rad）在关节距离窗口内，但仍须重新检查当前状态。
可先用官方 Drive 站立并安排吊架可靠支撑，再按官方流程进入 Idle/Low-Level Dev；
不能假定模式切换后原站立力矩仍存在，也不能仅凭吊带有连接就认为其已承重。

本策略训练的实体是完整 SFYG、固定搬运姿态机械臂与 0.5 kg 负载。
**裸底盘仍只做只读推理。** 本脚本只控制十个腿关节，不会移动/锁住 PIPER X；
机械臂须由独立控制器先到搬运姿态并维持，核对质量、安装和关节反馈。
仿真机械臂默认角为 `[0, 1.5707963, -1.4835299, 0, 0, 0]`（仿真关节坐标，
不能不经映射就复制成 PIPER SDK 命令）。preflight 文件是人工核对记录，并不是传感器自动验证。

## 保护与日志

* 200 Hz 独立发布线程；主推理每 20 ms 更新一次目标。主线程不更新时 60 ms 租约到期，
  即使推理/日志卡住也尝试转阻尼。状态接收最大年龄 40 ms、接收时差 15 ms；
  时间戳倒退/重复、不合法四元数、无效推理、主循环超时均停止继续执行策略。
* 需要 SDK 遥控器心跳，2 秒缺失中止；L1+□ / L1+×、终端 `abort`、Ctrl+C、SIGTERM、
  终端 EOF 均按故障中止。不要用 Ctrl+C 代替普通 `stop`。
* 准备/退出阶段倾角>20°、关节速度>3 rad/s 或持续 0.25 s 的>0.5 rad 跟踪误差触发保护。
  策略阶段倾角>35°或关节速度>16 rad/s 中止。
* 准备/退出阶段通过缩放增益限制估算 PD 力矩；固定增益策略阶段采用官方形式的目标投影。
  这不是力矩实测闭环，也不替代驱动器限幅。
* 主机看门狗不是认证的硬件安全回路。SIGKILL、主机掉电、网络中断、SDK 调用阻塞时，
  不能保证阻尼包送达；必须保留官方急停并核实机端失联保护。

`interface.json` 记录哈希/映射/检查项；`steps.jsonl` 记录 q/dq、重力、角速度、
观测/历史、原动作、过渡请求动作、PD 目标和增益；`summary.json` 记录退出原因、状态和发布数。
全部标注 `hardware_acceptance=false`，没有用仿真通过代替实机验收。

## 本地验证结果

MuJoCo 3.12.0，PyTorch 2.7.0，完整 SFYG+0.5 kg；近站立起点；单一确定性初始条件。
准备和退出阶段使用固定基座的**理想测试夹具**；BLEND/POLICY/STOPPING/ZERO
期间完全释放基座，没有托力。未建模真实吊带弹性、SDK 网络延迟和机器人驱动器。

| 指令/延迟 | 实际稳态 vx (m/s) | 停止尾段 vx RMS (m/s) | 最大自由基座倾角 | 结果 |
|---|---:|---:|---:|---|
| 0.20 / 0 ms | 0.16509 | 0.00783 | 4.21° | 完整流程通过 |
| 0.20 / 20 ms | 0.17844 | 0.00938 | 5.24° | 完整流程通过 |
| 0 / 0 ms | 0.00010 | 0.00787 | 1.40° | 完整流程通过 |
| 0 / 20 ms | 0.00004 | 0.00963 | 1.65° | 完整流程通过 |

另有深折叠起点的拒绝测试；它没有执行物理运动，不能计为起立成功。
18 项单元/回归测试通过，覆盖初始目标连续性、history、步态时钟、开始前重新核对当前姿态、停止时保留策略支撑、
过渡中停止、退出卸力、动作异常、力矩投影、坐标逆变换、租约过期、状态过期、
遥控中止、默认不创建发送器、裸机禁止执行、误输入不丢失当前支撑控制、开始命令不能覆盖超时故障，以及原只读脚本回归。

开发中曾发现：2 秒动作混入过慢，自由双足在接管早期失稳；改为 300 ms 后上述测试通过。
深折叠姿态初期存在大关节瞬态，不能仅看最终站稳判定安全，故当前明确拒绝该起点。

## 服务器复现仿真

```bash
cd ~/workspace/isaaclab_tron2/TRON2_WholeBody_LAB
git fetch origin
git switch sim2real
git pull --ff-only origin sim2real
conda activate limx
BUNDLE=logs/rsl_rl/sfyg_robust_5000_gpu1_20261027_131013/exported/model_5000
OUT="logs/mujoco/sfyg_lifecycle_$(date +%Y%m%d_%H%M%S)"
for DELAY in 0 4; do
  for VX in 0 .2; do
    python scripts/mujoco/test_sfyg_lifecycle.py \
      --bundle_dir "$BUNDLE" --delay_steps "$DELAY" --vx "$VX" \
      --output_dir "${OUT}/vx${VX}_delay${DELAY}" || break 2
  done
done
python -m pytest tests/test_sfyg_lifecycle.py tests/test_sfyg_locomotion_shadow.py -q
```

运行 MuJoCo 需要已有模型/网格；首次克隆需 `git submodule update --init --recursive`。
Jetson 只运行硬件入口，不需要安装 IsaacLab 或 MuJoCo。

## Jetson 操作

保持能运行之前只读脚本的 Python / torch / limxsdk 环境，不在本步骤升级 JetPack 或 SDK。

```bash
mkdir -p ~/workspace
cd ~/workspace
# 仅第一次执行；已有仓库就跳过 clone。
git clone --branch sim2real --single-branch https://github.com/Matt14-mander/TRON2_WholeBody_LAB.git
cd ~/workspace/TRON2_WholeBody_LAB
git fetch origin
git switch sim2real
git pull --ff-only origin sim2real
python3 -c 'import torch, numpy, limxsdk; print(torch.__version__)'
python3 scripts/hardware/run_sfyg_lifecycle.py --help
```

先复测原始只读推理（已下载的 bundle 不必重新从训练服务器传）：

```bash
cd ~/workspace/TRON2_WholeBody_LAB
python3 scripts/hardware/shadow_sfyg_locomotion.py \
  --bundle_dir "$HOME/models/sfyg_robust5000" --robot_ip 10.192.1.2 \
  --command 0 0 0 --duration 10 \
  --output_dir "$HOME/sfyg_shadow/logs/before_lifecycle_$(date +%Y%m%d_%H%M%S)"
```

机械臂和负载已装好、完成坐标检查、吊架和遥控急停就绪、机器人处于 Low-Level Dev 后，
复制检查模板到仓库外，逐项实际核对后填写，不要批量把 false 改 true 来绕过检查：

```bash
mkdir -p ~/sfyg_shadow/config
cp -n scripts/hardware/sfyg_robust5000_preflight.example.json ~/sfyg_shadow/config/robust5000.json
nano ~/sfyg_shadow/config/robust5000.json
```

`operator` 填现场操作者；四个检查项只有实际满足才填 true。
映射数组按已经核对的实机结果填写。模板 identity 只是候选值，不能通过修改零位让不匹配姿态强行过关。

在交互式 SSH 终端运行，**不要 nohup/后台启动**：

```bash
cd ~/workspace/TRON2_WholeBody_LAB
python3 -u scripts/hardware/run_sfyg_lifecycle.py \
  --bundle_dir "$HOME/models/sfyg_robust5000" \
  --preflight "$HOME/sfyg_shadow/config/robust5000.json" \
  --robot_ip 10.192.1.2 --execute \
  --output_dir "$HOME/sfyg_shadow/logs/lifecycle_$(date +%Y%m%d_%H%M%S)"
```

按现场状态逐条输入，每条回车，不要把下面整段一次性粘贴：

1. 吊架已可靠承重、机身竖直、腿接近站立：`prepare supported`。
2. 等待 `READY preparation_settled=True`，确认双脚接地、吊架保留防跌而不托起双脚，再输入 `start grounded`。
3. 首次只观察零速策略。状态机不会自行给行走指令。
4. 正常停止：`stop`。等待 `ZERO`，现场确认静止，再让吊架承重，输入 `release supported`。
5. 程序经过 RELEASE/DAMP/DONE 自动退出。若准备阶段就不继续，READY 也允许受支撑退出。

只有零速接管已经确认稳定，下一次才考虑在 POLICY 输入 `vx 0.15` 进行短距离前向测试；
当前仿真结果不能保证真机安全行走。发生异常用官方急停；`abort`/Ctrl+C 仅是尽力发送阻尼的进程中止。
本程序不会自动回蹲，也不会自动切回官方 Drive 模式。
