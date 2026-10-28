# 双 TRON2：键盘外力驱动的共享负载导纳 Demo

当前T1 teacher可通过显式抓取反力代理模式进行实验接入，见[三种多机场景的teacher命令与边界](sfyg_teacher_shared.md)。以下为原FT导纳说明。

2026-10-28。用户已报告 robust/model_5000.pt 通过单机 MuJoCo 测试。本入口复用同一份已导出的策略与单机物理校准，搭建两台同向并排的完整 SFYG、一个共享自由箱体、两处理想闭合抓取。第一版是参考FT导纳控制器，后续可作为PAINT高层策略的对照。

首轮服务器双weld测试在零指令/零外力阶段已出现相反方向的转动，5.96秒达到10 cm连接误差，内部水平反力峰值约107 N、物体外力最多1.5 N。这不是成功搬运结果，也不足以归因于导纳参数。新版修正为机器人侧FT测量，增加静态标定质量检查、求解子步，以及显式的球铰抓取对照。先检查零外力共享承载，再接导纳；PPO、奖励及低层权重不改。

```text
键盘 → 箱体上的世界坐标 Fx / Fy / Tz → 共同 MuJoCo 物理世界
                                          ↓
                         每台各自抓取反力 + 本机朝向
                                          ↓
                         偏置扣除 → 滤波 → 导纳 → 限速
                                          ↓
                          各自冻结的 locomotion policy
                                          ↓
                             腿部动作、PD与地面接触
```

键盘不向机器人发送共同速度。每台导纳只读取自身连接的六维反力和本机坐标轴，不接收箱体外力真值、另一台机器人的状态、箱体全局位置或队形误差。相机和诊断可以读取全局状态，但不反馈给控制器。两台可共享无状态网络权重，各自的10帧历史、上一动作、步态计数和延迟队列相互独立。

## 服务器同步与启动

复用**单机已经测试通过的导出目录**，不必重新训练或重新导出。进入服务器图形桌面的终端，在单机成功的环境中运行：

```bash
cd ~/workspace/isaaclab_tron2/TRON2_WholeBody_LAB
conda activate limx
git switch codex/sfyg-locomotion-reset
git pull --ff-only origin codex/sfyg-locomotion-reset

export SFYG_MJ_BUNDLE="logs/rsl_rl/sfyg_robust_5000_gpu1_20261027_131013/exported/model_5000"

# 加载真实策略，检查三套自由根关节、质量和初始抓取对齐；不推进仿真。
python scripts/mujoco/run_sfyg_cooperate.py \
  --bundle_dir "$SFYG_MJ_BUNDLE" --headless --check_only

# 先做两组零外力、零速度指令对照。不按键，不接通导纳。
for GRASP in weld ball; do
  OUT="logs/mujoco/sfyg_hold_${GRASP}_$(date +%Y%m%d_%H%M%S)"
  python scripts/mujoco/run_sfyg_cooperate.py \
    --bundle_dir "$SFYG_MJ_BUNDLE" --controller hold \
    --grasp_model "$GRASP" --physics_substeps 4 --delay_steps 2 \
    --force_world 0 0 --duration 20 --headless --output_dir "$OUT"
  cat "$OUT/evaluation.json"
done

# 仅在ball零指令承载稳定后，打开球铰抓取的导纳原型。
# 它允许相对转动，不是原双刚性抓取已经通过的证明。
export SFYG_COOP_RUN="logs/mujoco/sfyg_cooperate_$(date +%Y%m%d_%H%M%S)"
python scripts/mujoco/run_sfyg_cooperate.py \
  --bundle_dir "$SFYG_MJ_BUNDLE" \
  --grasp_model ball --physics_substeps 4 --delay_steps 2 \
  --keyboard --payload_mass 1.0 --spacing 1.0 \
  --duration 120 --trace --output_dir "$SFYG_COOP_RUN"

cat "$SFYG_COOP_RUN/evaluation.json"
```

若单机导出保存在别处，仅修改 `SFYG_MJ_BUNDLE`。策略仍在CPU推理，窗口和录像使用图形GPU，不启动Isaac。`--keyboard`不能与`--headless`同时使用；键盘事件来自MuJoCo窗口，先点击窗口使其获得焦点。关闭窗口会保存报告，不自动重置或续跑。

| 按键 | 作用 |
|---|---|
| W / S | 世界坐标Fx增加 / 减少0.5 N |
| A / D | 世界坐标Fy增加 / 减少0.5 N |
| Q / E | 世界坐标Tz增加 / 减少0.05 Nm |
| X | 请求撤除物体外力与外加力矩，按力变化率限制降到零 |
| R | 恢复命令行初始力；默认初始为零 |
| P | 暂停 / 继续物理仿真 |

默认水平合力范数≤4 N，yaw力矩绝对值≤0.2 Nm；力变化率≤2 N/s，yaw力矩变化率≤0.1 Nm/s。红色箭头表示实际施加给箱体的力，绿/蓝箭头表示各机器人测得的水平抓取反力；箭头长度为显示而限幅，日志保留原始值。标签给出当前模式、箱体实际外力和各自速度指令。

首次先观察3秒静态承载，再逐次按W增加到2–3 N，检查两机是否响应并持续迈步；按X检查撤力后的恢复。随后再测试S、A/D。Q/E属于新的小幅yaw力矩实验，当前低层训练未包含独立的负载yaw力矩随机化，应单独保留结果。

前3秒强制外力为零；即使提前按键，目标力也会等标定结束才生效。进入导纳前检查标定窗口：水平速度RMS≤0.05 m/s、yaw速度RMS≤0.10 rad/s、抓点误差≤1 cm且样本足够。若未稳定，以`unsettled_calibration`停止并保存报告，不把正在滑动/转动的均值称为静态偏置。`--controller hold`仍会继续，便于观察零指令承载。最后3秒请求撤力，导纳继续响应自身反力，不强行清零速度指令来隐藏残余运动。

## 无桌面录像与对照

自动测试：静止标定3秒，给共享物体+X方向2 N引导，最后3秒撤力：

```bash
export SFYG_COOP_VIDEO="logs/mujoco/sfyg_cooperate_force2_$(date +%Y%m%d_%H%M%S)"
MUJOCO_GL=egl python scripts/mujoco/run_sfyg_cooperate.py \
  --bundle_dir "$SFYG_MJ_BUNDLE" \
  --grasp_model ball --physics_substeps 4 --delay_steps 2 \
  --headless --video --force_world 2 0 --duration 30 \
  --payload_mass 1.0 --output_dir "$SFYG_COOP_VIDEO"

echo "视频：$SFYG_COOP_VIDEO/preview.mp4"
cat "$SFYG_COOP_VIDEO/evaluation.json"
```

去掉`--video`可纯headless检查，无需EGL。桌面运行可以同时加`--video`录像。`--force_world FX FY`和`--yaw_torque Tz`设置初始请求，键盘模式下也可使用。

相同外力下，把`--controller hold`加到命令中可得到零速度指令对照：两台低层与PD照常工作、物体照常受力，实际发送给低层的速度为零。日志同时保留导纳建议值与实际指令，用于区分主动跟随和被物体拖动。对照不改变读取的反力，也不把它替换成已知外力的均分。

支持`--bundle_dir path0 path1`为两机分别加载权重；两份导出必须保持相同的关节默认姿态、时间步、接口和USD来源。现阶段先使用同一份成功的robust权重，避免同时引入策略差异。

## 负载与物理连接

- 每台保留29个完整USD刚体和原腿部/上肢PD、转子惯量、腿延迟、自碰撞。执行器200 Hz、低层50 Hz；`--physics_substeps 4`只把每个5 ms执行器tick细分成4个1.25 ms求解步，目标与延迟队列仍每5 ms更新。默认1子步可保留旧积分条件，用于独立对照。
- 删除两台各自的0.5 kg实体负载，改成一个默认1 kg共享箱体。总质量约80.046 kg，箱体质量不是额外加在原先两份负载之上。
- 箱体有独立自由关节、质心、完整箱体惯量和碰撞。默认尺寸0.20×1.12×0.08 m，随机器人间距调整横向宽度。没有重力补偿或与世界的固定连接。
- 每机使用真实名义FK下的抓取点：`gripper_base_Link`局部(0.15,0,0)，与单机负载质心偏移相同。灰色细杆是无质量、无碰撞的理想把手延伸，手与物体之间的真实夹爪接触抓取尚未建模。
- 在两个抓取site之间使用MuJoCo weld，初始位置和朝向均对齐；不运行IK，也不在运行中修改机器人或箱体位姿。只过滤箱体与相邻夹爪的碰撞，其他碰撞保留。
- 约束`solref=.02 1`、`solimp=.99 .999 .001`、`torquescale=.2`。weld是带数值柔性的理想闭合连接，连接误差和角度误差均记录。

`--grasp_model ball`则使用两处3行`connect`点约束，只传力、不传抓点力矩。箱体质心移到抓点连线下4 cm，避免沿连线的自由转动处于中性平衡；抓点朝向差只是相对角度，不再是约束误差。球铰采用`solref=.04 1`和恒定`solimp=.99 .99 .001`，避免这个闭链对照中的状态相关刚度切换。它是明确不同的抓取模型，用来隔离转动耦合；不能用它的成功宣称双weld已解决。真实刚性抓取最终仍可能需要机械臂柔顺控制或共享负载训练。

依据：[MuJoCo site-based weld定义](https://mujoco.readthedocs.io/en/3.2.7/XMLreference.html#equality-weld)、[passive viewer键盘回调](https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer)。没有将物体位移直接映射成机器人位置目标，也没有悬挂、mocap或隐藏的基座引导力。

## 反力与导纳参数

各自抓取反力从该连接的`J^T*efc_force`提取，在**本机器人自由根关节**的六个自由度上反解抓取site的空间Jacobian，得到**物体→机器人、世界坐标、作用点在本机实际抓取site**的wrench。含重量、惯性和两机相互作用，不等同于训练时施加给单机独立负载的外力。

旧版从物体侧反力取反并移到机器人抓点；软约束发生间隙时，这会混入作用点差异造成的力矩。新版同时记录`grasp_world`（本机实际测量）和`legacy_payload_estimate_world`（旧算法对照），后者不输入控制器。重构检查分别核对机器人侧和物体侧；物体动力学残差仍由物体侧真实约束力计算，不把旧重构接近零当作两侧力矩等价的证据。这项修正影响导纳接管后的输入，不能解释接管之前的零指令失稳。

初始3秒给零速度；最后1秒各自标定静态wrench偏置。该偏置仅用于控制器信号处理，不施加任何抵消重力的物理力。滤波状态和虚拟速度保存在世界轴，更新时旋转到本机轴，避免转向时混用历史坐标。

导纳采用`M*dv/dt + D*v = filtered_grasp_wrench`，零刚度，不叠加队形反馈。平动默认虚拟质量4 kg、阻尼8 Ns/m、低通时间常数0.3 s；水平分量死区0.15 N，yaw死区0.03 Nm。平动速度范数限制0.15 m/s、角速度绝对值限制0.2 rad/s，平动/转向加速度限制0.10 m/s²、0.3 rad/s²。

例如本机稳态水平反力1 N、偏置为零且超过死区时，平动请求约(1−0.15)/8=0.106 m/s。没有照搬Go2的40 N速度归一化，避免小力只能产生接近零的速度；也没有增加强制最低行走速度。参数可显式修改`--virtual_mass`、`--damping`、`--filter_tau`、`--max_speed`、`--max_yaw`，首测保留默认值。

4 N是共享物体的外力上限，**不保证各自抓取反力≤2 N**。步态不同步、箱体转动及内部拉扯可能产生更大的力与力矩；程序不篡改物理反力来满足训练范围，报告保留峰值及标定后水平反力超过2 N的样本比例。

## 输出与判读

- `interface.json`：两机策略路径/哈希、总质量、连接形式、三套自由根关节、初始对齐、PD延迟和导纳设置。
- `steps.jsonl`：每个控制步的请求/实际物体外力、共享物体位置、各机实际速度/倾角/臂误差、原始抓取wrench、偏置、滤波后导纳输入、建议与实际速度指令、连接位置/角度误差、反力重构与物体动力学残差。
- `evaluation.json`：完成/终止原因、物体位移、连接误差、各机速度与末秒停车表现、反力峰值和训练范围外比例。始终标为诊断。
- `trace.npz`：`--trace`时记录各机独立观测/历史/动作，可复查低层输入。
- `preview.mp4`：`--video`时保存。

任何机器人倾角>45°、根高度<0.45 m、抓取点误差>0.1 m、非有限状态或MuJoCo内部重置会停止并写报告。`completed=true`只表示达到指定仿真时间；搬运成功还需检查实际行走、受力跟随、箱体位移及撤力后的残余运动。

## 本地验证边界

新增8项检查通过，加上单机、原始训练核心和随机化阶段的22项回归，共30项相关检查通过。覆盖三套自由根关节/唯一共享质量/无世界支撑、独立历史与动作、外力仅写入共享物体、重置清零、独立固定支撑夹具中的静载重量和正负外力/力矩、反号故障检出、释放后自由下落、键盘方向与合力限幅、导纳偏置/坐标转换/限速/撤力、视觉不改变物理状态，以及真实CLI的admittance/hold与轨迹报告。固定支撑只存在于解析载荷的测试夹具，生产场景没有这项支持。

本地MuJoCo 3.12.0下，短测权重完成0.6秒双机物理/录像短测，画面已检查；抓取点误差峰值约4.55 mm，抓取角误差峰值约4.37°，反力提取重构误差约8.9e-16。它不是服务器训练成功的5000轮权重，不能据此宣称长时间协同行走或停车通过。正式效果使用上面的服务器命令验证。单机已有入口与原始训练核心做回归检查，PPO、奖励、KL及学习率未改。

本次诊断修订的12项双机检查通过，新增覆盖球铰静载承重/抓点零力矩、带间隙时机器人侧与旧物体侧力矩的区别、求解子步不改变执行器tick及延迟、转动窗口不能通过静态标定。真实CLI分别检查weld/admittance、weld/hold和ball/4子步/admittance，并验证标定拒绝时没有施加引导外力。球铰的独立静载夹具使用4子步验证。

本地短测权重还完成ball/4子步/hold的0.4秒原生录像检查，抓点位置误差峰值约1.17 cm；机器人侧点力矩约9.2e-16 Nm，旧物体侧移点估计与其最大差约0.027 Nm。这里的短测窗口仍在启动瞬态，不能证明零指令承载通过。它们验证测量与物理适配，不替代服务器正式权重的20秒静态承载和后续导纳实验。
