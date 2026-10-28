# 当前 T1 teacher 的多机实验接入

支持原有三种场景：并排双机共享箱体、前后leader–follower、三机轻量沙发。
保留原导纳模式；没有修改teacher/底层权重、SFYG观测、PD或任何训练源码。

## 必须区分的输入语义

当前teacher在训练时读取的是单机0.5kg负载COM上的真实外加力。
共享物体中，各机抓取反力同时包含重力、惯性、伙伴作用和内部拉扯，不能视为相同信号。
这次接入是显式的`--teacher_input grasp-proxy`实验，所有输出标为`is_acceptance=false`，不是student或完整PAINT迁移验收。
窗口、启动日志、interface与evaluation均标记`TEACHER-PROXY / EXPERIMENTAL`。

每台机器人的适配过程：

```text
自身抓点的物体→机器人世界wrench
  → 扣除落稳阶段的自身静态世界偏置
  → 用当前base旋转到本机坐标
  → Fx裁剪到[0,8]N，除以40；其他分量不输入当前前向teacher
  → 原teacher147维观测与原始9维动作
  → 原执行规则（只执行vx，固定机械臂）
  → 各自冻结的SFYG底层
```

不添加导纳、力增益、滤波、最低行走速度或reference前馈。
每机独立偏置、8帧臂历史、底层10帧历史、动作和延迟队列；没有leader指令广播或物体外力均分。
高层/低层50Hz，原PD200Hz，求解子步仅细分物理积分。
固定3秒落稳和2个PD tick延迟必须与模型契约一致；落稳最后1秒采样偏置，原有全机静态检查仍然生效。
撤力或leader停车后仍由teacher读取代理信号自主响应，不人为清零其速度来伪造停车。
运行超过18秒时使用持续的本机输入，不周期性重置teacher历史，也不人为重放单机施力曲线。

日志保留完整原始反力、世界偏置、完整校正后本机wrench、送入网络的proxy_fx、raw9和负Fx/超过8N样本计数。
因此可以看到负向力和大力被裁剪的比例，不能把裁剪后的0--8N记录当成真实物理反力范围。
当前teacher不能倒退、侧移或转向；后向Fx裁剪为0。Fy/Fz/Tx/Ty/Tz保留诊断但未输入网络。
共享质量/惯量与机械臂受力分布仍与单机训练不同，效果必须通过实际demo判断。

## 服务器同步与公共变量

```bash
cd ~/workspace/isaaclab_tron2/TRON2_WholeBody_LAB
conda activate limx
git fetch origin
git switch codex/sfyg-locomotion-reset
git pull --ff-only origin codex/sfyg-locomotion-reset

export SFYG_MJ_BUNDLE="logs/rsl_rl/sfyg_robust_5000_gpu1_20261027_131013/exported/model_5000"
export TEACHER_BUNDLE="logs/rsl_rl/sfyg_teacher_t1_20261028_130353/train/exported/model_5000"
```

无需重新导出。两个参数都是目录，分别有deployment.json或manifest.json。
可以为各机分别传兼容低层bundle、为各follower分别传teacher_bundle；绑定SHA和完整接口仍会严格检查。
这些策略在CPU推理，MuJoCo窗口需要在服务器图形桌面终端打开。

## 1. 并排双机：键盘给物体施力

```bash
python scripts/mujoco/run_sfyg_cooperate.py \
  --bundle_dir "$SFYG_MJ_BUNDLE" --controller teacher \
  --teacher_bundle "$TEACHER_BUNDLE" --teacher_input grasp-proxy \
  --grasp_model ball --physics_substeps 4 --delay_steps 2 --max_speed .3 \
  --keyboard --force_world 12 0 --max_force 16 --force_step 1 --max_torque 0 \
  --release_tail 10 --duration 120 --trace \
  --output_dir "logs/mujoco/sfyg_teacher_side_$(date +%Y%m%d_%H%M%S)"
```

3秒落稳后，物体外力按原2N/s变化率逐渐达到总12N；它不等于每机6N。
此初值是为了观察当前teacher较大力才响应的行为；若不走，先看各机proxy_fx，不要自动提高网络增益。
W/S增减总Fx，X撤力，R恢复12N，P暂停。首轮只用这些键，不用侧向力或偏航。
物体总力最高16N，不保证单机实际反力不超过8N；网络输入会裁剪并记录越界。
末尾10秒留足撤力时间：16N按2N/s需要8秒，不能沿用仅3秒的撤力尾段。

## 2. 双机前后跟随

```bash
python scripts/mujoco/run_sfyg_team.py \
  --scenario leader-follow --bundle_dir "$SFYG_MJ_BUNDLE" \
  --follower teacher --teacher_bundle "$TEACHER_BUNDLE" --teacher_input grasp-proxy \
  --max_speed .3 --leader_vx .20 --leader_max_speed .20 --stop_seconds 6 \
  --keyboard --duration 120 --trace \
  --output_dir "logs/mujoco/sfyg_teacher_tandem_$(date +%Y%m%d_%H%M%S)"
```

## 3. 三机轻量沙发

```bash
python scripts/mujoco/run_sfyg_team.py \
  --scenario sofa --bundle_dir "$SFYG_MJ_BUNDLE" \
  --follower teacher --teacher_bundle "$TEACHER_BUNDLE" --teacher_input grasp-proxy \
  --max_speed .3 --leader_vx .20 --leader_max_speed .20 --stop_seconds 6 \
  --keyboard --duration 120 --trace \
  --output_dir "logs/mujoco/sfyg_teacher_sofa_$(date +%Y%m%d_%H%M%S)"
```

前后和沙发场景中，键盘只控制leader：W/S改变速度，X停车，R恢复初值，P暂停。
leader使用速度控制，不使用teacher；后方每台独立运行teacher代理适配器。
leader初值选0.20m/s，因为已有测试显示无外力0.10m/s不能可靠启动。它不会自动把follower速度改成0.20。
前机仍为后部球铰牵引、followers原固定臂连接；沙发仍是1kg轻量模型。

## 输出与测试边界

`interface.json`记录teacher模型、输入转换和已知差异；`steps.jsonl`中的`controller`/`high_controller`记录每步代理诊断。
`evaluation.json`中每个teacher机器人的`teacher_proxy`包含负向与超过8N的样本计数。
`trace.npz`新增teacher147维输入和raw9，便于检查真正执行的是teacher，非导纳回退。
模型、语义、执行限幅或落稳/延迟契约不匹配会明确报错；不自动绕过标定和不稳定终止。
student仍未实现。原导纳命令不带teacher参数时行为保持不变。

本地完成原有多机回归和新增代理适配检查，包括坐标变换、静态偏置、输入裁剪与越界记录、独立状态、复位、模型绑定以及三种真实CLI链路。
用于覆盖接管后路由的固定支撑测试夹具有软抓取下垂，因此该项测试显式模拟标定关卡通过，只验证控制器调用和实际指令，不作为静态稳定性通过证据。真实demo不会使用这些测试支撑或模拟关卡。
本地没有服务器正式5000轮权重，只能验证工程链路，不能宣称多机稳定行走或停车已经通过。
