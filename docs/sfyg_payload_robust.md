# SFYG：从能走的carry进入负载鲁棒性训练

2026-10-27。入口仍为`train_sfyg_reset.py`，新增显式`--phase robust`；source/carry的物理配置与原始训练核心保持不变。首次robust训练要求传入本路线的carry checkpoint，旧v3/v4不能混入。

服务器carry基线：`logs/rsl_rl/sfyg_carry_scratch_gpu1_10000_20261026_152656/model_10000.pt`。用户报告固定0.25 m/s、20秒测试完成，启动1秒后平均vx约0.230 m/s、最大倾角约4.82°。这只是单次无外力前进测试，不能替代抗扰动验收。

## 本轮设置

| 项目 | 首轮robust |
|---|---|
| 腿策略、奖励、PPO | 原版；没有增加KL回滚、学习率上限或新步态奖励 |
| 学习率 | 原始1e-3初始值与原始自适应规则；不保证保持在初始值 |
| 上肢 | 与carry一致的搬运姿态、800/40锁位PD |
| 负载质量 | 每次reset均匀采样0.25–0.75 kg，同步按质量比缩放原始惯量 |
| 摩擦 | 初始化时按16个倍率桶采样模型原值×0.8–1.2，每个环境固定；静/动摩擦用同一倍率，回弹不变 |
| 外力 | 实体负载质心处，世界水平面圆盘内采样，合力范数≤2 N；不施加yaw力矩 |
| 时序 | 复位后前0.5秒无外力，随后每2–5秒重采样，0.2秒线性过渡，25%的采样区间目标为零 |
| 名义环境 | 40%环境始终保持0.5 kg、原摩擦和零外力，保留无扰动行走样本 |
| 其他 | 原平地、命令分布、腿PD、延迟、观测噪声与复位；没有新增速度推扰或地形 |

本地原始模型摩擦读回为0.5，因此0.8–1.2倍率对应约0.4–0.6的系数；不是把摩擦系数直接设成0.8–1.2。地面材质保持原始值。摩擦采用startup随机化，避免在每次失败复位时反复更新大量PhysX材质；有限桶数避免生成过多唯一材质。

外力在每个物理子步按负载当前姿态从世界坐标转到局部坐标后重发。reset既清理采样器，也清理机器人外力缓冲。没有另外施加基座合成wrench。仅robust启用PhysX的`enable_external_forces_every_iteration`以支持持续外力，source/carry不改。

惯量始终以USD导入的默认质量/惯量为基准计算，不以已经随机后的数值再累乘。critic的质量项改为读取实际随机质量；顺序和维数保持不变。actor不读取真实外力，也没有新增未来外力预测输入。

## 权重与接口

policy42、history420、command3、actor48→10、critic123（另拼命令3）均与carry相同。carry→robust严格加载完整actor、critic、动作std和估计器，**不重建critic**。优化器重新创建，轮数从0开始；这是以carry权重初始化的新阶段，不是完整optimizer断点恢复。后续同robust阶段也支持权重初始化。

每次启动仍验证16个初始训练核心文件。新阶段只增加独立物理适配和诊断，不修改受保护的原PPO、runner、奖励、基础观测或SFYG资产配置。contract和checkpoint保存随机化参数、物理适配源码快照、来源checkpoint哈希、实际初始质量/摩擦及名义环境数。

## 同步与物理检查

```bash
cd ~/workspace/isaaclab_tron2/TRON2_WholeBody_LAB
conda activate limx
git pull --ff-only origin codex/sfyg-locomotion-reset

mkdir -p logs/rsl_rl
env LIVESTREAM=0 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 \
  python -u scripts/rsl_rl/train_sfyg_reset.py \
  --phase robust --mode probe --num_envs 16 --probe_steps 300 \
  --output_dir "logs/rsl_rl/sfyg_robust_probe_$(date +%Y%m%d_%H%M%S)" \
  --headless --device cuda:1 \
  --kit_args="--/renderer/multiGpu/enabled=false"
```

probe使用零腿动作并允许自动复位，检查物理施力、复位清零、实际质量/惯量、材质和名义环境。输出`physics_probe.json`；`passed_physics_checks=true`只表示物理适配通过，不是行走通过。

## GPU1启动首轮5000次更新

```bash
export SFYG_CARRY_CKPT="logs/rsl_rl/sfyg_carry_scratch_gpu1_10000_20261026_152656/model_10000.pt"
export SFYG_ROBUST_RUN="logs/rsl_rl/sfyg_robust_5000_gpu1_$(date +%Y%m%d_%H%M%S)"
printf '%s\n' "$SFYG_ROBUST_RUN" > logs/rsl_rl/latest_sfyg_robust_run.txt

nohup env LIVESTREAM=0 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 \
  python -u scripts/rsl_rl/train_sfyg_reset.py \
  --phase robust --checkpoint_path "$SFYG_CARRY_CKPT" \
  --payload_mass 0.5 --payload_mass_range 0.25 0.75 \
  --friction_scale_range 0.8 1.2 --force_n 2 --nominal_fraction 0.4 \
  --num_envs 4096 --max_iterations 5000 --save_interval 250 --seed 42 \
  --output_dir "$SFYG_ROBUST_RUN" --headless --device cuda:1 \
  --kit_args="--/renderer/multiGpu/enabled=false" \
  > "${SFYG_ROBUST_RUN}.console.log" 2>&1 < /dev/null &

ROBUST_PID=$!
printf '%s\n' "$ROBUST_PID" > "${SFYG_ROBUST_RUN}.pid"
echo "PID=$ROBUST_PID RUN=$SFYG_ROBUST_RUN"
tail -n 80 -f "${SFYG_ROBUST_RUN}.console.log"
```

应看到`[ROBUST]`中的范围、名义环境数量和`[RESET] phase=robust ... LR=0.001 early_stop=False`，然后进入Learning iteration。Ctrl+C只退出tail。新阶段最终文件是`model_5000.pt`，不是`model_15000.pt`。参数变化属于新物理任务，不能只凭原模型前进效果预测鲁棒性训练会成功。

500/1000轮先抽取候选，同时测试有扰动和零扰动，避免只看总reward。新增TensorBoard日志：`Robust/force_norm_mean_n`、`force_norm_max_n`、`force_active_fraction`、`payload_mass_mean_kg`、`friction_scale_mean`，以及`Motion/linear_progress_ratio`。这些是rollout中实际采样的统计，不是目标范围的重复打印。

## 两组评估

压力测试默认名义环境比例为0，避免单环境始终落在无扰动配额。摩擦每个进程startup固定；覆盖摩擦范围应换seed启动多个评估进程，或显式测试固定倍率0.8/1.0/1.2。

```bash
export SFYG_ROBUST_CKPT="$SFYG_ROBUST_RUN/model_1000.pt"

# 有负载质量变化、摩擦变化和水平外力：10回合。
env LIVESTREAM=0 python -u scripts/rsl_rl/train_sfyg_reset.py \
  --phase robust --mode eval --checkpoint_path "$SFYG_ROBUST_CKPT" \
  --payload_mass_range 0.25 0.75 --friction_scale_range 0.8 1.2 \
  --force_n 2 --nominal_fraction 0 --command 0.25 0 0 --episodes 10 \
  --output_dir "logs/rsl_rl/sfyg_robust_stress_$(date +%Y%m%d_%H%M%S)" \
  --headless --device cuda:0 --kit_args="--/renderer/multiGpu/enabled=false"

# 回测0.5kg、原摩擦、无外力；检查基本行走是否遗忘。
env LIVESTREAM=0 python -u scripts/rsl_rl/train_sfyg_reset.py \
  --phase robust --mode eval --checkpoint_path "$SFYG_ROBUST_CKPT" \
  --payload_mass_range 0.5 0.5 --friction_scale_range 1 1 \
  --force_n 0 --nominal_fraction 1 --command 0.25 0 0 --episodes 10 \
  --output_dir "logs/rsl_rl/sfyg_robust_nominal_$(date +%Y%m%d_%H%M%S)" \
  --headless --device cuda:0 --kit_args="--/renderer/multiGpu/enabled=false"
```

加`--video --episodes 1`可录像。该服务器GPU1录像曾有cuSOLVER初始化错误，因此这里播放用GPU0；权重可以跨GPU加载。逐步日志增加实际世界外力、负载质量、摩擦倍率和名义环境标志，终止前的数值会保留。验收需检查完整回合、速度/侧向摆动、实际迈步、跌倒率以及受力后的恢复，不把接触释放次数直接当有效步数。

压力测试也允许显式选择原carry checkpoint，作为未适应外力前的对照。报告会同时记录`phase=robust`和`checkpoint_phase=carry`，避免把对照误认为训练后的robust模型；source checkpoint不能直接用于这项评估。

本轮不自动升级到5 N、yaw力矩或粗糙地形。先保留这组有限范围结果，再依据实际表现决定下一阶段。

## 本地实现验证（2026-10-27）

- 14项CPU测试通过：保留原始训练核心、完整权重迁移、外力范数/水平面/释放、启动等待与部分复位、姿态变化下世界到局部转换、质量惯量不累乘、材质不累乘且回弹不变，以及推理模式下统计缓冲的生命周期。
- 原生Isaac物理探针16环境×300控制步通过。实际外力缓冲读回与提交值一致，峰值1.9833 N；竖直力、非负载刚体外力、附加力矩和reset后残留均为0。观测到负载质量0.2525–0.7406 kg；质量缓存误差0、惯量误差约2.3e-10、材质误差0。
- 使用本地carry短测权重完成64环境×8次原生PPO更新，保存model_0/4/8；actor48、policy42、history420、critic123保持一致。日志外力峰值≤1.9885 N，质量随复位变化，摩擦倍率按环境固定。
- 新robust checkpoint在受扰动环境严格加载并生成逐步数据；原carry权重也可在名义回测条件下加载。名义回测确认实际质量0.5 kg、摩擦倍率1、外力0。
- 测试用的是本地8轮短测权重，评估仍会跌倒；不构成抗扰动收敛或行走恢复证据。用户服务器实际能走的carry/model_10000.pt尚未用于本地长训，正式效果待服务器实验。

原始PPO/runner/奖励等16文件仍与f441d67一致，固定负载carry配置文件与e65a0a5一致。首轮实现只改独立物理适配、启动选择、明确的权重迁移和诊断。
