# locomotion v4 训练审计（2026-10-24）

范围：nominal 10000 次更新的 TensorBoard 日志、用户提供的 1000/5000/9000/10000 checkpoint 评估，以及当前仓库 PPO、奖励和动作实现。没有取得服务器模型权重，没有完成修复后的仿真收敛实验。本次修改位于当前 paint-transport-v4 工作分支，尚未提交。

## 证据与边界

- 最后一次训练记录平均回合长度 29.19 控制步，即 0.5838 秒；与最后 checkpoint 的站立评估约 0.56 秒接近。退化存在于训练数据中，不能仅归因于评估链路。
- 日志有明显波动。连续 250 个日志点的最长平均回合长度出现在 step 7528–7777，约 6.71 秒。它只能提示候选模型，不能证明 model_7750 最好：runner 的 Train/mean_episode_length 是最近 100 个已结束回合的混合任务统计，不是所有环境的存活率或站立成功率。
- model_1000 保持直立但站立漂移约 0.65 m、yaw RMS 约 0.20–0.22 rad/s；model_5000 的三个站立回合有两个通过；model_9000 和 model_10000 均很快跌倒。三个固定评估 seed 不足以估计可靠成功率。
- 最后动作均值最大绝对值约 4.996；这是 rollout 的极值，不能解读为所有关节平均动作都接近 5。末尾采样动作裁剪比例约 4.5%。

## 已确认并修复

1. PPO 在 KL 提前停止之前没有累计该 minibatch 的 KL，最终只对接受的更新求平均。若首个 minibatch 接近零、第二个超限，日志会持续显示很小的 KL。现在包含拒绝批次，另记 max_checked_kl、kl_checks、optimizer_steps 和 early_stopped。
2. 原 KL 公式在标准差比值内添加 1e-5，使完全相同的 10 维 Normal 分布也产生约 1e-4 KL；自适应 LR 随即在每轮尚未执行优化的首批上调。现在相同分布 KL 为零，且首批不触发 LR 上调。
3. 提前停止阈值为 1.5 倍 desired_kl，原降 LR 阈值却为 2 倍；两者之间会停止而不降 LR。现在触发提前停止也降 LR。提前停止仍是停止后续 minibatch，不会回滚已经执行的更新。
4. 原源码快照没有包含实际使用的 PPO/runner/storage/actor/encoder。新运行将同时归档这些文件及 SHA256。

这些是可复现的实现或诊断缺陷，尚不能证明它们解释全部训练失败。

## 新增诊断：估计器改变有效策略

PPO 先更新 actor/critic，再单独监督更新速度估计器。即使 output_detach=True，后一次 rollout 输入 actor 的估计速度仍会随估计器变化。原 actor minibatch 的 KL 没有覆盖这个变化。

现在每轮在最多 2048 个 rollout 样本上记录：

- PPO/post_actor_probe_kl：actor 更新后相对采样策略的 KL；也能看到最后一个梯度步造成的变化。
- PPO/encoder_policy_probe_kl：仅估计器更新造成的有效策略变化。
- PPO/post_update_probe_kl：actor 和估计器都更新后相对采样策略的 KL。

这组指标用于定位问题，尚未加入估计器回滚或 KL 约束。teacher 的零输出估计器路径也有回归覆盖。

## 奖励审计

- 速度跟踪为正奖励；姿态、动作、站立实际速度等为负惩罚；failure 使用真实 terminated。未发现简单的正负号反转。
- failure 权重 -250 乘控制周期 0.02，单次失败贡献 -5。TensorBoard Episode/Episode_Reward/failure 为 -0.25 是按名义 20 秒归一化后的回合累计值，不能当作每步只扣 0.25。
- base_height 继承 target_height=0.7；函数虽然注释称 L2，实际计算绝对误差 L1。这个差异需要明确记录，不在没有对照试验时直接更改目标函数。
- standing_yaw 权重 -0.5 相对 standing_speed -4 较小，但还有 yaw 跟踪奖励。model_1000 漂移不能单独证明应直接增大某个权重。
- 动作/力矩平滑、姿态和接触项存在多目标权衡。没有从日志中证明“策略为了躲惩罚故意跌倒”；不能把假设作为根因。

## 动作与观测审计

- 训练和评估都使用：encoder(history360) + policy117 + command3 → actor10。估计器监督目标取 critic 前三个真实机体系线速度。
- 左腿 pitch/roll/yaw/knee/ankle、右腿同序；动作项显式验证关节顺序。
- actor 的 ScaledTanh(5) 只约束 Normal 均值，随机样本仍可能超界。runner 保存原始采样动作及其 log probability，向环境发送裁剪动作；不能简单把 buffer 动作改成裁剪值而保持原 log probability。
- 环境再次裁剪至 ±5，以 q_default + 0.25 * action 设置目标，即最大相对偏移 ±1.25 rad。当前没有针对每个关节目标的 soft-limit 裁剪；实际关节位置超限另有奖励惩罚。扩大后的目标范围和 PD 力矩饱和值得后续记录，但不能仅凭动作极值判定全部关节撞限位。
- 机械臂/夹爪由固定 PD 姿态保持，腿策略只输出十维。实体负载力每个物理子步重算坐标，nominal 实际外力为零。
- 默认 nominal 的 root/joint 复位无随机偏移，base mass/push 关闭。评估 seed 的差别不应直接描述成有意采样了不同初始姿态；还可能包含接触求解和复位状态差异。

## 验证与下一步

4 个新 PPO 回归测试通过，覆盖首批 LR、拒绝首批、拒绝第二批及估计器改变策略的诊断。已有 locomotion v4 9 项、transport v4 14 项、standwalk 13 项测试全部通过，共 40 项。包含真实 CPU PPO 更新与导出一致性；没有运行修复后的 Isaac Sim 长训。

下一次先在独立输出目录进行 250–500 次更新的诊断运行，并定期用确定性策略评估站立及行走。对比新 KL、实际 optimizer_steps、回合长度和动作裁剪率；若修复降低更新失控，再安排更长实验。若估计器漂移显著，应设计冻结估计器/降低其学习率的单变量对照。奖励、动作范围的修改应另开对照，不与本次 PPO 修复同时混入，以免无法归因。不能直接宣称修复后已收敛。
