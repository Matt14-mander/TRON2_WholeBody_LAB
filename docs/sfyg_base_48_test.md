# SFYG BASE 48维模型测试

服务器模型目录：~/workspace/limx_deploy/tron2_rl_deploy_python/controllers/model/SFYG_BASE_TRON2A。
source.json 指向 locked_baseline 的 model_15000.pt，encoder 420→3、policy 48→10，ONNX 无 batch 维。

现有 main.py 是 WholeBody 78维入口，不直接支持该模型。新增 tron2_mujoco_sim/tools/test_sfyg_base_policy.py 不使用 SDK，不启动 main.py/simulator.py，不替换已有网络。输入按训练 PPO 的 encoder + proprio + command 顺序组成48维。

默认参考当前本地 BASE 资产：臂目标全零，臂/夹爪原生隐式 PD 800/40、200 Nm，腿部仍使用现有MIT PD/armature/effort配置，physics 1ms，策略50Hz。不是之前名义臂姿60/6配置。为稳定高刚度锁臂使用implicitfast和native反馈；不是对Isaac数值积分/速度限幅的逐项复刻。网络目录没有params.yaml，所以暂时借用原SFYG_TRON2A/params.yaml的18关节映射、腿PD、动作/观测缩放和步态参数；运行前需对照本轮实际训练配置。模型维度和校验和正确并不能证明这些配置相同。

Windows PowerShell 上传单个入口：

```powershell
scp 'E:\Robot\Sustech-多人形协作\tron2_mujoco_sim\tools\test_sfyg_base_policy.py' matt14@server-Super-Server:~/workspace/limx_deploy/tron2_mujoco_sim/tools/
```

服务器先停掉旧SDK控制器和仿真进程，激活limx：

```bash
conda activate limx
cd "$HOME/workspace/limx_deploy/tron2_mujoco_sim"
export DEPLOY="$HOME/workspace/limx_deploy/tron2_rl_deploy_python"
export BASE_MODEL="$DEPLOY/controllers/model/SFYG_BASE_TRON2A"
export PAINT_MJ_RUNTIME="$HOME/workspace/isaaclab_tron2/TRON2_WholeBody_LAB/logs/paint/runtime/mujoco327"
base_test() {
  PYTHONPATH="$PAINT_MJ_RUNTIME${PYTHONPATH:+:$PYTHONPATH}" \
    python -u tools/test_sfyg_base_policy.py \
    --deploy-repo "$DEPLOY" --model-dir "$BASE_MODEL" "$@"
}
base_test --command 0 0 0 --duration 10
```

先看零速完整结果，再运行低速：

```bash
base_test --command -0.25 0 0 --duration 10
```

图形桌面加--viewer；无显示的SSH不加。命令每次创建全新模型和状态。模型无额外payload、无estimator、无future wrench，仅评估BASE。

实际训练臂姿若不同，可用--arm-position六个弧度值，--arm-kp/--arm-kd/--arm-effort和--base-height显式调整。不要为了掩盖失败反复调参；先确认训练配置。零速会按当前本地训练的连续episode-time步态相位运行，历史首帧填满10帧。

每0.5s输出time,z,tilt_deg,vx_body,vy_body,max_abs_action；每物理步检查倾角超过35度或底座高度低于0.45m时终止并返回错误。结束打印COMPLETED、全过程最大倾角、热身后的机身系平均速度。完成不等于跟踪达标，需要比较实际速度与指令并观察高度/倾角。若本轮训练改了速度方向、动作缩放、关节零点、步态相位逻辑，需同步接口后再评估。

本地仅通过两个测试（输入顺序/维度检查和合成零输出ONNX的0.06s物理烟雾测试），不声称新model_15000的sim2sim已验证。SDK主入口尚未新增BASE模式；直接测试通过后再接SDK链路。
