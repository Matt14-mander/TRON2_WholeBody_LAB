# SFYG 单机 MuJoCo：固定搬运姿态与实体负载

入口 `scripts/mujoco/export_sfyg.py` 与 `scripts/mujoco/run_sfyg.py`。仅支持从 f441d67 恢复的 `train_sfyg_reset.py` source/carry/robust checkpoint。当前服务器候选为 `logs/rsl_rl/sfyg_robust_5000_gpu1_20261027_131013/model_5000.pt`。本阶段只完成单机部署；没有接入导纳或双机约束。

不改变训练奖励、PPO、KL、学习率或原始资产。部署模型由仓库现有 SFYG MJCF 在内存中构建，并以 USD 参考校准关节坐标与质量惯量。训练 checkpoint 原样保留，导出确定性 actor 均值与历史速度估计器，不采样动作噪声。

## 同步与导出

```bash
cd ~/workspace/isaaclab_tron2/TRON2_WholeBody_LAB
conda activate limx
git switch codex/sfyg-locomotion-reset
git pull --ff-only origin codex/sfyg-locomotion-reset

export SFYG_MJ_CKPT="logs/rsl_rl/sfyg_robust_5000_gpu1_20261027_131013/model_5000.pt"
export SFYG_MJ_BUNDLE="${SFYG_MJ_CKPT%/*}/exported/model_5000"

python scripts/mujoco/export_sfyg.py \
  --checkpoint "$SFYG_MJ_CKPT" --output_dir "$SFYG_MJ_BUNDLE"
```

导出无需启动 Isaac，无需 GPU，也无需 `pxr`。使用仓库内原始网络类严格加载 actor/critic/encoder，再保存部署用 actor+encoder；随机批量输入对比 eager/TorchScript/重新加载的输出。生成 `policy.pt` 和 `deployment.json`，后者包含关节名称、默认角度、观测顺序/缩放、PD参数、延迟、负载、频率、源模型和导出权重哈希。已有非空导出目录不会覆盖；需要重导出时显式选择新目录。

MuJoCo 运行需 `torch`、`numpy`、`mujoco>=3.2.7`；录像另需 `imageio` 和 FFmpeg。先检查现有环境：

```bash
python -c "import mujoco, torch, imageio; print('MuJoCo', mujoco.__version__, 'Torch', torch.__version__)"
```

若环境缺少 MuJoCo/录像依赖，可以在该 conda 环境内安装，不需要升级 Torch 或 RSL：

```bash
python -m pip install "mujoco==3.2.7" "imageio[ffmpeg]>=2.31"
```

## 首次服务器测试

先核对接口，不推进仿真：

```bash
python scripts/mujoco/run_sfyg.py \
  --bundle_dir "$SFYG_MJ_BUNDLE" --headless --check_only
```

桌面播放前进指令，默认0.5 kg、摩擦系数0.5、没有外力，不需要启动 Isaac。策略在CPU推理，窗口和录像仍使用图形GPU：

```bash
export SFYG_MJ_FORWARD="logs/mujoco/sfyg_robust5000_forward_$(date +%Y%m%d_%H%M%S)"

python scripts/mujoco/run_sfyg.py \
  --bundle_dir "$SFYG_MJ_BUNDLE" \
  --command 0.25 0 0 --duration 20 --trace \
  --output_dir "$SFYG_MJ_FORWARD"

cat "$SFYG_MJ_FORWARD/evaluation.json"
```

窗口跟随机器人，实时播放。关闭窗口会结束本次测试并记录 `viewer_closed`，不会重启机器人。首次建议先运行这条前进测试，与 Isaac 的0.25 m/s对照。

没有桌面或需要录像时，用 EGL 渲染；CPU推理仍保持不变：

```bash
export SFYG_MJ_VIDEO="logs/mujoco/sfyg_robust5000_video_$(date +%Y%m%d_%H%M%S)"

MUJOCO_GL=egl python scripts/mujoco/run_sfyg.py \
  --bundle_dir "$SFYG_MJ_BUNDLE" \
  --command 0.25 0 0 --duration 20 --headless --video --trace \
  --output_dir "$SFYG_MJ_VIDEO"

echo "视频：$SFYG_MJ_VIDEO/preview.mp4"
cat "$SFYG_MJ_VIDEO/evaluation.json"
```

无需录像的纯物理测试只加 `--headless`，无需设置 `MUJOCO_GL`。EGL录像使用图形GPU，但不使用 CUDA 策略推理。

前进通过后分别做站立、起停和水平外力测试；每条命令自动生成独立输出目录：

```bash
# 站立20秒；步态观测仍按源码的回合时间推进。
python scripts/mujoco/run_sfyg.py --bundle_dir "$SFYG_MJ_BUNDLE" \
  --command 0 0 0 --duration 20 --headless

# 前3秒站立，3秒加速，末尾3秒减速、再保持3秒零指令。
python scripts/mujoco/run_sfyg.py --bundle_dir "$SFYG_MJ_BUNDLE" \
  --command 0.25 0 0 --duration 30 --profile start-stop --headless --trace

# 启动0.5秒后，以0.2秒过渡施加世界坐标水平力；施力到负载质心。
python scripts/mujoco/run_sfyg.py --bundle_dir "$SFYG_MJ_BUNDLE" \
  --command 0.25 0 0 --duration 20 --force_world 0 2 --headless
```

`--force_world FX FY`为固定方向诊断，范数限制≤2 N，没有额外力矩；并非复刻训练时完整随机力时序。`--payload_mass`/`--friction`可以做固定参数对照，首测保留默认值。默认 seed=42，为两组腿执行器独立采样0–4个5ms物理步延迟；报告保存实际采样值。可用 `--delay_steps 0`或2显式指定全部腿的延迟，锁位机械臂仍无延迟。不同随机种子或延迟下的结果要分别保留。

## 保留的策略语义

| 项目 | 部署实现 |
|---|---|
| 当前观测42 | 角速度3×0.25、重力3、腿相对角度10、腿速度10×0.05、上一原始动作10、步态sin/cos2、步态参数4 |
| 观测裁剪 | 与训练一致，先按各项clip再scale；部署不加观测噪声 |
| 历史 | 10×42，旧到新、按时间展平；首次观测重复填满历史，reset清理 |
| 估计器 | 420→3，保留已训练权重；不以MuJoCo真实线速度替换 |
| actor | 拼接估计3＋当前观测42＋速度指令3，48→10 |
| 频率 | 物理200 Hz，低层50 Hz |
| 步态 | 0.8 Hz、offset0.5、duty0.5、height0.2；相位由回合步数计算，站立不冻结观测相位 |
| 腿动作 | `q_default + 0.25*a`，没有新增tanh、动作裁剪或目标软限位 |
| 上肢 | carry/robust搬运姿态，6臂＋2夹爪，PD800/40、力上限200；不加重力补偿 |
| 负载 | gripper_base局部偏移(0.15,0,0)，实体0.20×0.12×0.08 m箱体，默认0.5 kg，具有质量/惯量/碰撞 |

机械臂没有焊死在世界或删除质量；其关节由PD锁位。负载为夹爪的固定子刚体，过滤其与相邻夹爪的接触。机器人自碰撞保留，没有基座轨迹驱动、mocap、悬挂或运行时位置重置。

## 模型校准及已知差异

`sfyg_asset_reference.json`来自本仓库USD的29个刚体、28个连接。部署重建零位parent/child变换和关节轴，读取质量、质心、主惯量与主轴；补齐旧MJCF遗漏的0.01 kg `base_imu`。原始无负载质量39.523 kg、默认带载40.023 kg。脚踝/yaw与其他腿关节分别采用训练的40/140力矩上限和0.053923687/0.161777558转子惯量。

USD的两个空相机帧 `d435_Link`/`d435_optical_frame`使用自动质心哨兵值，部署使用其原MJCF的零质心，报告列出该假设；不把-inf写入物理模型。参考的USD层哈希启动时校验，资产改变会拒绝继续使用旧参考。

PD在MuJoCo中使用 `general`执行器表达 `kp*(target-q)-kd*dq`，通过 `implicitfast`积分并施加力上限。它不是PhysX隐式驱动求解器的逐步等价实现。当前训练使用的旧 `velocity_limit`在本地Isaac隐式执行器中不生效；部署记录该参数，不擅自引入硬速度裁剪。

接触使用3维接触和0.5切向摩擦，移除旧MJCF中额外的足端扭转/滚动摩擦。MuJoCo与PhysX的接触、静动摩擦与回弹求解不等价，不能仅凭参数一致承诺行走迁移。相关API参考：[MuJoCo执行器定义](https://mujoco.readthedocs.io/en/stable/XMLreference.html#actuator-general)、[mj_objectVelocity](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-objectvelocity)。速度统计由世界CoM速度旋转到常规base_Link坐标，避免误用惯性主轴。

刷新参考是资产维护操作，常规部署不需要。需要独立 `usd-core` 或已配置的OpenUSD Python运行时：

```bash
python scripts/mujoco/sfyg_asset_reference.py --refresh
```

## 输出与判读

- `interface.json`：checkpoint/策略/原始MJCF哈希、实际质量、PD、延迟、惯量对照和仿真版本。
- `steps.csv`：指令与实际vx/vy/wz、额外vz、世界位置、倾角、机械臂误差、接触、外力、力矩饱和和估计速度。
- `evaluation.json`：是否完成、结束原因、位移、平均速度、跟踪RMSE、倾角与臂误差。它始终标为诊断，不自动宣称搬运验收通过。
- `trace.npz`：仅`--trace`时保存，含当前观测、10帧历史、指令、原始动作、目标角和关节角。
- `preview.mp4`：仅`--video`时生成。

倾角>45°、根高度<0.45 m、非有限状态或MuJoCo内部重置会停止并保存报告；这是MuJoCo诊断阈值，不声称与Isaac的接触终止条件相同。接触释放次数不是经过验证的有效步数。起停应查看CSV中末尾保持段实际速度，不以全回合平均值代替停车效果。

## 本地验证（2026-10-27）

22项测试通过：原始训练核心身份、权重迁移、负载随机化，以及新增的观测顺序/缩放、历史初始化和复位、USD关节坐标FK、质量惯量、原始PD公式和力上限、动作不裁剪、上下肢延迟、施力/复位、真实TorchScript导出与CLI轨迹报告。

本地robust 8轮短测权重导出前后最大误差0，完成单机模型加载、物理推进和1秒MP4录像，画面已检查。该权重在MuJoCo约1.48秒达到跌倒阈值，不能证明行走成功；服务器正式5000轮权重未在本地取得，效果待上述服务器测试。

同时修复Isaac评估第2回合reset发生在inference_mode之外的问题；本地使用短测模型连续3回合生成完整evaluation.json，未再出现该错误。短测模型仍会跌倒，与此reset修复无关；训练核心没有改变。
