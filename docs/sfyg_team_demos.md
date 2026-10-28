# TRON2 前后跟随与三机沙发 Demo

独立入口：`scripts/mujoco/run_sfyg_team.py`。先使用已有SFYG carry/robust低层与本地FT导纳。
现有单机、并排双机入口保留。没有修改训练入口、PPO、teacher、底层观测/动作、PD或策略权重。

## 场景和物理边界

| 场景 | 角色与布局 | 默认共享负载 |
|---|---|---|
| `leader-follow` | 同向前后两台；前机后部球铰牵引，后机保持原机械臂姿态 | 1 kg自由横梁，基座间距1.8 m |
| `sofa` | 前方1台leader、后方左右2台follower；同向 | 1 kg轻量沙发，后机横向间距1.1 m |

leader后部接口位于base局部`(-.22,0,.45)m`，通过可见无质量理想支架连接。
这改变leader的受力位置，需单独验证，不能把原单机测试视为后部受载已经通过。
所有机械臂仍保持原名义前伸姿态；不是前机用手向后抓取，也未加入IK、拾取或放下。
follower在原gripper局部`(.15,0,0)m`连接，默认球铰；`--grasp_model weld`只改变follower连接，leader始终为球铰。

共享物体只有一个freejoint，原每台独立负载被移除；两机3个自由根、三机4个自由根。
沙发的框架/靠背/坐垫/扶手/腿由碰撞几何构成，按体积分配给定总质量，由MuJoCo计算质心和惯量。
把手是可见、无质量、无碰撞的理想刚性延伸。没有世界支撑、基座位姿驱动、隐藏引导力或物体轨迹驱动。
生产运行每个物理tick将外加wrench设为零，运动来自leader腿部驱动和真实机械连接。
默认1 kg是轻量演示物体，不代表真实重量沙发搬运已验证；加质量须单独测试。

200 Hz执行器、50 Hz底层推理沿用原接口；默认4个求解子步和2个执行器tick延迟。
求解子步不改变PD或延迟队列更新频率。除局部抓取碰撞过滤外保留机器人/物体碰撞。

## 控制与替换接口

```text
键盘或固定速度 → leader底层策略 → 机械连接 → 自由共享物体
                                             ↓
                         各follower自身抓取反力、朝向及臂状态
                                             ↓
                          高层接口 → 本机底层策略 → 腿PD
```

- leader默认0.10 m/s，最大0.15 m/s，加速度0.05 m/s²，仅沿本机x轴。
- follower默认`admittance`：复用现有4 kg虚拟质量、8 Ns/m阻尼、0.3 s低通，最大0.20 m/s。
- 默认`--follower_axes forward`仅输出本机vx，允许正负以缓解拉扯；vy/wz为零。`planar`开放侧向/偏航，是待独立验证的扩展。
- 前3 s全体零指令，最后1 s标定各follower反力偏置。速度、yaw和抓点误差检查覆盖全部机器人，包括第三台。
- 标定不稳即停止。`--hold_all`是零指令物理诊断，允许继续观察标定失败，不用于绕过正式跟随检查。
- 最后5 s让leader按加速度限制降到零；follower继续响应实际反力，不被强制归零。
- `--follower hold`仅将followers的执行速度设零，leader仍运动，用于检查主动跟随和被动拖动的区别。
- 两个模式均不向follower广播leader速度、位置、目标轨迹或物体全局位姿。

`sfyg_team_control.py`定义`FollowerPolicy`、`FollowerInput`和`HighAction`。
每台follower独立实例、滤波/偏置状态；每台低层有独立历史、上一动作、步态计数、延迟队列。
`FollowerInput`包含自身抓取世界wrench、base旋转、臂位置误差/速度/上一执行目标偏移及本地时间。
`HighAction`输出本机坐标速度3和臂偏移6；本阶段非零臂输出会被拒绝，保留原固定臂执行。

teacher/student接口位置已经保留，但没有声称接入现有T1权重。
传`--follower teacher`或`student`会在仿真前明确报错，不静默退回导纳。
后续适配必须校验checkpoint契约、历史与reset、力语义、动作范围，并实现独立适配器。
当前T1真实外加力不等于共享负载抓取反力；不能直接替换输入，也不能将物体外力简单均分。
策略先采用启动时选择；不支持运行中无缝切换。
底层`--bundle_dir`支持一份共享权重，或按leader/followers顺序传2/3份兼容bundle。
当前仅支持同构SFYG物理接口；不声称已经支持异构人形leader。

## 服务器同步

```bash
cd ~/workspace/isaaclab_tron2/TRON2_WholeBody_LAB
conda activate limx
git fetch origin
git switch codex/sfyg-locomotion-reset
git pull --ff-only origin codex/sfyg-locomotion-reset
git submodule update --init --recursive
export SFYG_MJ_BUNDLE="logs/rsl_rl/sfyg_robust_5000_gpu1_20261027_131013/exported/model_5000"
```

复用单机已经通过的导出目录，不需要重训或重新导出。若路径不同，只改该变量。
MuJoCo策略在CPU推理，不启动Isaac或修改当前teacher训练配置；窗口/录像需要图形资源。

## 首次运行与键盘控制

先检查两套模型和接口，然后各跑20秒零指令承载：

```bash
for SCENE in leader-follow sofa; do
  python scripts/mujoco/run_sfyg_team.py \
    --scenario "$SCENE" --bundle_dir "$SFYG_MJ_BUNDLE" \
    --headless --check_only || break

  python scripts/mujoco/run_sfyg_team.py \
    --scenario "$SCENE" --bundle_dir "$SFYG_MJ_BUNDLE" \
    --headless --hold_all --duration 20 \
    --output_dir "logs/mujoco/${SCENE}_hold_$(date +%Y%m%d_%H%M%S)" || break
done
```

检查报告中的倾角、臂误差、连接误差和标定结果；`completed`只表示运行到了指定时长。
随后在服务器图形桌面的终端分别运行：

```bash
# 两机前后跟随
python scripts/mujoco/run_sfyg_team.py \
  --scenario leader-follow --bundle_dir "$SFYG_MJ_BUNDLE" \
  --keyboard --leader_vx .10 --duration 120 --trace \
  --output_dir "logs/mujoco/sfyg_tandem_$(date +%Y%m%d_%H%M%S)"

# 三机轻量沙发（关闭上一个窗口后再运行）
python scripts/mujoco/run_sfyg_team.py \
  --scenario sofa --bundle_dir "$SFYG_MJ_BUNDLE" \
  --keyboard --leader_vx .10 --duration 120 --trace \
  --output_dir "logs/mujoco/sfyg_sofa_$(date +%Y%m%d_%H%M%S)"
```

3秒标定通过后leader自动缓慢加速。若希望起始速度为零，改`--leader_vx 0`并按W启动。
点击MuJoCo窗口使其获得焦点：W/S增减leader速度0.025 m/s，X请求停车，R恢复启动设定速度，P暂停。
S最小降到0，不是leader倒车；没有A/D转向操作。箭头显示真实抓取反力，角色标签显示执行vx。
`--camera fixed`使用固定机位；默认跟随物体，地面标线保留位移参照。

## 自动录像与对照

```bash
for SCENE in leader-follow sofa; do
  MUJOCO_GL=egl python scripts/mujoco/run_sfyg_team.py \
    --scenario "$SCENE" --bundle_dir "$SFYG_MJ_BUNDLE" \
    --headless --video --duration 30 --leader_vx .10 --trace \
    --output_dir "logs/mujoco/${SCENE}_video_$(date +%Y%m%d_%H%M%S)"
done
```

桌面模式也可加`--video`。没有EGL时去掉`--video`运行纯headless诊断。
相同参数增加`--follower hold`可以录制follower零指令对照；不能用它证明主动协作。

输出`interface.json`、`steps.jsonl`、`evaluation.json`，可选`trace.npz`和`preview.mp4`。
记录bundle/源码哈希、质量/惯量/连接、各机指令与速度、真实反力、物体位姿/倾角、间距范围、连接误差和末秒速度RMS。
跌倒、抓点误差>10 cm、物体接地、数值异常或MuJoCo内部重置均停止；CLI非完成退出码2，异常为非零。
关闭窗口会保存报告并标记未完成，不能当作完整测试通过。

## 验证范围（2026-10-28）

- 原有27项单机/双机/SFYG回归通过，新增9项团队检查通过。
- 覆盖2/3机自由根与唯一负载、每个原机器人刚体质量、名义FK对齐、第三份配置校验、第三机标定、独立状态和复位。
- 三机固定支撑测试夹具验证1kg重量与8N水平外力的反力平衡、球铰零点力矩及动力学重构；固定支撑仅在测试夹具中。
- 真实CLI场景短测、控制器替换注入、独立指令路由、标定拒绝、策略接口拒绝错误臂输出、只读视觉覆盖已检查。
- 两个场景用本地短训导出各完成0.4秒原生MuJoCo录像，并查看画面。三机12秒负例在1.54秒跌倒，正确记录失败并退出2，没有拼接重置或报告成功。
- 本地没有服务器正式5000轮底层权重；短测验证工程链路，不证明长时间跟随、停车或三机稳定承载。正式效果需使用服务器bundle测试。
