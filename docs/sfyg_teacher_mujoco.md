# 当前 T1 teacher 的单机 MuJoCo demo

入口`run_sfyg_teacher.py`直接加载已经导出的teacher与对应SFYG低层，不修改权重，不训练或校正输出。
它用于先查看当前5000轮teacher的实际行为；原来的单机速度播放、FT导纳、双机和三机场景均保留。

## 输入和执行

- 0.5kg实体负载，完整SFYG模型、原名义搬运臂姿态与PD；沿用既有USD质量/惯量校准及MuJoCo接触映射。
- 严格检查teacher绑定的低层checkpoint SHA、完整低层接口、teacher契约和两个导出模型的SHA。
- teacher：8×18维臂历史（位置偏差、原单位速度、上一执行目标偏差），加当前归一化真实外加力，共147维。
- 历史与训练端一样在reset时清零；初始actor历史全零；每控制步追加一次，旧到新。
- 高层和低层均50Hz，PD与延迟队列200Hz；使用teacher契约的固定2 tick延迟。
- 原始9维输出按训练接口执行：仅vx，范围[0,0.3]m/s，<=0.02归零；其余维度不执行，原臂姿态保持。
- 使用真正的原始actor输出，不添加reference前馈、最低行走速度、滤波或微调。
- 每回合18s：3s落稳、2s加力、8s保持、2s撤力、3s零力。只有初始落稳阶段覆盖速度；撤力后仍由teacher自己输出停车指令。
- 外力幅值每20ms保持，每5ms按当前base轴重新旋转，施加在自身负载COM；无外加力矩。
- 这是**特权teacher**：读取已知外加力，不是抓取反力、FT导纳或student。本入口不连接共享负载，不能据此宣称多机teacher适配已经完成。

## 服务器同步与启动

在服务器图形桌面的终端使用原来的limx环境：

```bash
cd ~/workspace/isaaclab_tron2/TRON2_WholeBody_LAB
conda activate limx
git fetch origin
git switch codex/sfyg-locomotion-reset
git pull --ff-only origin codex/sfyg-locomotion-reset
git submodule update --init --recursive

export SFYG_MJ_BUNDLE="logs/rsl_rl/sfyg_robust_5000_gpu1_20261027_131013/exported/model_5000"
export TEACHER_BUNDLE="logs/rsl_rl/sfyg_teacher_t1_20261028_130353/train/exported/model_5000"

# 校验模型绑定、导出哈希和物理模型，不推进仿真
python scripts/mujoco/run_sfyg_teacher.py \
  --bundle_dir "$SFYG_MJ_BUNDLE" --teacher_bundle "$TEACHER_BUNDLE" \
  --headless --check_only

# 先看8N：GUI运行，自动完成一次18秒施力/撤力过程
python scripts/mujoco/run_sfyg_teacher.py \
  --bundle_dir "$SFYG_MJ_BUNDLE" --teacher_bundle "$TEACHER_BUNDLE" \
  --force_n 8 --episodes 1 --trace \
  --output_dir "logs/mujoco/sfyg_teacher_8N_$(date +%Y%m%d_%H%M%S)"
```

复用现有导出目录即可，不需要重新训练或重新导出。teacher_bundle是包含`manifest.json/policy.pt`的目录，不是`.pt` checkpoint。
低层bundle是包含`deployment.json/policy.pt`的目录。两份`policy.pt`用途不同，不能相互替换。
窗口标签显示当前时间、实际施力幅值、teacher速度指令和实际vx；红箭头在负载COM显示世界坐标实际力方向，相机跟随机器人。
没有键盘调速：此版先复现训练的固定施力过程。关闭窗口会保存报告并标为未完成。
策略均在CPU确定性推理，不启动Isaac，不需要选择CUDA设备。

## 多力档与输出

```bash
for FORCE in 0 2 4 6 8; do
  python scripts/mujoco/run_sfyg_teacher.py \
    --bundle_dir "$SFYG_MJ_BUNDLE" --teacher_bundle "$TEACHER_BUNDLE" \
    --force_n "$FORCE" --episodes 3 --headless --trace \
    --output_dir "logs/mujoco/sfyg_teacher_${FORCE}N_$(date +%Y%m%d_%H%M%S)" || break
done
```

输出`interface.json`、`steps.jsonl`、`evaluation.json`，可选`trace.npz`。
评价沿用Isaac的保持段速度均值/RMSE、末段停稳、启动延迟、倾角与臂误差；原始高层动作和实际执行指令均保留。
trace包含teacher147维观测、low42维观测/10×42历史、raw9、command3、leg10。
所有报告标记`is_acceptance=false`，属于跨仿真诊断，不能代替多机承载验收。

倾角>35°、根高度<0.45m、非法body接触、数值异常、MuJoCo内部重置均终止；记录失败，不自动重置续跑伪造完整回合。
非正常结束CLI返回2。正常多回合只在前一个完整结束后重置，清空高/低层全部历史与延迟队列。
`--smoke_steps`仅用于开发短测（最多100步），报告不会标为完整回合。

可选MuJoCo录像仍可用`--video`；无桌面时加`MUJOCO_GL=egl`和`--headless`。
这与此前取消的Isaac录像修复无关，Isaac测试入口没有改动。

## 本地验证（2026-10-28）

- 新增8项真实TorchScript/MuJoCo检查：训练端147维历史/动作逐步一致、严格模型绑定、错误模型拒绝、固定臂、旋转施力、原PD延迟、复位、完整短时测试夹具和只读可视化。
- 加上原teacher、单机MuJoCo和源码一致性检查，共35项相关测试通过。
- 本地已有短训低层和两轮teacher导出完成真实CLI检查及0.4秒短测；截图已查看。
- 未收敛本地权重的长测正确记录跌倒并非零退出。正式5000轮模型仅在服务器，本地工程测试不证明其MuJoCo行走或停车效果。
