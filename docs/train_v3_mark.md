# 训练形态、零外力
{
  "interrupted": false,
  "robot": "training",
  "wrench": "zero",
  "cases": {
    "stand": {
      "episodes": 3,
      "completed": 3,
      "passed": 3
    },
    "walk": {
      "episodes": 3,
      "completed": 3,
      "passed": 0
    },
    "slow": {
      "episodes": 3,
      "completed": 3,
      "passed": 0
    },
    "forward": {
      "episodes": 3,
      "completed": 3,
      "passed": 0
    },
    "left": {
      "episodes": 3,
      "completed": 3,
      "passed": 0
    },
    "right": {
      "episodes": 3,
      "completed": 3,
      "passed": 0
    },
    "yaw_left": {
      "episodes": 3,
      "completed": 3,
      "passed": 0
    },
    "yaw_right": {
      "episodes": 3,
      "completed": 3,
      "passed": 0
    },
    "start_stop": {
      "episodes": 3,
      "completed": 3,
      "passed": 0
    },
    "stop_start": {
      "episodes": 3,
      "completed": 3,
      "passed": 0
    }
  },
  "all_passed": false
}

# 完整 SFYG
{
  "interrupted": false,
  "robot": "wholebody",
  "wrench": "zero",
  "cases": {
    "stand": {
      "episodes": 3,
      "completed": 3,
      "passed": 0
    },
    "walk": {
      "episodes": 3,
      "completed": 3,
      "passed": 0
    },
    "forward": {
      "episodes": 3,
      "completed": 3,
      "passed": 0
    },
    "left": {
      "episodes": 3,
      "completed": 3,
      "passed": 0
    },
    "right": {
      "episodes": 3,
      "completed": 3,
      "passed": 0
    },
    "yaw_left": {
      "episodes": 3,
      "completed": 3,
      "passed": 0
    },
    "yaw_right": {
      "episodes": 3,
      "completed": 3,
      "passed": 0
    },
    "start_stop": {
      "episodes": 3,
      "completed": 3,
      "passed": 0
    },
    "stop_start": {
      "episodes": 3,
      "completed": 3,
      "passed": 0
    }
  },
  "all_passed": false
}


| 指令 | 实际平均速度 | 判断 |
|---|---|---|
| 前进 `vx=0.15` | `vx=0.1532` | 平均前进速度准确 |
| 后退 `vx=-0.25` | `vx=-0.2084` | 能后退，但速度不足 |
| 慢退 `vx=-0.05` | `vx=-0.0200` | 低速跟踪不足 |
| 左移 `vy=0.10` | `vy=0.0643` | 方向正确，速度不足 |
| 右移 `vy=-0.10` | `vy=-0.0091` | **平均右移速度接近零，明显不足** |

| 检查项 | 要验证什么 | 当前状态 |
|---|---|---|
| 基础速度能力 | 覆盖 teacher 将使用的前后、侧向、转向速度，包含持续低速和组合指令 | 部分通过，右移明显不足 |
| 启停与变速 | 多种速度之间切换、停车、再次起步；采用 teacher 相同的指令更新频率和限速 | 已有单一目标速度测试，覆盖不足 |
| 完整机械臂＋负载 | 使用 **teacher 实际的臂姿、PD、负载质量和安装位置**，验证站立与行走 | 尚未完整验证 |
| 臂部运动 | 覆盖 teacher 输出的臂目标范围、运动速度及组合动作 | 小幅测试尚不能覆盖 |
| 实际交互力 | 在负载／末端施加多方向、不同持续时间的力和力矩，验证跟随过程中不倒 | 尚未系统验证 |
| 低层输入契约 | 尤其是预测 wrench 的来源、坐标系、作用点、归一化和时序 | **需要先明确** |
| 随机化与恢复 | 多种初始状态、不同步态阶段受扰后能恢复，而非只在固定复位状态稳定 | 当前3个 seed 的名义测试不足 |