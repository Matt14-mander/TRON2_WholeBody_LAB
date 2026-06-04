from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from bipedal_locomotion.assets.config.solefoot_yg_tron2a_cfg import SOLEFOOT_YG_TRON2A_CFG
from bipedal_locomotion.tasks.locomotion.robots.limx_solefoot_tron2a_env_cfg import (
    SF_TRON2A_BaseEnvCfg,
    SF_TRON2A_BaseEnvCfg_PLAY,
    SF_TRON2A_BlindFlatEnvCfg,
    SF_TRON2A_BlindFlatEnvCfg_PLAY,
    SF_TRON2A_BlindRoughEnvCfg,
    SF_TRON2A_BlindRoughEnvCfg_PLAY,
)

# 腿 + 躯干 link 白名单（SFYG）。
# 用于把"作用于全 robot body / 全关节"的随机化事件限定到非机械臂部分。
SFYG_LEG_TORSO_LINKS = [
    "base_Link",
    "proximal_pitch_[LR]_Link",
    "proximal_roll_[LR]_Link",
    "proximal_yaw_[LR]_Link",
    "knee_[LR]_Link",
    "ankle_pitch_[LR]_Link",
]

# 腿 + 躯干 关节白名单（SFYG）。与 joint_order_name 一致即可，排除机械臂关节。
SFYG_LEG_JOINTS = [
    "proximal_pitch_[LR]_Joint",
    "proximal_roll_[LR]_Joint",
    "proximal_yaw_[LR]_Joint",
    "knee_[LR]_Joint",
    "ankle_pitch_[LR]_Joint",
]


def _exclude_arm_from_events_and_rewards(self):
    """把基类中使用 body_names='.*' 或 joint_names='.*' 的项限定到非机械臂部分。

    纯仿真赛阶段这些 event 已被禁用（=None），不需要修改 params。
    """
    # 1. physics material 随机化：原 body_names='.*'，缩到腿+躯干
    if self.events.robot_physics_material is not None:
        self.events.robot_physics_material.params["asset_cfg"].body_names = SFYG_LEG_TORSO_LINKS
    # 2. actuator 增益随机化：原 joint_names='.*'，缩到腿部关节。
    #    机械臂用 arm_lock，必须避免它的 PD 增益被扰动，否则锁位会失效。
    if self.events.randomize_actuator_gains is not None:
        self.events.randomize_actuator_gains.params["asset_cfg"].joint_names = SFYG_LEG_JOINTS
    # 3. center of mass 随机化：基类未指定 body_names，默认匹配全 body；缩到腿+躯干。
    if self.events.robot_center_of_mass is not None:
        self.events.robot_center_of_mass.params["asset_cfg"].body_names = SFYG_LEG_TORSO_LINKS
    # 4. stand_still reward：原 joint_names='.*'，缩到腿部关节
    if self.rewards.stand_still is not None:
        self.rewards.stand_still.params["asset_cfg"].joint_names = SFYG_LEG_JOINTS
    # 5. reset_robot_joints：基类无 asset_cfg（默认全关节），
    #    会给 gripper（限位 0~0.05 m）±0.5 m 偏移、给 arm 关节大偏移触发 arm_lock 暴力回拉。
    #    注入 asset_cfg 缩到腿部关节，机械臂保持 default joint pos。
    if self.events.reset_robot_joints is not None:
        self.events.reset_robot_joints.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=SFYG_LEG_JOINTS)
    # 6. dof_pos_limits / dof_vel_limits reward：原 joint_names='.*'，缩到腿部关节，
    #    避免 reset 偏移引起的瞬时 penalty 干扰策略学习。
    if self.rewards.dof_pos_limits is not None:
        self.rewards.dof_pos_limits.params["asset_cfg"].joint_names = SFYG_LEG_JOINTS
    if self.rewards.dof_vel_limits is not None:
        self.rewards.dof_vel_limits.params["asset_cfg"].joint_names = SFYG_LEG_JOINTS
    # 7. feet_distance reward：SFYG 因为机械臂重量分布与 SF 不同，期望步距更窄，
    #    权重从 base 0.2 减半到 0.1，避免过强约束牵制 locomotion 学习。
    self.rewards.feet_distance.weight = 0.1


######################
# SFYG_TRON2A Base Environment
######################


@configclass
class SFYG_TRON2A_BaseEnvCfg(SF_TRON2A_BaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = SOLEFOOT_YG_TRON2A_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        _exclude_arm_from_events_and_rewards(self)


@configclass
class SFYG_TRON2A_BaseEnvCfg_PLAY(SF_TRON2A_BaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = SOLEFOOT_YG_TRON2A_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        _exclude_arm_from_events_and_rewards(self)


############################
# SFYG_TRON2A Blind Flat Environment
############################


@configclass
class SFYG_TRON2A_BlindFlatEnvCfg(SF_TRON2A_BlindFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = SOLEFOOT_YG_TRON2A_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        _exclude_arm_from_events_and_rewards(self)


@configclass
class SFYG_TRON2A_BlindFlatEnvCfg_PLAY(SF_TRON2A_BlindFlatEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = SOLEFOOT_YG_TRON2A_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        _exclude_arm_from_events_and_rewards(self)


############################
# SFYG_TRON2A Blind Rough Environment
############################


@configclass
class SFYG_TRON2A_BlindRoughEnvCfg(SF_TRON2A_BlindRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = SOLEFOOT_YG_TRON2A_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        _exclude_arm_from_events_and_rewards(self)
        # rough 地形下速度跟踪更难，权重相对 base (lin=1.0, ang=0.5) 提到 1.5×；
        # 转向单独再加一档（→ 1.5×base 之外再 ×1.5 ≈ base 2.25×），rough 上转向更难学。
        self.rewards.track_lin_vel_xy.weight = 1.5
        self.rewards.track_ang_vel_z.weight = 1.125
        # 命令分布：不要横移，x 负向速度减半
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        # push_robot 已在 base cfg 关闭（纯仿真赛）
        # rough 地形下加大期望抬脚高度防绊脚，base swing_height=0.20 -> 0.28
        # 并把 gait_reward 权重 0.75 -> 1.0，让 swing-height 跟踪更具引导性
        self.commands.gait_command.ranges.swing_height = (0.28, 0.28)
        self.rewards.gait_reward.weight = 1.0
        # 同比例放松膝盖位形约束（与 stand_still 偏离 default 惩罚有重叠，避免压制策略）：
        # knee_joint_orientation: -2.0 -> -0.7，orientation_exp_knee: 0.2 -> 0.1
        self.rewards.knee_joint_orientation.weight = -0.7
        self.rewards.orientation_exp_knee.weight = 0.1


@configclass
class SFYG_TRON2A_BlindRoughEnvCfg_PLAY(SF_TRON2A_BlindRoughEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = SOLEFOOT_YG_TRON2A_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        _exclude_arm_from_events_and_rewards(self)
        self.rewards.track_lin_vel_xy.weight = 1.5
        self.rewards.track_ang_vel_z.weight = 1.125
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        # PLAY 基类会 self.events.push_robot = None，这里无需再调扰动
        # 与训练 cfg 保持一致的抬脚高度与 gait 权重，方便回放对照
        self.commands.gait_command.ranges.swing_height = (0.28, 0.28)
        self.rewards.gait_reward.weight = 1.0
        self.rewards.knee_joint_orientation.weight = -0.7
        self.rewards.orientation_exp_knee.weight = 0.1
