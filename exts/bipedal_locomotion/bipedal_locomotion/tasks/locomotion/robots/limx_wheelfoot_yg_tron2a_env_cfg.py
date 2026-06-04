from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from bipedal_locomotion.assets.config.wheelfoot_yg_tron2a_cfg import WHEELFOOT_YG_TRON2A_CFG
from bipedal_locomotion.tasks.locomotion.robots.limx_wheelfoot_tron2a_env_cfg import (
    WF_TRON2A_BaseEnvCfg,
    WF_TRON2A_BaseEnvCfg_PLAY,
    WF_TRON2A_BlindFlatEnvCfg,
    WF_TRON2A_BlindFlatEnvCfg_PLAY,
    WF_TRON2A_BlindRoughEnvCfg,
    WF_TRON2A_BlindRoughEnvCfg_PLAY,
)

# 腿 + 躯干 link 白名单（WFYG）。
# 用于把"作用于全 robot body / 全关节"的随机化事件限定到非机械臂部分。
WFYG_LEG_TORSO_LINKS = [
    "base_Link",
    "proximal_pitch_[LR]_Link",
    "proximal_roll_[LR]_Link",
    "proximal_yaw_[LR]_Link",
    "knee_[LR]_Link",
    "wheel_[LR]_Link",
]

# 腿 + wheel 关节白名单（WFYG）。
WFYG_LEG_JOINTS = [
    "proximal_pitch_[LR]_Joint",
    "proximal_roll_[LR]_Joint",
    "proximal_yaw_[LR]_Joint",
    "knee_[LR]_Joint",
    "wheel_[LR]_Joint",
]


def _exclude_arm_from_events_and_rewards(self):
    """把基类中使用 body_names='.*' / joint_names='.*' / 默认全 body 的项限定到非机械臂部分。

    WF base cfg 中对应项（cfg/WF_TRON2A/limx_base_env_cfg.py）：
    - events.robot_physics_material (line 289): body_names='.*'
    - events.randomize_actuator_gains (line 300): joint_names='.*'
    - events.robot_center_of_mass (line 310): 默认全 body
    - events.radomize_rigid_body_mass_inertia (line 280, 注意拼写 typo)：默认全 body
    rewards.stand_still (line 378) 在 WF 中无 params，无须覆写。

    纯仿真赛阶段这些 event 已被禁用（=None），不需要修改 params。
    """
    if self.events.robot_physics_material is not None:
        self.events.robot_physics_material.params["asset_cfg"].body_names = WFYG_LEG_TORSO_LINKS
    if self.events.randomize_actuator_gains is not None:
        # 机械臂用 arm_lock，避免它的 PD 增益被扰动，否则锁位会失效。
        self.events.randomize_actuator_gains.params["asset_cfg"].joint_names = WFYG_LEG_JOINTS
    if self.events.robot_center_of_mass is not None:
        self.events.robot_center_of_mass.params["asset_cfg"].body_names = WFYG_LEG_TORSO_LINKS
    if self.events.radomize_rigid_body_mass_inertia is not None:
        # 注意：保留基类拼写 typo `radomize_*`（与基类属性名匹配）。
        self.events.radomize_rigid_body_mass_inertia.params["asset_cfg"].body_names = WFYG_LEG_TORSO_LINKS
    # 5. reset_robot_joints：基类无 asset_cfg（默认全关节），
    #    会给 gripper（限位 0~0.05 m）±0.2 m 偏移、给 arm 关节大偏移触发 arm_lock 暴力回拉。
    #    注入 asset_cfg 缩到腿部关节，机械臂保持 default joint pos。
    if self.events.reset_robot_joints is not None:
        self.events.reset_robot_joints.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=WFYG_LEG_JOINTS)
    # 6. non_wheel_pos_limits reward：原 joint_names='(?!wheel_).*' 仍匹配 arm/gripper，
    #    缩到腿+wheel 关节，避免 reset 偏移引起的瞬时 penalty 干扰策略学习。
    if self.rewards.non_wheel_pos_limits is not None:
        self.rewards.non_wheel_pos_limits.params["asset_cfg"].joint_names = WFYG_LEG_JOINTS


######################
# WFYG_TRON2A Base Environment
######################


@configclass
class WFYG_TRON2A_BaseEnvCfg(WF_TRON2A_BaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = WHEELFOOT_YG_TRON2A_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        _exclude_arm_from_events_and_rewards(self)


@configclass
class WFYG_TRON2A_BaseEnvCfg_PLAY(WF_TRON2A_BaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = WHEELFOOT_YG_TRON2A_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        _exclude_arm_from_events_and_rewards(self)


############################
# WFYG_TRON2A Blind Flat Environment
############################


@configclass
class WFYG_TRON2A_BlindFlatEnvCfg(WF_TRON2A_BlindFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = WHEELFOOT_YG_TRON2A_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        _exclude_arm_from_events_and_rewards(self)


@configclass
class WFYG_TRON2A_BlindFlatEnvCfg_PLAY(WF_TRON2A_BlindFlatEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = WHEELFOOT_YG_TRON2A_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        _exclude_arm_from_events_and_rewards(self)


############################
# WFYG_TRON2A Blind Rough Environment
############################


@configclass
class WFYG_TRON2A_BlindRoughEnvCfg(WF_TRON2A_BlindRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = WHEELFOOT_YG_TRON2A_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        _exclude_arm_from_events_and_rewards(self)


@configclass
class WFYG_TRON2A_BlindRoughEnvCfg_PLAY(WF_TRON2A_BlindRoughEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = WHEELFOOT_YG_TRON2A_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        _exclude_arm_from_events_and_rewards(self)
