"""Paper-aligned WFYG whole-body locomotion environments.

The locomotion policy controls the original ten leg/wheel actions. The arm is
an independently actuated subsystem reserved for OCS2, while training uses a
smooth external-wrench sequence generator in place of running MPC per env.
"""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from bipedal_locomotion.assets.config.wheelfoot_yg_tron2a_wholebody_cfg import (
    WHEELFOOT_YG_TRON2A_WHOLEBODY_CFG,
)
from bipedal_locomotion.tasks.locomotion import mdp
from bipedal_locomotion.tasks.locomotion.cfg.WF_TRON2A.limx_base_env_cfg import (
    CommandsCfg,
    ObservarionsCfg,
    RewardsCfg,
)
from bipedal_locomotion.tasks.locomotion.robots.limx_wheelfoot_yg_tron2a_env_cfg import (
    WFYG_LEG_JOINTS,
    WFYG_TRON2A_BlindFlatEnvCfg,
    WFYG_TRON2A_BlindFlatEnvCfg_PLAY,
    WFYG_TRON2A_BlindRoughEnvCfg,
    WFYG_TRON2A_BlindRoughEnvCfg_PLAY,
)


ARM_JOINTS = ["arm[1-6]_Joint"]


@configclass
class WholeBodyCommandsCfg(CommandsCfg):
    arm_wrench = mdp.WrenchSequenceCommandCfg()


@configclass
class WholeBodyObservationsCfg(ObservarionsCfg):
    @configclass
    class PolicyCfg(ObservarionsCfg.PolicyCfg):
        wrench_prediction = ObsTerm(
            func=mdp.normalized_wrench_prediction,
            params={"command_name": "arm_wrench"},
            clip=(-10.0, 10.0),
        )

    @configclass
    class CriticCfg(ObservarionsCfg.CriticCfg):
        arm_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS)},
        )
        arm_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS)},
            scale=0.1,
        )
        clean_wrench_prediction = ObsTerm(
            func=mdp.normalized_wrench_prediction,
            params={"command_name": "arm_wrench", "clean": True},
            clip=(-10.0, 10.0),
        )
        current_external_wrench = ObsTerm(
            func=mdp.normalized_current_external_wrench,
            params={"command_name": "arm_wrench"},
            clip=(-10.0, 10.0),
        )

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    commands: ObservarionsCfg.CommandsObsCfg = ObservarionsCfg.CommandsObsCfg()
    # Keep the existing proprioceptive history encoder independent of the
    # explicit wrench sequence. A dedicated wrench RNN belongs to stage 3.
    obsHistory: ObservarionsCfg.HistoryObsCfg = ObservarionsCfg.HistoryObsCfg()


@configclass
class WholeBodyRewardsCfg(RewardsCfg):
    # Separate base and arm regularization so unlocking the arm does not change
    # the scale of legacy locomotion rewards implicitly.
    joint_torque_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-4e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=WFYG_LEG_JOINTS)},
    )
    joint_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=WFYG_LEG_JOINTS)},
    )
    joint_power_l1 = RewTerm(
        func=mdp.joint_powers_l1,
        weight=-1e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=WFYG_LEG_JOINTS)},
    )
    vel_non_wheel_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.004,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=WFYG_LEG_JOINTS)},
    )
    arm_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS)},
    )
    arm_joint_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS)},
    )
    arm_joint_torque_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS)},
    )


class _WholeBodyMixin:
    def _configure_whole_body(self):
        self.scene.robot = WHEELFOOT_YG_TRON2A_WHOLEBODY_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    def _disable_training_wrench_for_play(self):
        zero_ranges = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
        self.commands.arm_wrench.force_ranges = zero_ranges
        self.commands.arm_wrench.torque_ranges = zero_ranges
        self.commands.arm_wrench.force_observation_noise_std = (0.0, 0.0, 0.0)
        self.commands.arm_wrench.torque_observation_noise_std = (0.0, 0.0, 0.0)
        self.commands.arm_wrench.force_acceleration_gain = (0.0, 0.0, 0.0)
        self.commands.arm_wrench.torque_acceleration_gain = (0.0, 0.0, 0.0)
        self.commands.arm_wrench.unobserved_noise_std = 0.0
        self.commands.arm_wrench.unobserved_wrench_limits = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@configclass
class WFYG_TRON2A_WholeBodyFlatEnvCfg(_WholeBodyMixin, WFYG_TRON2A_BlindFlatEnvCfg):
    commands: WholeBodyCommandsCfg = WholeBodyCommandsCfg()
    observations: WholeBodyObservationsCfg = WholeBodyObservationsCfg()
    rewards: WholeBodyRewardsCfg = WholeBodyRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self._configure_whole_body()


@configclass
class WFYG_TRON2A_WholeBodyFlatEnvCfg_PLAY(_WholeBodyMixin, WFYG_TRON2A_BlindFlatEnvCfg_PLAY):
    commands: WholeBodyCommandsCfg = WholeBodyCommandsCfg()
    observations: WholeBodyObservationsCfg = WholeBodyObservationsCfg()
    rewards: WholeBodyRewardsCfg = WholeBodyRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self._configure_whole_body()
        self._disable_training_wrench_for_play()


@configclass
class WFYG_TRON2A_WholeBodyRoughEnvCfg(_WholeBodyMixin, WFYG_TRON2A_BlindRoughEnvCfg):
    commands: WholeBodyCommandsCfg = WholeBodyCommandsCfg()
    observations: WholeBodyObservationsCfg = WholeBodyObservationsCfg()
    rewards: WholeBodyRewardsCfg = WholeBodyRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self._configure_whole_body()


@configclass
class WFYG_TRON2A_WholeBodyRoughEnvCfg_PLAY(_WholeBodyMixin, WFYG_TRON2A_BlindRoughEnvCfg_PLAY):
    commands: WholeBodyCommandsCfg = WholeBodyCommandsCfg()
    observations: WholeBodyObservationsCfg = WholeBodyObservationsCfg()
    rewards: WholeBodyRewardsCfg = WholeBodyRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self._configure_whole_body()
        self._disable_training_wrench_for_play()
