"""SFYG TRON2A asset configuration for whole-body control."""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from bipedal_locomotion.actuators import DelayedImplicitActuatorCfg


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
usd_path = os.path.join(REPO_ROOT, "robot_description/tron2/SFYG_TRON2A/usd/robot.usd")


SOLEFOOT_YG_TRON2A_WHOLEBODY_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=usd_path,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            disable_gravity=False,
            retain_accelerations=True,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
        ),
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.85),
        joint_pos={
            "proximal_pitch_[LR]_Joint": 0.0,
            "proximal_roll_[LR]_Joint": 0.0,
            "proximal_yaw_L_Joint": -3.14159,
            "proximal_yaw_R_Joint": 3.14159,
            "knee_[LR]_Joint": 0.0,
            "ankle_pitch_[LR]_Joint": 0.0,
            # arm2=0 and arm3=0 are boundaries in the supplied URDF.
            "arm1_Joint": 0.0,
            "arm2_Joint": 1.5707963,
            "arm3_Joint": -1.4835299,
            "arm4_Joint": 0.0,
            "arm5_Joint": 0.0,
            "arm6_Joint": 0.0,
            "gripper1_Joint": 0.05,
            "gripper2_Joint": -0.05,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        "base_legs": DelayedImplicitActuatorCfg(
            joint_names_expr=[
                "proximal_pitch_[RL]_Joint",
                "proximal_roll_[RL]_Joint",
                "knee_[RL]_Joint",
            ],
            armature=0.161777558,
            effort_limit=140.0,
            velocity_limit=12.57,
            stiffness=159.67,
            damping=10.16,
            friction=0.0,
            min_delay=0,
            max_delay=4,
        ),
        "base_legs_light": DelayedImplicitActuatorCfg(
            joint_names_expr=[
                "proximal_yaw_[RL]_Joint",
                "ankle_pitch_[RL]_Joint",
            ],
            armature=0.053923687,
            effort_limit=40.0,
            velocity_limit=14.66,
            stiffness=53.22,
            damping=3.39,
            friction=0.0,
            min_delay=0,
            max_delay=4,
        ),
        # Commissioning gains only. OCS2 supplies desired position, velocity,
        # and feed-forward effort through the deployment bridge.
        "arm_mpc": ImplicitActuatorCfg(
            joint_names_expr=["arm[1-6]_Joint"],
            stiffness=60.0,
            damping=6.0,
            effort_limit=100.0,
            velocity_limit=5.0,
            friction=0.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["gripper[12]_Joint"],
            stiffness=100.0,
            damping=5.0,
            effort_limit=10.0,
            velocity_limit=3.0,
            friction=0.0,
        ),
    },
)
