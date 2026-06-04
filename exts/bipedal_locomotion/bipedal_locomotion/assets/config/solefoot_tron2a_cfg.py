import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from bipedal_locomotion.actuators import DelayedImplicitActuatorCfg

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
usd_path = os.path.join(REPO_ROOT, "robot_description/tron2/SF_TRON2A/usd/robot.usd")

SOLEFOOT_TRON2A_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=usd_path,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.85),
        joint_pos={
            # left leg
            "proximal_pitch_L_Joint": 0.0,
            "proximal_roll_L_Joint": 0.0,
            "proximal_yaw_L_Joint": 0.0,
            "knee_L_Joint": 0.0,
            "ankle_pitch_L_Joint": 0.0,
            # right leg
            "proximal_pitch_R_Joint": 0.0,
            "proximal_roll_R_Joint": 0.0,
            "proximal_yaw_R_Joint": 0.0,
            "knee_R_Joint": 0.0,
            "ankle_pitch_R_Joint": 0.0,
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
            min_delay=0,  # physics time steps (min: 5.0*0=0.0ms)
            max_delay=4,  # physics time steps (max: 5.0*4=20.0ms)
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
            min_delay=0,  # physics time steps (min: 5.0*0=0.0ms)
            max_delay=4,  # physics time steps (max: 5.0*4=20.0ms)
        ),
    },
)
