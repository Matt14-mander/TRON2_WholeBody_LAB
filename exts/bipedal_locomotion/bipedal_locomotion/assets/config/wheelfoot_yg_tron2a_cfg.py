import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
usd_path = os.path.join(REPO_ROOT, "robot_description/tron2/WFYG_TRON2A/usd/robot.usd")

WHEELFOOT_YG_TRON2A_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=usd_path,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=10000.0,
            max_angular_velocity=10000.0,
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
        pos=(0.0, 0.0, 0.9),
        joint_pos={
            # left leg
            "proximal_pitch_L_Joint": 0.0,
            "proximal_roll_L_Joint": 0.0,
            "proximal_yaw_L_Joint": 0.0,
            "knee_L_Joint": 0.0,
            "wheel_L_Joint": 0.0,
            # right leg
            "proximal_pitch_R_Joint": 0.0,
            "proximal_roll_R_Joint": 0.0,
            "proximal_yaw_R_Joint": 0.0,
            "knee_R_Joint": 0.0,
            "wheel_R_Joint": 0.0,
            # arm locked at zero
            "arm1_Joint": 0.0,
            "arm2_Joint": 0.0,
            "arm3_Joint": 0.0,
            "arm4_Joint": 0.0,
            "arm5_Joint": 0.0,
            "arm6_Joint": 0.0,
            # gripper opened to max stroke. URDF axis 对称反向：
            # gripper1 limit [0.0, 0.05]，gripper2 limit [-0.05, 0.0]。
            "gripper1_Joint": 0.05,
            "gripper2_Joint": -0.05,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": ImplicitActuatorCfg(
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
        ),
        "base_legs_light": ImplicitActuatorCfg(
            joint_names_expr=[
                "proximal_yaw_[RL]_Joint",
            ],
            armature=0.053923687,
            effort_limit=40.0,
            velocity_limit=14.66,
            stiffness=53.22,
            damping=3.39,
            friction=0.0,
        ),
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[
                "wheel_L_Joint",
                "wheel_R_Joint",
            ],
            effort_limit=20.0,
            velocity_limit=40.0,
            stiffness=0.0,
            damping=0.6,
            friction=0.0,
        ),
        # High-stiffness lock: arm/gripper held at init_state pose; not part of RL action space.
        # stiffness >> leg (~160 Nm/rad) so小扰动被迅速拉回零位。
        "arm_lock": ImplicitActuatorCfg(
            joint_names_expr=["arm[1-6]_Joint", "gripper[12]_Joint"],
            stiffness=800.0,
            damping=40.0,
            effort_limit=200.0,
            velocity_limit=10.0,
            friction=0.0,
        ),
    },
)
