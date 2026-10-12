"""Configure nominal robot resets without changing rendering or actuator timing."""

from copy import deepcopy


def configure_nominal_ocs2_reset(env_cfg, all_robot_joints):
    """Reset root and every joint to asset defaults with zero random offsets.

    ``all_robot_joints`` is a fresh SceneEntityCfg("robot") with no joint-name
    filter. SFYG normally filters its reset event to legs only, which is useful
    during training but insufficient for resetting an OCS2 arm episode.
    Observation noise and actuator delays keep their task-configured values.
    """
    events = getattr(env_cfg, "events", None)
    base = getattr(events, "reset_robot_base", None)
    joints = getattr(events, "reset_robot_joints", None)
    if base is None or joints is None:
        raise RuntimeError("The selected OCS2 task must define base and joint reset events.")
    # Validate before changing either event, and avoid mutating inherited/shared
    # params. Do not modify the task's asset default pose or joint positions.
    for name in ("pose_range", "velocity_range"):
        if not isinstance(base.params.get(name), dict):
            raise ValueError(f"OCS2 base reset requires a {name} dictionary.")
    for name in ("position_range", "velocity_range"):
        if name not in joints.params:
            raise ValueError(f"OCS2 joint reset requires {name}.")
    base = deepcopy(base)
    joints = deepcopy(joints)
    for name in ("pose_range", "velocity_range"):
        base.params[name] = {axis: (0.0, 0.0) for axis in base.params[name]}
    joints.params["position_range"] = (0.0, 0.0)
    joints.params["velocity_range"] = (0.0, 0.0)
    joints.params["asset_cfg"] = deepcopy(all_robot_joints)
    events.reset_robot_base = base
    events.reset_robot_joints = joints
