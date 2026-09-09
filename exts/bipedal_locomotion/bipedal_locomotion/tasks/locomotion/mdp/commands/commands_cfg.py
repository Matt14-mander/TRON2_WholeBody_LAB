import math
from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass

from .gait_command import GaitCommand  # Import the GaitCommand class
from .wrench_sequence_command import WrenchSequenceCommand


@configclass
class UniformGaitCommandCfg(CommandTermCfg):
    """Configuration for the gait command generator."""

    class_type: type = GaitCommand  # Specify the class type for dynamic instantiation

    @configclass
    class Ranges:
        """Uniform distribution ranges for the gait parameters."""

        frequencies: tuple[float, float] = MISSING
        """Range for gait frequencies [Hz]."""
        offsets: tuple[float, float] = MISSING
        """Range for phase offsets [0-1]."""
        durations: tuple[float, float] = MISSING
        """Range for contact durations [0-1]."""
        swing_height: tuple[float, float] = MISSING
        """Range for contact durations [0-1]."""

    ranges: Ranges = MISSING
    """Distribution ranges for the gait parameters."""

    resampling_time_range: tuple[float, float] = MISSING
    """Time interval for resampling the gait (in seconds)."""


@configclass
class WrenchSequenceCommandCfg(CommandTermCfg):
    """Configuration for the smooth external-wrench sequence generator."""

    class_type: type = WrenchSequenceCommand
    asset_name: str = "robot"
    body_name: str = "base_Link"
    resampling_time_range: tuple[float, float] = (8.0, 12.0)
    prediction_times: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8)
    force_ranges: tuple[tuple[float, float], ...] = (
        (-80.0, 80.0),
        (-80.0, 80.0),
        (-120.0, 40.0),
    )
    torque_ranges: tuple[tuple[float, float], ...] = (
        (-30.0, 30.0),
        (-30.0, 30.0),
        (-20.0, 20.0),
    )
    beta_range: tuple[float, float] = (0.5, 3.0)
    endpoint_step_scale: float = 0.02
    force_observation_noise_std: tuple[float, float, float] = (2.0, 2.0, 2.0)
    torque_observation_noise_std: tuple[float, float, float] = (0.5, 0.5, 0.5)
    force_acceleration_gain: tuple[float, float, float] = (3.0, 3.0, 3.0)
    torque_acceleration_gain: tuple[float, float, float] = (0.5, 0.5, 0.5)
    unobserved_noise_std: float = 0.25
    unobserved_wrench_limits: tuple[float, float, float, float, float, float] = (
        40.0,
        40.0,
        40.0,
        10.0,
        10.0,
        10.0,
    )
