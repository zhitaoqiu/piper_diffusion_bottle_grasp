"""Piper robot configuration dataclass."""

from dataclasses import dataclass, field
from pathlib import Path

from lerobot.robots.config import RobotConfig


@dataclass
class PiperRobotConfig(RobotConfig):
    can_port: str = "can0"
    gripper_exist: bool = True
    velocity_pct: int = 50
    gripper_effort: int = 1000
    enable_timeout: float = 10.0
    joint_limit_rad: float = 3.14
    disable_torque_on_disconnect: bool = True
    # id and calibration_dir inherited from RobotConfig
    id: str | None = None
    calibration_dir: Path | None = None
