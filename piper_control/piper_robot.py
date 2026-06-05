"""PiperRobot — LeRobot-standard Robot implementation for Piper robotic arm."""

import logging
import sys
import time
from functools import cached_property

from lerobot.types import RobotAction, RobotObservation

import os as _os
_sdk_path = _os.environ.get("PIPER_SDK_PATH")
if _sdk_path is None:
    _sdk_path = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "piper_sdk_py_driver")
if _sdk_path not in sys.path:
    sys.path.insert(0, _sdk_path)
del _os

from piper_sdk_py_driver.sdk_adapter import PiperSdkAdapter, JointState, EndPose, ArmStatus

from .config_piper import PiperRobotConfig

from lerobot.robots.robot import Robot

logger = logging.getLogger(__name__)

JOINT_KEYS = [f"j{i}.pos" for i in range(1, 7)]
GRIPPER_KEY = "gripper.pos"
ALL_ACTION_KEYS = JOINT_KEYS + [GRIPPER_KEY]

PIPER_GRIPPER_MAX_M = 0.101


class PiperRobot(Robot):
    """LeRobot-compatible Piper arm wrapper.

    Standard interface
    ------------------
    connect(calibrate=True)  – open CAN port, enable motors
    disconnect()             – disable motors, close CAN port
    get_observation()        – dict like {"j1.pos": 0.12, ..., "gripper.pos": 0.10}
    send_action(action)      – dict in, dict out (returns clipped action actually sent)
    observation_features     – dict describing observation structure
    action_features          – dict describing action structure
    is_connected             – True when CAN port is open

    Piper-specific
    --------------
    enable(blocking=True)    – enable motors without opening CAN
    disable()                – disable motors without closing CAN
    is_enabled               – True when motors are powered
    get_joint_positions()    – convenience: returns list[float] [j1..j6, grip]
    set_joint_positions(positions, velocity_pct, gripper_effort) – convenience wrapper
    get_joint_state()        – JointState dataclass
    get_end_pose()           – EndPose dataclass
    get_arm_status()         – ArmStatus dataclass
    """

    config_class = PiperRobotConfig
    name = "piper"

    # Instance-level attributes settable between send_action calls
    velocity_pct: int
    gripper_effort: int

    def __init__(self, config: PiperRobotConfig | None = None, **kwargs):
        if config is None:
            config = PiperRobotConfig(**kwargs)
        super().__init__(config)
        self.config: PiperRobotConfig = config

        self.velocity_pct = config.velocity_pct
        self.gripper_effort = config.gripper_effort

        self._adapter = PiperSdkAdapter(
            can_port=config.can_port,
            gripper_exist=config.gripper_exist,
            enable_timeout=config.enable_timeout,
        )
        self.joint_limit = config.joint_limit_rad
        self._disable_torque_on_disconnect = config.disable_torque_on_disconnect

    # ---- observation / action features ----

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {k: float for k in ALL_ACTION_KEYS}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {k: float for k in ALL_ACTION_KEYS}

    # ---- lifecycle ----

    @property
    def is_connected(self) -> bool:
        return self._adapter.is_connected()

    def connect(self, calibrate: bool = True) -> None:
        """Open CAN port and enable motors.

        Args:
            calibrate: Ignored — Piper uses absolute encoders, always calibrated.
        """
        if self.is_connected:
            logger.info(f"{self} already connected.")
            return
        self._adapter.connect()
        time.sleep(0.3)
        self.configure()
        logger.info(f"{self} connected on {self.config.can_port}.")

    def disconnect(self) -> None:
        """Disconnect safely; preserve the SDK session when torque must stay on."""
        if not self.is_connected:
            return
        if not self._disable_torque_on_disconnect:
            # Piper's low-level SDK disconnect disables the arm. Keep the SDK
            # session alive when callers explicitly request torque retention;
            # deploy/reset processes exit immediately after this path.
            logger.info(f"{self} keeps SDK session open to preserve arm torque.")
            return
        if self.is_enabled:
            self.disable()
        self._adapter.disconnect()
        logger.info(f"{self} disconnected.")

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        self.enable(blocking=True)

    # ---- state reading ----

    def get_observation(self) -> RobotObservation:
        """Return joint positions as a flat dict.

        Keys: "j1.pos" ... "j6.pos", "gripper.pos"
        """
        js = self._adapter.read_joint_state()
        obs: RobotObservation = {}
        for i, key in enumerate(JOINT_KEYS):
            obs[key] = float(js.position[i])
        obs[GRIPPER_KEY] = float(js.position[6])
        return obs

    # ---- action ----

    def send_action(self, action: RobotAction) -> RobotAction:
        """Send a joint position command.

        Args:
            action: dict with keys "j1.pos".."j6.pos", "gripper.pos".

        Returns:
            The action actually sent (after safety clamping).
        """
        positions = [0.0] * 7
        for i, key in enumerate(JOINT_KEYS):
            if key not in action:
                raise KeyError(f"send_action missing required key: {key!r}")
            positions[i] = float(action[key])
        if GRIPPER_KEY not in action:
            raise KeyError(f"send_action missing required key: {GRIPPER_KEY!r}")
        positions[6] = float(action[GRIPPER_KEY])

        # Safety clamp arm joints
        for i in range(6):
            if abs(positions[i]) > self.joint_limit:
                positions[i] = max(-self.joint_limit, min(self.joint_limit, positions[i]))

        # Clamp gripper
        positions[6] = max(0.0, min(positions[6], PIPER_GRIPPER_MAX_M))

        self._adapter.send_joint_positions(
            positions,
            velocity_percent=self.velocity_pct,
            gripper_effort=self.gripper_effort,
        )

        # Return clamped action
        sent: RobotAction = {}
        for i, key in enumerate(JOINT_KEYS):
            sent[key] = positions[i]
        sent[GRIPPER_KEY] = positions[6]
        return sent

    # ---- Piper-specific: enable / disable ----

    def enable(self, blocking: bool = True) -> bool:
        return self._adapter.enable(blocking=blocking)

    def disable(self) -> None:
        self._adapter.disable()

    @property
    def is_enabled(self) -> bool:
        return self._adapter.is_enabled

    @property
    def is_ok(self) -> bool:
        return self._adapter.is_ok()

    # ---- Piper-specific: convenience methods ----

    def get_joint_positions(self):
        """Return [j1..j6, gripper_m] as a list of floats. Convenience wrapper."""
        obs = self.get_observation()
        return [obs[k] for k in ALL_ACTION_KEYS]

    def set_joint_positions(self, positions, velocity_pct=30, gripper_effort=1000):
        """Send a list [j1..j6, gripper_m]. Convenience wrapper around send_action."""
        saved_v = self.velocity_pct
        saved_e = self.gripper_effort
        self.velocity_pct = velocity_pct
        self.gripper_effort = gripper_effort
        try:
            action = {k: float(p) for k, p in zip(ALL_ACTION_KEYS, positions)}
            return self.send_action(action)
        finally:
            self.velocity_pct = saved_v
            self.gripper_effort = saved_e

    # ---- Piper-specific: detailed state access ----

    def get_joint_state(self) -> JointState:
        return self._adapter.read_joint_state()

    def get_end_pose(self) -> EndPose:
        return self._adapter.read_end_pose()

    def get_arm_status(self) -> ArmStatus:
        return self._adapter.read_arm_status()
