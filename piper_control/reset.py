"""Explicit reset interpolation helpers for piper_control fixed-start runs."""

from __future__ import annotations

import math
import time

import numpy as np

from .schema import DEFAULT_MAX_DELTA_RAD, STANDARD_START_QPOS, as_qpos

DEFAULT_MAX_GRIPPER_STEP_M = 0.004


def interpolate_qpos_path(
    start,
    target,
    *,
    max_arm_step=DEFAULT_MAX_DELTA_RAD,
    max_gripper_step_m: float = DEFAULT_MAX_GRIPPER_STEP_M,
) -> list[np.ndarray]:
    start_qpos = as_qpos(start, label="reset start qpos")
    target_qpos = as_qpos(target, label="reset target qpos")
    max_arm_step = np.asarray(max_arm_step, dtype=np.float32).reshape(6)
    if np.any(max_arm_step <= 0) or max_gripper_step_m <= 0:
        raise ValueError("reset interpolation steps must be positive.")
    arm_steps = np.max(np.abs(target_qpos[:6] - start_qpos[:6]) / max_arm_step)
    gripper_steps = abs(float(target_qpos[6] - start_qpos[6])) / max_gripper_step_m
    n_steps = max(1, math.ceil(max(float(arm_steps), gripper_steps)))
    return [
        start_qpos + (target_qpos - start_qpos) * (step / n_steps)
        for step in range(1, n_steps + 1)
    ]


def reset_to_standard_start(
    bus,
    target=STANDARD_START_QPOS,
    *,
    confirmed: bool = False,
    hz: float = 30.0,
    velocity_pct: int = 25,
) -> np.ndarray:
    if not confirmed:
        raise PermissionError("reset_to_standard_start requires explicit operator confirmation.")
    if hz <= 0:
        raise ValueError("reset hz must be positive.")
    current = bus.read_qpos()
    for qpos in interpolate_qpos_path(current, target):
        bus.write_qpos(qpos, velocity_pct=velocity_pct)
        time.sleep(1.0 / hz)
    return bus.read_qpos()
