"""Fixed-start checks for the Piper piper_control data path."""

from __future__ import annotations

import numpy as np

from .schema import (
    QposTolerance,
    STANDARD_START_QPOS,
    StartGuardMode,
    ZONE_ARM_TOLERANCE_RAD,
    ZONE_GRIPPER_OPEN_MIN_M,
    as_qpos,
)


def qpos_diff(current, target) -> np.ndarray:
    return np.abs(as_qpos(current, label="current qpos") - as_qpos(target, label="target qpos"))


def start_pose_guard(
    current,
    target=STANDARD_START_QPOS,
    *,
    mode: StartGuardMode = "zone",
    tolerance: QposTolerance = QposTolerance(),
) -> bool:
    current_qpos = as_qpos(current, label="current qpos")
    diff = qpos_diff(current_qpos, target)
    if mode == "strict":
        return bool(np.all(diff[:6] <= tolerance.arm_rad) and diff[6] <= tolerance.gripper_m)
    if mode == "zone":
        arm_ok = bool(np.all(diff[:6] <= np.asarray(ZONE_ARM_TOLERANCE_RAD, dtype=np.float32)))
        gripper_ok = bool(float(current_qpos[6]) >= ZONE_GRIPPER_OPEN_MIN_M)
        return arm_ok and gripper_ok
    raise ValueError(f"Unknown start guard mode: {mode!r}")


def describe_guard_result(
    current,
    target=STANDARD_START_QPOS,
    *,
    mode: StartGuardMode = "zone",
    tolerance: QposTolerance = QposTolerance(),
) -> str:
    current_qpos = as_qpos(current, label="current qpos")
    diff = qpos_diff(current_qpos, target)
    if mode == "strict":
        return (
            f"arm max diff={float(np.max(diff[:6])):.5f}/{tolerance.arm_rad:.5f}, "
            f"gripper diff={float(diff[6]):.5f}/{tolerance.gripper_m:.5f}"
        )

    per_joint = " ".join(
        f"j{i + 1}={float(diff[i]):.4f}/{ZONE_ARM_TOLERANCE_RAD[i]:.4f}"
        for i in range(6)
    )
    gripper = f"gripper={float(current_qpos[6]):.5f} m (need >= {ZONE_GRIPPER_OPEN_MIN_M})"
    return f"zone arm: {per_joint}, {gripper}"
