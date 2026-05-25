#!/usr/bin/env python3
"""Move Piper to the adapter-v2 fixed start pose with the gripper open."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from adapter_v2.piper_bus import PiperMotorsBusV2, PiperMotorsBusV2Config
from adapter_v2.reset import interpolate_qpos_path
from adapter_v2.schema import GRIPPER_OPEN_M, PIPER_GRIPPER_MAX_M, STANDARD_START_QPOS, as_qpos


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reset Piper to adapter-v2 start pose and open the gripper."
    )
    parser.add_argument("--can-port", default="can0")
    parser.add_argument(
        "--start-pose-file",
        type=Path,
        default=PROJECT_ROOT / "config" / "adapter_v2_start_pose.json",
    )
    parser.add_argument("--velocity-pct", type=int, default=20)
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument("--max-arm-step", type=float, default=0.02)
    parser.add_argument("--max-gripper-step", type=float, default=0.003)
    parser.add_argument(
        "--gripper-open",
        type=float,
        default=None,
        help="Override final gripper opening in meters. Defaults to adapter-v2 open pose.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip the confirmation prompt.")
    parser.add_argument(
        "--disable-torque-on-exit",
        action="store_true",
        help="Disable torque when the script exits. Default keeps torque on.",
    )
    args = parser.parse_args()

    if args.velocity_pct <= 0:
        parser.error("--velocity-pct must be > 0.")
    if args.hz <= 0:
        parser.error("--hz must be > 0.")
    if args.max_arm_step <= 0 or args.max_gripper_step <= 0:
        parser.error("--max-arm-step and --max-gripper-step must be > 0.")
    if args.gripper_open is not None and not 0 <= args.gripper_open <= PIPER_GRIPPER_MAX_M:
        parser.error(f"--gripper-open must be in [0, {PIPER_GRIPPER_MAX_M}].")
    return args


def fmt_vec(values, precision: int = 4) -> str:
    return "[" + ", ".join(f"{float(value):.{precision}f}" for value in values) + "]"


def load_target_qpos(path: Path, gripper_open: float | None) -> np.ndarray:
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        values = data.get("qpos", data.get("joint_positions"))
        if values is None:
            raise KeyError(f"{path} must contain 'qpos' or 'joint_positions'.")
        target = as_qpos(values, label=f"start pose file {path}").copy()
    else:
        print(f"[WARN] start pose file not found: {path}. Using built-in adapter-v2 start pose.")
        target = STANDARD_START_QPOS.copy()

    if gripper_open is None:
        target[6] = max(float(target[6]), GRIPPER_OPEN_M)
    else:
        target[6] = float(gripper_open)
    target[6] = min(float(target[6]), PIPER_GRIPPER_MAX_M)
    return target


def confirm(args: argparse.Namespace, target: np.ndarray) -> None:
    if args.dry_run or args.yes:
        return
    print()
    print("This will move the real Piper arm to adapter-v2 start pose:")
    print(f"  target: {fmt_vec(target)}")
    print(f"  can   : {args.can_port}")
    print("Type YES to continue: ", end="", flush=True)
    answer = input().strip()
    if answer != "YES":
        raise SystemExit("Aborted.")


def main() -> int:
    args = parse_args()
    target = load_target_qpos(args.start_pose_file, args.gripper_open)
    confirm(args, target)

    bus = PiperMotorsBusV2(
        PiperMotorsBusV2Config(
            can_port=args.can_port,
            velocity_pct=args.velocity_pct,
            disable_torque_on_disconnect=args.disable_torque_on_exit,
        )
    )

    print("=" * 72)
    print("Adapter-v2 start reset")
    print(f"  start pose : {args.start_pose_file}")
    print(f"  target     : {fmt_vec(target)}")
    print(f"  velocity   : {args.velocity_pct}%")
    print(f"  dry run    : {args.dry_run}")
    print("=" * 72)

    try:
        bus.connect()
        current = bus.read_qpos()
        path = interpolate_qpos_path(
            current,
            target,
            max_arm_step=np.full(6, args.max_arm_step, dtype=np.float32),
            max_gripper_step_m=args.max_gripper_step,
        )
        print(f"  current    : {fmt_vec(current)}")
        print(f"  waypoints  : {len(path)}")

        if args.dry_run:
            print("  Dry run only. No command sent.")
            return 0

        for index, qpos in enumerate(path, start=1):
            bus.write_qpos(qpos, velocity_pct=args.velocity_pct)
            if index == 1 or index == len(path) or index % 10 == 0:
                print(f"  step {index:03d}/{len(path):03d}: {fmt_vec(qpos)}")
            time.sleep(1.0 / args.hz)

        final = bus.read_qpos()
        print(f"  final      : {fmt_vec(final)}")
        print("Done. Adapter-v2 start pose reached; gripper is open.")
        return 0
    finally:
        bus.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
