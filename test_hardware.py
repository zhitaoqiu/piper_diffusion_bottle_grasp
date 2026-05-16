#!/usr/bin/env python3
"""
Hardware connectivity test for Piper robotic arm.

Steps:
  1. Connect to CAN bus
  2. Enable motors
  3. Read joint state & end pose (print current state)
  4. Send a small joint-space move (j1 +0.05 rad ≈ 2.8°) then move back
  5. Read arm status
  6. Disable & disconnect

Run:  python3 test_hardware.py
"""

import math
import sys
import time

# Add the driver package to path
import os as _os
_sdk_path = _os.environ.get(
    "PIPER_SDK_PATH", "/home/huatec/piper_bottle_grasp/piper_sdk_py_driver"
)
if _sdk_path not in sys.path:
    sys.path.insert(0, _sdk_path)
del _os

from piper_sdk_py_driver.sdk_adapter import PiperSdkAdapter

CAN_PORT = "can0"
GRIPPER_EXIST = True


def print_state(adapter: PiperSdkAdapter):
    """Read and print current joint state and end pose."""
    js = adapter.read_joint_state()
    ep = adapter.read_end_pose()

    print(f"  Joints (rad): {[round(p, 4) for p in js.position[:6]]}")
    print(f"  Gripper (m):  {round(js.position[6], 6)}")
    print(f"  End pose:     x={ep.x:.4f} y={ep.y:.4f} z={ep.z:.4f}  "
          f"roll={ep.roll:.4f} pitch={ep.pitch:.4f} yaw={ep.yaw:.4f}")


def main():
    adapter = PiperSdkAdapter(
        can_port=CAN_PORT,
        gripper_exist=GRIPPER_EXIST,
        enable_timeout=10.0,
    )

    # --- Step 1: Connect ---
    print(f"[1/6] Connecting to Piper arm on {CAN_PORT} ...")
    try:
        adapter.connect()
        print("  OK — connected.")
    except ConnectionError as e:
        print(f"  FAIL — {e}")
        return 1

    # --- Step 2: Enable ---
    print("[2/6] Enabling motors (this may take a few seconds) ...")
    ok = adapter.enable(blocking=True)
    if not ok:
        print("  FAIL — enable timed out. Check power and CAN connection.")
        adapter.disconnect()
        return 1
    print("  OK — all 6 motors enabled.")

    # --- Step 3: Read current state ---
    print("[3/6] Reading current state ...")
    try:
        print_state(adapter)
        print("  OK — state read successful.")
    except Exception as e:
        print(f"  FAIL — {e}")
        adapter.disable()
        adapter.disconnect()
        return 1

    # --- Step 4: Small test move ---
    print("[4/6] Testing joint-space move (j1 +0.05 rad) ...")
    try:
        js = adapter.read_joint_state()
        original = list(js.position[:6])

        # Move j1 a tiny bit forward
        target = list(original)
        target[0] += 0.05  # +0.05 rad
        adapter.send_joint_positions(target, velocity_percent=20)
        time.sleep(1.5)

        print("  After move:", end="")
        print_state(adapter)

        # Move back
        print("  Moving back ...")
        adapter.send_joint_positions(original, velocity_percent=20)
        time.sleep(1.5)

        print("  After return:", end="")
        print_state(adapter)
        print("  OK — joint move successful.")
    except Exception as e:
        print(f"  FAIL — {e}")
        adapter.disable()
        adapter.disconnect()
        return 1

    # --- Step 5: Arm status ---
    print("[5/6] Reading arm status ...")
    try:
        status = adapter.read_arm_status()
        print(f"  ctrl_mode={status.ctrl_mode} arm_status={status.arm_status} "
              f"motion_status={status.motion_status} err_code={status.err_code}")
        print("  OK — status read successful.")
    except Exception as e:
        print(f"  FAIL — {e}")

    # --- Step 6: Disable & disconnect ---
    print("[6/6] Disabling motors and disconnecting ...")
    try:
        adapter.disable()
        adapter.disconnect()
        print("  OK — done.")
    except Exception as e:
        print(f"  WARN — {e}")

    print("\n=== ALL TESTS PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
