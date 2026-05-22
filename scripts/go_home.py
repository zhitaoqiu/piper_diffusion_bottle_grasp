#!/usr/bin/env python3
"""Move Piper to saved home position. Torque stays ON after exit."""
import sys, json, numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from hardware.piper_wrapper import PiperRobot

HOME_PATH = PROJECT_ROOT / "config" / "start_pose.json"

def main():
    home = json.loads(HOME_PATH.read_text())
    target = np.array(home["joint_positions"], dtype=np.float32)
    print(f"Home position: [{', '.join(f'{x:.4f}' for x in target)}]")

    robot = PiperRobot(disable_torque_on_disconnect=False)
    robot.connect()
    print("Moving to home at 30% velocity...")
    robot.set_joint_positions(target.tolist(), velocity_pct=30)
    print("Done. Arm stays ENABLED.")

if __name__ == "__main__":
    main()
