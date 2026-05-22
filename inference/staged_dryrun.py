#!/usr/bin/env python3
"""
Staged safety dry-run for Diffusion Policy on Piper.
Follows ACT's proven pattern: policy for approach, scripted for post-approach.

Shadow mode (no robot movement):
  --stage shadow --shadow-mode single   One observation → full action chunk inspection.
  --stage shadow --shadow-mode closed   Per-step closed-loop via select_action().

Phased stages (A/B/C/D — manual gate between each, run separately):
  A_approach  — Policy-driven approach, gripper open, low speed, stop before close.
  B_close     — SCRIPTED close gripper only, arm joints frozen.
  C_lift_move — SCRIPTED lift (J3 -= 0.06), gripper stays closed.
  D_place_release — SCRIPTED descend (J2 += 0.04) → release → retreat to start_pose.

Safety:
  SPACE = pause, Q/ESC = emergency stop
  Joint limits, NaN/inf, action spike, stagnation detection
  Per-step joint delta clamp, wrist freeze @ J2 > 1.45
  First-command safety check BEFORE any motor command (Stage A)

Usage:
  conda activate piper_act
  python inference/staged_dryrun.py --stage shadow --shadow-mode single
  python inference/staged_dryrun.py --stage shadow --shadow-mode closed
  python inference/staged_dryrun.py --stage A_approach
  python inference/staged_dryrun.py --stage B_close
  python inference/staged_dryrun.py --stage C_lift_move
  python inference/staged_dryrun.py --stage D_place_release
"""

import argparse, datetime, json, sys, time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hardware.piper_wrapper import PiperRobot
from camera.rs_camera import RealSenseCamera, USBCamera, find_realsense_devices

# ── Constants ──────────────────────────────────────────────────────────
PIPER_GRIPPER_MAX_M = 0.101
GRIPPER_OPEN = 0.08
GRIPPER_CLOSE = 0.0
WRIST_FREEZE_J2 = 1.45
READY_J2 = 1.50
READY_COUNT_MIN = 5
STAGNATION_STEPS = 20
STAGNATION_THRESHOLD = 0.0008
SPIKE_MEDIAN_MULTIPLIER = 5.0
ACTION_SMOOTH_ALPHA = 0.5
MAX_STEPS_DEFAULT = 200
DEFAULT_HZ = 10.0
MAX_DELTA_REF = np.array([0.03, 0.03, 0.03, 0.012, 0.012, 0.012], dtype=np.float32)

STATE_KEY = "observation.state"
WRIST_IMAGE_KEY = "observation.images.wrist_rgb"
GLOBAL_IMAGE_KEY = "observation.images.global_rgb"
JOINT_NAMES = ["J1", "J2", "J3", "J4", "J5", "J6", "Grip"]
RECORD_DIR = PROJECT_ROOT / "logs" / "staged_dryrun"
CHECKPOINT = ("outputs/train/diffusion_same_data_comparison/"
              "checkpoints/last/pretrained_model")
HOME_PATH = PROJECT_ROOT / "config" / "start_pose.json"

# ── Stage A (policy-driven approach) params ──
APPROACH_ACTION_SCALE = 0.3
APPROACH_EXEC_RATIO = 0.35
APPROACH_VELOCITY_PCT = 10
APPROACH_MAX_DELTA = np.array([0.015, 0.015, 0.015, 0.006, 0.006, 0.006], dtype=np.float32)
GRIP_CLOSE_THRESHOLD = 0.06  # stop approach when predicted grip < this

# ── Scripted stage params ──
SCRIPTED_VELOCITY_PCT = 15
SCRIPTED_HZ = 30.0
SCRIPTED_MAX_STEP_RAD = 0.02
SCRIPTED_MAX_STEP_GRIP = 0.002
LIFT_J3_DELTA = -0.06     # J3 change for lift (negative = up)
DESCEND_J2_DELTA = 0.04   # J2 change for descend (positive = down)


# ── Utilities ──────────────────────────────────────────────────────────
def fmt_vec(values, precision=3):
    return "[" + ", ".join(f"{float(v):.{precision}f}" for v in values) + "]"


def image_to_tensor(image, device):
    return (torch.from_numpy(image).float().div(255.0)
            .permute(2, 0, 1).unsqueeze(0).to(device))


def build_preview(wrist_frame, global_frame, text, stage_label, color=(0, 255, 0)):
    frames = []
    if wrist_frame is not None:
        frames.append(cv2.cvtColor(wrist_frame.rgb, cv2.COLOR_RGB2BGR))
    if global_frame is not None:
        frames.append(cv2.cvtColor(global_frame.rgb, cv2.COLOR_RGB2BGR))
    if not frames:
        preview = np.zeros((480, 640, 3), dtype=np.uint8)
    else:
        preview = frames[0]
        for frame in frames[1:]:
            frame = cv2.resize(frame, (preview.shape[1], preview.shape[0]))
            preview = np.hstack([preview, frame])
    cv2.putText(preview, stage_label, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    cv2.putText(preview, text, (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return preview


def interpolate_joint_path(start, target, max_step_rad, max_step_gripper):
    """Generate intermediate joint targets (not including start, including target)."""
    diff = np.asarray(target, dtype=np.float32) - np.asarray(start, dtype=np.float32)
    arm_steps = int(np.ceil(np.max(np.abs(diff[:6])) / max_step_rad)) if max_step_rad > 0 else 1
    grip_steps = int(np.ceil(abs(diff[6]) / max_step_gripper)) if max_step_gripper > 0 else 1
    n_steps = max(arm_steps, grip_steps, 1)
    waypoints = []
    for i in range(1, n_steps + 1):
        alpha = i / n_steps
        interp = np.asarray(start, dtype=np.float32) + diff * alpha
        waypoints.append(interp)
    return waypoints


# ── Safety ─────────────────────────────────────────────────────────────
def first_command_safety_check(robot_state, predicted_action, action_scale,
                                max_delta, gripper_enabled):
    """Print full first-command safety breakdown. Returns True if safe."""
    rs = np.asarray(robot_state, dtype=np.float32)
    pa = np.asarray(predicted_action, dtype=np.float32)
    md = np.asarray(max_delta, dtype=np.float32)

    raw_delta = pa - rs
    scaled = rs.copy()
    scaled[:6] = rs[:6] + action_scale * raw_delta[:6]
    if gripper_enabled:
        scaled[6] = np.clip(rs[6] + action_scale * raw_delta[6],
                            0.0, PIPER_GRIPPER_MAX_M)
    else:
        scaled[6] = GRIPPER_OPEN

    clamped = scaled.copy()
    for j in range(6):
        d = clamped[j] - rs[j]
        d = np.clip(d, -md[j], md[j])
        clamped[j] = rs[j] + d
    clamped[:6] = np.clip(clamped[:6], -3.0, 3.0)

    print("\n  " + "=" * 58)
    print("  FIRST-COMMAND SAFETY CHECK")
    print("  " + "=" * 58)
    print(f"  {'Joint':<6} {'CurrState':>10} {'PredAction':>10} "
          f"{'RawΔ':>10} {'Scaled':>10} {'Clamped':>10} {'FinalΔ':>10} {'Limit':>8}")
    print(f"  {'-' * 58}")
    safe = True
    for j in range(7):
        limit = "OK"
        if j < 6 and abs(clamped[j]) > 3.0:
            limit = "LIMIT!"
            safe = False
        if j == 6 and (clamped[6] < 0 or clamped[6] > PIPER_GRIPPER_MAX_M):
            limit = "LIMIT!"
            safe = False
        print(f"  {JOINT_NAMES[j]:<6} {rs[j]:10.4f} {pa[j]:10.4f} "
              f"{raw_delta[j]:10.4f} {scaled[j]:10.4f} {clamped[j]:10.4f} "
              f"{clamped[j]-rs[j]:10.4f} {limit:>8}")
    print(f"  {'-' * 58}")
    print(f"  Safety assessment: {'PASS' if safe else 'FAIL — DO NOT SEND'}")
    print("  " + "=" * 58 + "\n")
    return safe


def check_nan_inf(robot_state, predicted_action):
    if np.any(np.isnan(predicted_action)) or np.any(np.isinf(predicted_action)):
        return True, "NaN/inf in predicted action"
    if np.any(np.isnan(robot_state)) or np.any(np.isinf(robot_state)):
        return True, "NaN/inf in robot state"
    return False, ""


def check_joint_limit(sent_target):
    if np.any(np.abs(sent_target[:6]) > 3.0):
        return True, "joint_limit"
    if sent_target[6] < 0 or sent_target[6] > PIPER_GRIPPER_MAX_M:
        return True, "gripper_limit"
    return False, ""


def check_action_spike(delta_history):
    if len(delta_history) < 10:
        return False, ""
    recent = np.abs(np.array(list(delta_history))).flatten()
    median_d = float(np.median(recent)) if len(recent) > 0 else 0.001
    current_max = float(np.max(np.abs(delta_history[-1])))
    if median_d > 0.0001 and current_max > SPIKE_MEDIAN_MULTIPLIER * max(median_d, 0.001):
        return True, f"action_spike (max={current_max:.4f} vs median={median_d:.4f})"
    return False, ""


def apply_action_pipeline(predicted_action, robot_state, action_scale,
                           max_delta, gripper_enabled, last_smoothed,
                           alpha=ACTION_SMOOTH_ALPHA):
    """Return smoothed+clamped target for the action pipeline (Stages A only)."""
    rs = np.asarray(robot_state, dtype=np.float32)
    pa = np.asarray(predicted_action, dtype=np.float32)
    md = np.asarray(max_delta, dtype=np.float32)

    scaled = rs.copy()
    scaled[:6] = rs[:6] + action_scale * (pa[:6] - rs[:6])
    if gripper_enabled:
        scaled[6] = np.clip(rs[6] + action_scale * (pa[6] - rs[6]),
                            0.0, PIPER_GRIPPER_MAX_M)
    else:
        scaled[6] = GRIPPER_OPEN

    clamped = scaled.copy()
    for j in range(6):
        d = clamped[j] - rs[j]
        d = np.clip(d, -md[j], md[j])
        clamped[j] = rs[j] + d

    wrist_frozen = rs[1] > WRIST_FREEZE_J2
    after_wrist = clamped.copy()
    if wrist_frozen:
        after_wrist[3:6] = rs[3:6]

    if last_smoothed is not None and alpha > 0:
        smoothed_arm = alpha * after_wrist[:6] + (1.0 - alpha) * last_smoothed
    else:
        smoothed_arm = after_wrist[:6].copy()

    smoothed = np.concatenate([smoothed_arm, [after_wrist[6]]])
    smoothed[:6] = np.clip(smoothed[:6], -3.0, 3.0)
    smoothed[6] = np.clip(smoothed[6], 0.0, PIPER_GRIPPER_MAX_M)

    return {
        "robot_state": rs,
        "predicted": pa,
        "raw_delta": pa - rs,
        "scaled": scaled,
        "clamped": clamped,
        "after_wrist": after_wrist,
        "smoothed": smoothed,
        "wrist_frozen": wrist_frozen,
    }


# ══════════════════════════════════════════════════════════════════════════
#  STAGE A — Policy-driven approach (ACT test-mode A pattern)
# ══════════════════════════════════════════════════════════════════════════
def run_approach(policy, preprocessor, postprocessor, robot,
                 wrist_cam, global_cam, needs_wrist, needs_global,
                 device, record_dir, hz, max_steps, no_gui):
    """Policy-driven approach phase. Gripper forced open. Stops when predicted
    grip < GRIP_CLOSE_THRESHOLD or after max_steps."""
    action_scale = APPROACH_ACTION_SCALE
    exec_ratio = APPROACH_EXEC_RATIO
    velocity_pct = APPROACH_VELOCITY_PCT
    max_delta = APPROACH_MAX_DELTA
    exec_steps = max(1, int(max_steps * exec_ratio))

    print(f"\n  === STAGE A: APPROACH (policy-driven) ===\n")
    print(f"  Executing up to {exec_steps} steps at {velocity_pct}% velocity")
    print(f"  action_scale={action_scale}  gripper=FORCED OPEN ({GRIPPER_OPEN:.3f}m)")
    print(f"  max_joint_delta={fmt_vec(max_delta, 4)}")
    print(f"  Stop rule: predicted grip < {GRIP_CLOSE_THRESHOLD:.3f}m → approach complete")

    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    records = {
        "stage": "A_approach",
        "timestamp": datetime.datetime.now().isoformat(),
        "hz": hz, "exec_steps": exec_steps,
        "start_joint_positions": robot.get_joint_positions(),
        "steps": [],
    }

    last_smoothed = None
    last_state = None
    delta_history = deque(maxlen=20)
    user_quit = False
    stop_reason = "completed"
    stagnation_count = 0
    ready_count = 0
    paused = False
    first_command_shown = False

    for step in range(exec_steps):
        loop_start = time.time()

        wrist_frame = wrist_cam.read() if wrist_cam else None
        global_frame = global_cam.read() if global_cam else None
        robot_state = robot.get_joint_positions()
        rs = np.asarray(robot_state, dtype=np.float32)
        ts = time.time()

        obs = {STATE_KEY: torch.from_numpy(rs).unsqueeze(0).to(device)}
        if needs_wrist and wrist_frame:
            obs[WRIST_IMAGE_KEY] = image_to_tensor(wrist_frame.rgb, device)
        if needs_global and global_frame:
            obs[GLOBAL_IMAGE_KEY] = image_to_tensor(global_frame.rgb, device)

        queue_len_before = len(policy._queues.get("action", []))

        with torch.inference_mode():
            normalized_obs = preprocessor(obs)
            action = policy.select_action(normalized_obs)
            action = postprocessor(action)
        if action.dim() == 2:
            action = action.squeeze(0)
        pa = action.cpu().numpy().astype(np.float32)

        queue_len_after = len(policy._queues.get("action", []))
        chunk_regenerated = (queue_len_before == 0)

        pipeline = apply_action_pipeline(
            pa, rs, action_scale=action_scale, max_delta=max_delta,
            gripper_enabled=False, last_smoothed=last_smoothed,
            alpha=ACTION_SMOOTH_ALPHA)
        sent_target = pipeline["smoothed"]

        # First-command safety check
        if step == 0 and not first_command_shown:
            safe = first_command_safety_check(
                rs, pa, action_scale, max_delta, gripper_enabled=False)
            first_command_shown = True
            if not safe:
                print("  First command safety check FAILED. Aborting.")
                stop_reason = "first_command_failed"
                break
            print("  Press ENTER to send first command, or Q to abort: ", end="", flush=True)
            try:
                line = input().strip().lower()
            except EOFError:
                line = ""
            if line == "q":
                stop_reason = "user_abort"
                break

        # Safety checks
        halt_nan, r_nan = check_nan_inf(rs, pa)
        halt_limit, r_limit = check_joint_limit(sent_target)
        halt_spike, r_spike = check_action_spike(delta_history)
        if halt_nan:
            print(f"\n  [HALT] {r_nan}"); stop_reason = r_nan; break
        if halt_limit:
            print(f"\n  [HALT] {r_limit}: {fmt_vec(sent_target)}"); stop_reason = r_limit; break
        if halt_spike:
            print(f"\n  [HALT] {r_spike}"); stop_reason = r_spike; break

        # Stage A stop rule: predicted grip close
        if pa[6] < GRIP_CLOSE_THRESHOLD:
            print(f"\n  [STOP] Predicted grip close ({pa[6]:.4f}m < {GRIP_CLOSE_THRESHOLD:.3f}) — approach complete.")
            stop_reason = "approach_complete"
            break

        # Stagnation (before 70% of exec)
        if step < exec_steps * 0.7 and last_state is not None:
            state_diff = float(np.max(np.abs(rs[:6] - last_state[:6])))
            if state_diff < STAGNATION_THRESHOLD:
                stagnation_count += 1
            else:
                stagnation_count = 0
            if stagnation_count >= STAGNATION_STEPS:
                print(f"\n  [HALT] Stagnation: {STAGNATION_STEPS} steps < {STAGNATION_THRESHOLD}")
                stop_reason = "stagnation"; break

        # Ready stop (J2 high for consecutive steps)
        if rs[1] > READY_J2 and step > exec_steps * 0.65:
            ready_count += 1
        else:
            ready_count = 0
        stop_act = (ready_count >= READY_COUNT_MIN) or (step + 1 >= exec_steps)

        step_rec = {
            "step": step, "timestamp": ts,
            "robot_state": rs.tolist(),
            "predicted_action": pa.tolist(),
            "sent_target": sent_target.tolist(),
            "wrist_frozen": pipeline["wrist_frozen"],
            "queue_len_before": queue_len_before,
            "queue_len_after": queue_len_after,
            "chunk_regenerated": chunk_regenerated,
        }
        records["steps"].append(step_rec)

        # Send to robot
        robot.set_joint_positions(sent_target.tolist(), velocity_pct=velocity_pct)

        if step % 10 == 0 or step == exec_steps - 1:
            regen_marker = " (REGEN)" if chunk_regenerated else ""
            print(f"  step {step+1:03d}/{exec_steps}  "
                  f"J2={rs[1]:.3f} grip={rs[6]:.3f} q={queue_len_after}{regen_marker}  "
                  f"Δmax={float(np.max(np.abs(sent_target[:6]-rs[:6]))):.4f}")

        if not no_gui:
            label = f"{'PAUSED' if paused else 'APPROACH'} {step+1}/{exec_steps}"
            preview = build_preview(wrist_frame, global_frame, label,
                                    "Stage A: APPROACH",
                                    color=(0, 0, 255) if not paused else (0, 165, 255))
            cv2.imshow("Staged Dry-Run", preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                user_quit = True; stop_reason = "user_quit"; break
            if key == ord(' '):
                paused = not paused

        elapsed = time.time() - loop_start
        step_time = 1.0 / hz
        if elapsed < step_time:
            time.sleep(step_time - elapsed)

        while paused:
            time.sleep(1.0 / hz)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                paused = False; user_quit = True; stop_reason = "user_quit"; break
            if key == ord(' '):
                paused = False
        if user_quit:
            break

        last_smoothed = pipeline["smoothed"][:6]
        last_state = rs
        delta_history.append(pa - rs)

        if stop_act:
            stop_reason = "ready" if ready_count >= READY_COUNT_MIN else "max_steps"
            break

    n_steps = len(records["steps"])
    records["stop_reason"] = stop_reason
    records["user_quit"] = user_quit

    if n_steps > 0:
        ra = np.array([s["predicted_action"] for s in records["steps"]])
        print(f"\n  Action stats ({n_steps} steps):")
        print(f"  {'Dim':>6}  {'mean':>10}  {'min':>10}  {'max':>10}  {'range':>10}")
        for d in range(7):
            print(f"  {JOINT_NAMES[d]:>6}  {ra[:, d].mean():10.4f}  "
                  f"{ra[:, d].min():10.4f}  {ra[:, d].max():10.4f}  "
                  f"{ra[:, d].max()-ra[:, d].min():10.4f}")

    save_path = record_dir / "episode_record.json"
    with open(save_path, "w") as f:
        json.dump(records, f, indent=2, default=str)
    print(f"\n  Record saved to {save_path}")
    print(f"  Approach finished ({stop_reason}, {n_steps} steps).")
    print("  [STAGE A] Check: is gripper aligned with bottle at pre-grasp position?")
    final = robot.get_joint_positions()
    print(f"  [STAGE A] Final J2 = {final[1]:.5f} rad  grip = {final[6]:.5f} m\n")


# ══════════════════════════════════════════════════════════════════════════
#  STAGE B — Scripted close gripper (ACT test-mode B close pattern)
# ══════════════════════════════════════════════════════════════════════════
def run_scripted_close(robot, wrist_cam, global_cam, record_dir, no_gui):
    """Scripted: close gripper only, arm joints frozen at current position."""
    print(f"\n  === STAGE B: CLOSE GRIPPER (scripted) ===\n")
    cur = np.asarray(robot.get_joint_positions(), dtype=np.float32)
    print(f"  Start position: {fmt_vec(cur)}")
    print(f"  Target: gripper 0.000m (closed), arm joints FROZEN")

    close_pose = cur.copy()
    close_pose[6] = GRIPPER_CLOSE
    path = interpolate_joint_path(cur, close_pose,
                                  SCRIPTED_MAX_STEP_RAD, SCRIPTED_MAX_STEP_GRIP)
    print(f"  Interpolated path: {len(path)} steps")

    user_quit = False
    paused = False
    for i, target in enumerate(path):
        loop_start = time.time()

        if not no_gui:
            wrist_frame = wrist_cam.read() if wrist_cam else None
            global_frame = global_cam.read() if global_cam else None
            preview = build_preview(wrist_frame, global_frame,
                                    f"{'PAUSED' if paused else 'CLOSE'} {i+1}/{len(path)}",
                                    "Stage B: CLOSE GRIPPER",
                                    color=(0, 0, 255) if not paused else (0, 165, 255))
            cv2.imshow("Staged Dry-Run", preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                user_quit = True; break
            if key == ord(' '):
                paused = not paused

        while paused:
            time.sleep(1.0 / SCRIPTED_HZ)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                paused = False; user_quit = True; break
            if key == ord(' '):
                paused = False
        if user_quit:
            break

        target[:6] = np.clip(target[:6], -3.14, 3.14)
        target[6] = np.clip(target[6], 0.0, PIPER_GRIPPER_MAX_M)
        robot.set_joint_positions(target.tolist(), velocity_pct=SCRIPTED_VELOCITY_PCT)

        if i == 0 or i == len(path) - 1 or (i + 1) % 10 == 0:
            print(f"    close {i+1:3d}/{len(path):3d}  grip={target[6]:.4f}")

        elapsed = time.time() - loop_start
        step_time = 1.0 / SCRIPTED_HZ
        if elapsed < step_time:
            time.sleep(step_time - elapsed)

    if user_quit:
        print("  User quit during close.")
        return

    # Dwell for gripper to engage
    print("  Holding close for 0.6s ...")
    hold_start = time.time()
    while time.time() - hold_start < 0.6:
        robot.set_joint_positions(close_pose.tolist(), velocity_pct=SCRIPTED_VELOCITY_PCT)
        time.sleep(1.0 / SCRIPTED_HZ)

    final = robot.get_joint_positions()
    print(f"  Gripper closed. Final grip = {final[6]:.5f} m")
    print("  [STAGE B] Verify: is bottle grasped?\n")


# ══════════════════════════════════════════════════════════════════════════
#  STAGE C — Scripted lift (ACT test-mode B lift pattern)
# ══════════════════════════════════════════════════════════════════════════
def run_scripted_lift(robot, wrist_cam, global_cam, record_dir, no_gui):
    """Scripted: lift (J3 -= 0.06 rad), gripper stays closed."""
    print(f"\n  === STAGE C: LIFT (scripted) ===\n")
    cur = np.asarray(robot.get_joint_positions(), dtype=np.float32)
    print(f"  Start position: {fmt_vec(cur)}")
    print(f"  Target: J3 += {LIFT_J3_DELTA:.3f} rad, gripper locked closed")

    lift_pose = cur.copy()
    lift_pose[2] += LIFT_J3_DELTA
    lift_pose[2] = np.clip(lift_pose[2], -3.14, 3.14)
    lift_pose[6] = GRIPPER_CLOSE
    path = interpolate_joint_path(cur, lift_pose,
                                  SCRIPTED_MAX_STEP_RAD, SCRIPTED_MAX_STEP_GRIP)
    print(f"  Interpolated path: {len(path)} steps")

    user_quit = False
    paused = False
    for i, target in enumerate(path):
        loop_start = time.time()

        if not no_gui:
            wrist_frame = wrist_cam.read() if wrist_cam else None
            global_frame = global_cam.read() if global_cam else None
            preview = build_preview(wrist_frame, global_frame,
                                    f"{'PAUSED' if paused else 'LIFT'} {i+1}/{len(path)}",
                                    "Stage C: LIFT",
                                    color=(0, 0, 255) if not paused else (0, 165, 255))
            cv2.imshow("Staged Dry-Run", preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                user_quit = True; break
            if key == ord(' '):
                paused = not paused

        while paused:
            time.sleep(1.0 / SCRIPTED_HZ)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                paused = False; user_quit = True; break
            if key == ord(' '):
                paused = False
        if user_quit:
            break

        target[:6] = np.clip(target[:6], -3.14, 3.14)
        target[6] = np.clip(target[6], 0.0, PIPER_GRIPPER_MAX_M)
        robot.set_joint_positions(target.tolist(), velocity_pct=SCRIPTED_VELOCITY_PCT)

        if i == 0 or i == len(path) - 1 or (i + 1) % 10 == 0:
            print(f"    lift {i+1:3d}/{len(path):3d}  {fmt_vec(target, 3)}")

        elapsed = time.time() - loop_start
        step_time = 1.0 / SCRIPTED_HZ
        if elapsed < step_time:
            time.sleep(step_time - elapsed)

    if user_quit:
        print("  User quit during lift.")
        return

    # Hold lift
    print("  Holding lift for 0.5s ...")
    hold_start = time.time()
    while time.time() - hold_start < 0.5:
        robot.set_joint_positions(lift_pose.tolist(), velocity_pct=SCRIPTED_VELOCITY_PCT)
        time.sleep(1.0 / SCRIPTED_HZ)

    final = robot.get_joint_positions()
    print(f"  Lift complete. Final J3 = {final[2]:.5f} rad")
    print("  [STAGE C] Verify: is bottle lifted?\n")


# ══════════════════════════════════════════════════════════════════════════
#  STAGE D — Scripted descend → release → retreat
# ══════════════════════════════════════════════════════════════════════════
def run_scripted_place_release(robot, wrist_cam, global_cam, record_dir, no_gui):
    """Scripted: descend (J2 += 0.04) → release (open gripper) → retreat (go home)."""
    print(f"\n  === STAGE D: DESCEND + RELEASE + RETREAT (scripted) ===\n")

    # ── Descend ──
    cur = np.asarray(robot.get_joint_positions(), dtype=np.float32)
    print(f"  Start position: {fmt_vec(cur)}")
    print(f"  Phase 1: Descend (J2 += {DESCEND_J2_DELTA:.3f} rad)")

    descend_pose = cur.copy()
    descend_pose[1] += DESCEND_J2_DELTA
    descend_pose[1] = np.clip(descend_pose[1], -3.14, 3.14)
    descend_pose[6] = GRIPPER_CLOSE
    path = interpolate_joint_path(cur, descend_pose,
                                  SCRIPTED_MAX_STEP_RAD, SCRIPTED_MAX_STEP_GRIP)
    print(f"  Interpolated path: {len(path)} steps")

    user_quit = False
    paused = False

    for i, target in enumerate(path):
        t_start = time.time()

        if not no_gui:
            wrist_frame = wrist_cam.read() if wrist_cam else None
            global_frame = global_cam.read() if global_cam else None
            preview = build_preview(wrist_frame, global_frame,
                                    f"{'PAUSED' if paused else 'DESCEND'} {i+1}/{len(path)}",
                                    "Stage D: PLACE+RELEASE",
                                    color=(0, 0, 255) if not paused else (0, 165, 255))
            cv2.imshow("Staged Dry-Run", preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                user_quit = True; break
            if key == ord(' '):
                paused = not paused

        while paused:
            time.sleep(1.0 / SCRIPTED_HZ)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                paused = False; user_quit = True; break
            if key == ord(' '):
                paused = False
        if user_quit:
            break
        target[:6] = np.clip(target[:6], -3.14, 3.14)
        target[6] = np.clip(target[6], 0.0, PIPER_GRIPPER_MAX_M)
        robot.set_joint_positions(target.tolist(), velocity_pct=SCRIPTED_VELOCITY_PCT)
        if i == 0 or i == len(path) - 1 or (i + 1) % 10 == 0:
            print(f"    descend {i+1:3d}/{len(path):3d}  {fmt_vec(target, 3)}")
        elapsed = time.time() - t_start
        step_time = 1.0 / SCRIPTED_HZ
        if elapsed < step_time:
            time.sleep(step_time - elapsed)

    if user_quit:
        print("  User quit during descend.")
        return
    print("  Descend complete.")

    # ── Release ──
    print(f"\n  Phase 2: Release (open gripper to {GRIPPER_OPEN:.3f}m)")
    cur = np.asarray(robot.get_joint_positions(), dtype=np.float32)
    release_pose = cur.copy()
    release_pose[6] = GRIPPER_OPEN
    path = interpolate_joint_path(cur, release_pose,
                                  SCRIPTED_MAX_STEP_RAD, max_step_gripper=0.004)
    for i, target in enumerate(path):
        t_start = time.time()

        if not no_gui:
            wrist_frame = wrist_cam.read() if wrist_cam else None
            global_frame = global_cam.read() if global_cam else None
            preview = build_preview(wrist_frame, global_frame,
                                    f"{'PAUSED' if paused else 'RELEASE'} {i+1}/{len(path)}",
                                    "Stage D: PLACE+RELEASE",
                                    color=(0, 0, 255) if not paused else (0, 165, 255))
            cv2.imshow("Staged Dry-Run", preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                user_quit = True; break
            if key == ord(' '):
                paused = not paused

        while paused:
            time.sleep(1.0 / SCRIPTED_HZ)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                paused = False; user_quit = True; break
            if key == ord(' '):
                paused = False
        if user_quit:
            break
        target[:6] = np.clip(target[:6], -3.14, 3.14)
        target[6] = np.clip(target[6], 0.0, PIPER_GRIPPER_MAX_M)
        robot.set_joint_positions(target.tolist(), velocity_pct=SCRIPTED_VELOCITY_PCT)
        if i == 0 or i == len(path) - 1:
            print(f"    release {i+1:3d}/{len(path):3d}  grip={target[6]:.4f}")
        elapsed = time.time() - t_start
        step_time = 1.0 / SCRIPTED_HZ
        if elapsed < step_time:
            time.sleep(step_time - elapsed)

    if user_quit:
        print("  User quit during release.")
        return

    # Dwell after release
    print("  Holding release for 0.5s ...")
    hold_start = time.time()
    while time.time() - hold_start < 0.5:
        robot.set_joint_positions(release_pose.tolist(), velocity_pct=SCRIPTED_VELOCITY_PCT)
        time.sleep(1.0 / SCRIPTED_HZ)
    print("  Gripper released.")

    # ── Retreat to home ──
    if HOME_PATH.exists():
        home = json.loads(HOME_PATH.read_text())
        home_target = np.array(home["joint_positions"], dtype=np.float32)
    else:
        print(f"  [WARN] Home pose not found: {HOME_PATH}. Skipping retreat.")
        home_target = None

    if home_target is not None:
        print(f"\n  Phase 3: Retreat to home ({fmt_vec(home_target)})")
        cur = np.asarray(robot.get_joint_positions(), dtype=np.float32)
        home_target[6] = cur[6]  # preserve current gripper state
        path = interpolate_joint_path(cur, home_target,
                                      max_step_rad=0.03, max_step_gripper=0.004)
        for i, target in enumerate(path):
            t_start = time.time()

            if not no_gui:
                wrist_frame = wrist_cam.read() if wrist_cam else None
                global_frame = global_cam.read() if global_cam else None
                preview = build_preview(wrist_frame, global_frame,
                                        f"{'PAUSED' if paused else 'RETREAT'} {i+1}/{len(path)}",
                                        "Stage D: PLACE+RELEASE",
                                        color=(0, 0, 255) if not paused else (0, 165, 255))
                cv2.imshow("Staged Dry-Run", preview)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q'), ord('Q')):
                    user_quit = True; break
                if key == ord(' '):
                    paused = not paused

            while paused:
                time.sleep(1.0 / SCRIPTED_HZ)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q'), ord('Q')):
                    paused = False; user_quit = True; break
                if key == ord(' '):
                    paused = False
            if user_quit:
                break
            target[:6] = np.clip(target[:6], -3.14, 3.14)
            target[6] = np.clip(target[6], 0.0, PIPER_GRIPPER_MAX_M)
            robot.set_joint_positions(target.tolist(), velocity_pct=SCRIPTED_VELOCITY_PCT)
            if i == 0 or i == len(path) - 1 or (i + 1) % 10 == 0:
                print(f"    retreat {i+1:3d}/{len(path):3d}  {fmt_vec(target, 3)}")
            elapsed = time.time() - t_start
            step_time = 1.0 / SCRIPTED_HZ
            if elapsed < step_time:
                time.sleep(step_time - elapsed)

    print("  [STAGE D] Place + release + retreat complete.\n")


# ══════════════════════════════════════════════════════════════════════════
#  SHADOW MODES (unchanged from original)
# ══════════════════════════════════════════════════════════════════════════
def run_shadow_single(policy, preprocessor, postprocessor, robot,
                       wrist_cam, global_cam, needs_wrist, needs_global,
                       device, record_dir):
    """One observation → one predict_action_chunk → plot full chunk."""
    print("\n  === SINGLE-CHUNK SHADOW INSPECTION ===\n")
    print("  Reading current observation ...")

    robot_state = robot.get_joint_positions()
    rs = np.asarray(robot_state, dtype=np.float32)
    print(f"  Current robot state: {fmt_vec(rs)}")

    obs = {STATE_KEY: torch.from_numpy(rs).unsqueeze(0).to(device)}
    if needs_wrist and wrist_cam:
        wrist_frame = wrist_cam.read()
        obs[WRIST_IMAGE_KEY] = image_to_tensor(wrist_frame.rgb, device)
    if needs_global and global_cam:
        global_frame = global_cam.read()
        obs[GLOBAL_IMAGE_KEY] = image_to_tensor(global_frame.rgb, device)

    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    from lerobot.policies.utils import populate_queues

    normalized_obs = preprocessor(obs)
    if policy.config.image_features:
        normalized_obs = dict(normalized_obs)
        from lerobot.policies.diffusion.modeling_diffusion import OBS_IMAGES
        normalized_obs[OBS_IMAGES] = torch.stack(
            [normalized_obs[k] for k in policy.config.image_features], dim=-4)
    policy._queues = populate_queues(policy._queues, normalized_obs)

    with torch.inference_mode():
        actions = policy.predict_action_chunk(normalized_obs)
    if actions.dim() == 2:
        actions = actions
    chunk = actions.cpu().numpy().astype(np.float32)
    n_act = len(chunk)

    print(f"\n  Generated action chunk: {n_act} steps (horizon={policy.config.horizon}, "
          f"n_action_steps={policy.config.n_action_steps})")
    print(f"  {'Dim':>6}  {'mean':>10}  {'min':>10}  {'max':>10}  {'range':>10}")
    for d in range(7):
        print(f"  {JOINT_NAMES[d]:>6}  {chunk[:, d].mean():10.4f}  "
              f"{chunk[:, d].min():10.4f}  {chunk[:, d].max():10.4f}  "
              f"{chunk[:, d].max() - chunk[:, d].min():10.4f}")

    first_command_safety_check(rs, chunk[0], action_scale=0.5,
                                max_delta=MAX_DELTA_REF, gripper_enabled=False)

    layers = {"raw": chunk[:, :6], "clamped": np.zeros_like(chunk[:, :6]),
              "smoothed": np.zeros_like(chunk[:, :6])}
    last_sm = None
    for i in range(n_act):
        pipeline = apply_action_pipeline(
            chunk[i], rs, action_scale=0.5, max_delta=MAX_DELTA_REF,
            gripper_enabled=False, last_smoothed=last_sm, alpha=ACTION_SMOOTH_ALPHA)
        layers["clamped"][i] = pipeline["clamped"][:6]
        layers["smoothed"][i] = pipeline["smoothed"][:6]
        last_sm = pipeline["smoothed"][:6] if i == 0 else last_sm

    for d in range(7):
        abs_deltas = np.abs(np.diff(chunk[:, d]))
        if len(abs_deltas) > 0:
            med = float(np.median(abs_deltas))
            mx = float(np.max(abs_deltas))
            if med > 0 and mx > SPIKE_MEDIAN_MULTIPLIER * max(med, 0.001):
                print(f"  [WARN] {JOINT_NAMES[d]} chunk-internal spike: "
                      f"max_step_diff={mx:.4f} vs median={med:.4f}")

    grip_seq = chunk[:, 6]
    grip_close_idx = np.argmin(grip_seq)
    print(f"\n  Gripper: min={grip_seq.min():.4f} (step {grip_close_idx})  "
          f"pattern: {grip_seq[0]:.4f} -> {grip_seq[grip_close_idx]:.4f} -> {grip_seq[-1]:.4f}")

    ts = datetime.datetime.now().isoformat()
    record = {
        "mode": "shadow_single", "timestamp": ts,
        "start_joint_positions": rs.tolist(),
        "chunk_n_actions": n_act,
        "chunk": chunk.tolist(),
        "layers_raw": layers["raw"].tolist(),
        "layers_clamped": layers["clamped"].tolist(),
        "layers_smoothed": layers["smoothed"].tolist(),
        "gripper_chunk": chunk[:, 6].tolist(),
    }
    save_path = record_dir / "single_chunk_record.json"
    with open(save_path, "w") as f:
        json.dump(record, f, indent=2, default=str)
    print(f"\n  Record saved to {save_path}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    ref_rs = rs[:6]
    for d in range(6):
        ax = axes[d // 3][d % 3]
        ax.axhline(y=ref_rs[d], color='gray', linestyle='--', alpha=0.5, label='current')
        ax.plot(layers["raw"][:, d], 'r-', alpha=0.6, linewidth=0.8, label='raw')
        ax.plot(layers["clamped"][:, d], 'orange', alpha=0.8, linewidth=1.2, label='clamped')
        ax.plot(layers["smoothed"][:, d], 'b-', linewidth=1.5, label='smoothed')
        ax.set_title(f"{JOINT_NAMES[d]} (chunk={n_act} steps)")
        ax.set_ylabel("rad")
        if d == 0:
            ax.legend(fontsize=7)
    ax = axes[2][1]
    ax.plot(chunk[:, 6], 'g-', linewidth=1.5)
    ax.set_title("Gripper")
    ax.set_ylabel("m")
    axes[2][2].axis("off")
    fig.tight_layout()
    fig.savefig(str(record_dir / "single_chunk_curves.png"), dpi=100)
    plt.close(fig)
    print(f"  Curves saved to {record_dir / 'single_chunk_curves.png'}")

    return True


def run_shadow_closed(policy, preprocessor, postprocessor, robot,
                       wrist_cam, global_cam, needs_wrist, needs_global,
                       device, record_dir, hz, max_steps):
    """Per-step closed-loop via select_action(). No motor commands sent."""
    print("\n  === CLOSED-LOOP SHADOW ===\n")
    print(f"  {max_steps} steps, {hz} Hz. Stagnation check DISABLED.")

    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    records = {
        "mode": "shadow_closed",
        "timestamp": datetime.datetime.now().isoformat(),
        "hz": hz, "max_steps": max_steps,
        "start_joint_positions": robot.get_joint_positions(),
        "steps": [],
    }

    rs_init = np.asarray(robot.get_joint_positions(), dtype=np.float32)
    last_smoothed = None
    delta_history = deque(maxlen=20)
    user_quit = False
    stop_reason = "completed"
    step = 0
    paused = False
    chunk_regeneration_count = 0

    force_scale = 0.5
    force_md = MAX_DELTA_REF

    while step < max_steps:
        loop_start = time.time()

        wrist_frame = wrist_cam.read() if wrist_cam else None
        global_frame = global_cam.read() if global_cam else None
        robot_state = robot.get_joint_positions()
        rs = np.asarray(robot_state, dtype=np.float32)
        ts = time.time()

        obs = {STATE_KEY: torch.from_numpy(rs).unsqueeze(0).to(device)}
        if needs_wrist and wrist_frame:
            obs[WRIST_IMAGE_KEY] = image_to_tensor(wrist_frame.rgb, device)
        if needs_global and global_frame:
            obs[GLOBAL_IMAGE_KEY] = image_to_tensor(global_frame.rgb, device)

        queue_len_before = len(policy._queues.get("action", []))

        with torch.inference_mode():
            normalized_obs = preprocessor(obs)
            action = policy.select_action(normalized_obs)
            action = postprocessor(action)
        if action.dim() == 2:
            action = action.squeeze(0)
        pa = action.cpu().numpy().astype(np.float32)

        queue_len_after = len(policy._queues.get("action", []))
        chunk_regenerated = (queue_len_before == 0)
        if chunk_regenerated:
            chunk_regeneration_count += 1

        chunk_boundary_jump = False
        if chunk_regenerated and step > 0 and len(records["steps"]) > 0:
            prev_pa = np.array(records["steps"][-1]["predicted_action"])
            jump = float(np.max(np.abs(pa - prev_pa)))
            if jump > 0.05:
                chunk_boundary_jump = True

        pipeline = apply_action_pipeline(
            pa, rs, action_scale=force_scale, max_delta=force_md,
            gripper_enabled=False, last_smoothed=last_smoothed,
            alpha=ACTION_SMOOTH_ALPHA)

        halt_nan, halt_reason_nan = check_nan_inf(rs, pa)
        halt_limit, halt_reason_limit = check_joint_limit(pipeline["smoothed"])
        halt_spike, halt_reason_spike = check_action_spike(delta_history)
        halt = halt_nan or halt_limit or halt_spike
        halt_reason = halt_reason_nan or halt_reason_limit or halt_reason_spike

        if halt:
            print(f"\n  [HALT] Safety: {halt_reason}")
            stop_reason = halt_reason
            break

        step_rec = {
            "step": step, "timestamp": ts, "robot_state": rs.tolist(),
            "predicted_action": pa.tolist(),
            "queue_len_before": queue_len_before,
            "queue_len_after": queue_len_after,
            "chunk_regenerated": chunk_regenerated,
            "chunk_boundary_jump": chunk_boundary_jump,
            "hypothetical_raw": pipeline["predicted"].tolist(),
            "hypothetical_scaled": pipeline["scaled"].tolist(),
            "hypothetical_clamped": pipeline["clamped"].tolist(),
            "hypothetical_smoothed": pipeline["smoothed"].tolist(),
            "wrist_frozen": pipeline["wrist_frozen"],
        }
        records["steps"].append(step_rec)

        regen_marker = " *** REGENERATED ***" if chunk_regenerated else ""
        jump_marker = " !!! BOUNDARY JUMP !!!" if chunk_boundary_jump else ""
        queue_info = (f"queue={queue_len_after}"
                      f"  chunk_regens={chunk_regeneration_count}"
                      f"{regen_marker}{jump_marker}")
        if step % 10 == 0 or chunk_regenerated or chunk_boundary_jump:
            print(f"  [SHADOW] step {step+1:03d}/{max_steps}  {queue_info}")
            if chunk_regenerated or chunk_boundary_jump:
                print(f"    predicted: {fmt_vec(pa)}")
                print(f"    smoothed:  {fmt_vec(pipeline['smoothed'])}")

        if wrist_cam or global_cam:
            label = f"SHADOW {step+1}/{max_steps}"
            preview = build_preview(wrist_frame, global_frame, label,
                                    "Stage 0: SHADOW closed", color=(0, 165, 255))
            cv2.imshow("Staged Dry-Run", preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                user_quit = True
                stop_reason = "user_quit"
                break
            if key == ord(' '):
                paused = not paused

        elapsed = time.time() - loop_start
        step_time = 1.0 / hz
        if elapsed < step_time:
            time.sleep(step_time - elapsed)

        while paused:
            time.sleep(1.0 / hz)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                paused = False
                user_quit = True
                stop_reason = "user_quit"
                break
            if key == ord(' '):
                paused = False
        if user_quit:
            break

        last_smoothed = pipeline["smoothed"][:6]
        delta_history.append(pa - rs)
        step += 1

    n_steps = len(records["steps"])
    records["stop_reason"] = stop_reason
    records["user_quit"] = user_quit
    records["chunk_regeneration_count"] = chunk_regeneration_count

    if n_steps > 0:
        ra = np.array([s["predicted_action"] for s in records["steps"]])
        hra = np.array([s["hypothetical_raw"] for s in records["steps"]])
        hcl = np.array([s["hypothetical_clamped"] for s in records["steps"]])
        hsm = np.array([s["hypothetical_smoothed"] for s in records["steps"]])

        print(f"\n  Action stats ({n_steps} steps):")
        print(f"  {'Dim':>6}  {'raw_mean':>10}  {'raw_range':>10}  "
              f"{'clamped_range':>10}  {'smoothed_range':>10}")
        for d in range(7):
            print(f"  {JOINT_NAMES[d]:>6}  {ra[:, d].mean():10.4f}  "
                  f"{ra[:, d].max()-ra[:, d].min():10.4f}  "
                  f"{hcl[:, d].max()-hcl[:, d].min() if d < 6 else 0:10.4f}  "
                  f"{hsm[:, d].max()-hsm[:, d].min() if d < 6 else 0:10.4f}")

        for d in range(7):
            abs_deltas = np.abs(np.diff(ra[:, d]))
            if len(abs_deltas) > 0:
                med = float(np.median(abs_deltas))
                mx = float(np.max(abs_deltas))
                if med > 0 and mx > SPIKE_MEDIAN_MULTIPLIER * max(med, 0.001):
                    print(f"  [WARN] {JOINT_NAMES[d]} spike: "
                          f"max_step_diff={mx:.4f} vs median={med:.4f}")

        boundaries = [s["step"] for s in records["steps"] if s["chunk_regenerated"]]
        jumps = [s["step"] for s in records["steps"] if s["chunk_boundary_jump"]]
        print(f"\n  Chunk regenerations: {chunk_regeneration_count} (at steps {boundaries})")
        if jumps:
            print(f"  Chunk boundary jumps > 0.05 rad: {len(jumps)} (at steps {jumps})")
        else:
            print("  No significant chunk boundary jumps detected.")

        grip_seq = ra[:, 6]
        grip_close_idx = np.argmin(grip_seq)
        print(f"  Gripper: min={grip_seq.min():.4f} (step {grip_close_idx})  "
              f"pattern: {grip_seq[0]:.4f} -> {grip_seq[grip_close_idx]:.4f} "
              f"-> {grip_seq[-1]:.4f}")

        save_path = record_dir / "closed_loop_record.json"
        with open(save_path, "w") as f:
            json.dump(records, f, indent=2, default=str)
        print(f"\n  Record saved to {save_path}")

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 3, figsize=(18, 14))
        for d in range(6):
            ax = axes[d // 3][d % 3]
            steps_arr = np.arange(n_steps)
            ax.plot(steps_arr, hra[:, d], 'r-', alpha=0.4, linewidth=0.6, label='raw')
            ax.plot(steps_arr, hcl[:, d], 'orange', alpha=0.7, linewidth=1.0, label='clamped')
            ax.plot(steps_arr, hsm[:, d], 'b-', linewidth=1.5, label='smoothed')
            ax.set_title(f"{JOINT_NAMES[d]}")
            ax.set_ylabel("rad")
            if d == 0:
                ax.legend(fontsize=7)
        ax = axes[2][1]
        ax.plot(ra[:, 6], 'g-', linewidth=1.5)
        ax.set_title("Gripper (raw predicted)")
        ax.set_ylabel("m")
        axes[2][2].axis("off")
        fig.tight_layout()
        fig.savefig(str(record_dir / "closed_loop_curves.png"), dpi=100)
        plt.close(fig)
        print(f"  Curves saved to {record_dir / 'closed_loop_curves.png'}")


# ── Stage labels ───────────────────────────────────────────────────────
STAGE_LABELS = {
    "shadow": "Stage 0: SHADOW",
    "A_approach": "Stage A: APPROACH (policy-driven)",
    "B_close": "Stage B: CLOSE GRIPPER (scripted)",
    "C_lift_move": "Stage C: LIFT (scripted)",
    "D_place_release": "Stage D: PLACE+RELEASE+RETREAT (scripted)",
}

NEEDS_POLICY = {"shadow", "A_approach"}


# ── Main ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Staged safety dry-run for Diffusion Policy (ACT-proven pattern)")
    parser.add_argument("--stage", type=str, required=True,
                        choices=["shadow", "A_approach", "B_close",
                                 "C_lift_move", "D_place_release"])
    parser.add_argument("--shadow-mode", type=str, default="single",
                        choices=["single", "closed"])
    parser.add_argument("--checkpt", type=str, default=CHECKPOINT)
    parser.add_argument("--can-port", type=str, default="can0")
    parser.add_argument("--hz", type=float, default=DEFAULT_HZ)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS_DEFAULT)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-global", action="store_true")
    parser.add_argument("--global-camera", type=str, default="auto")
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument("--record-dir", type=str, default=str(RECORD_DIR))
    args = parser.parse_args()

    stage = args.stage
    is_shadow = (stage == "shadow")
    needs_policy = stage in NEEDS_POLICY

    print("=" * 60)
    print(f"  {STAGE_LABELS[stage]}")
    if is_shadow:
        print(f"  Shadow mode: {args.shadow_mode}")
    print("=" * 60)

    record_dir = (Path(args.record_dir) / stage
                  / datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    record_dir.mkdir(parents=True, exist_ok=True)

    # ── Device ──
    if needs_policy:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA not available.")
        device = torch.device(args.device)
        print(f"  Device: {device}")
    else:
        device = None

    # ── Load policy (only for shadow + A_approach) ──
    policy = preprocessor = postprocessor = None
    needs_wrist = needs_global = False
    if needs_policy:
        print(f"\n[1/4] Loading policy ...")
        print(f"  checkpoint: {args.checkpt}")
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
        policy = DiffusionPolicy.from_pretrained(args.checkpt)
        policy.to(device)
        policy.eval()
        print(f"  horizon={policy.config.horizon}  "
              f"n_action_steps={policy.config.n_action_steps}  "
              f"n_obs_steps={policy.config.n_obs_steps}")

        from lerobot.policies.factory import make_pre_post_processors
        d = str(device)
        pre = {"device_processor": {"device": d}, "normalizer_processor": {"device": d}}
        post = {"unnormalizer_processor": {"device": device.type},
                "device_processor": {"device": "cpu"}}
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy.config, pretrained_path=args.checkpt,
            preprocessor_overrides=pre, postprocessor_overrides=post)
        print("  Processors ready.")

        features = getattr(policy.config, "input_features", {}) or {}
        needs_wrist = WRIST_IMAGE_KEY in features
        needs_global = GLOBAL_IMAGE_KEY in features

    # ── Cameras ──
    step_label = "[2/4]" if needs_policy else "[1/2]"
    print(f"\n{step_label} Cameras ...")
    wrist_cam = None; global_cam = None
    if needs_policy and needs_wrist:
        rs_serials = find_realsense_devices()
        wrist_cam = RealSenseCamera(serial=rs_serials[0] if rs_serials else "",
                                    width=640, height=480, fps=30, enable_depth=False)
        print(f"  Wrist: {rs_serials[0] if rs_serials else 'none'}")
    if needs_policy and needs_global and not args.no_global:
        global_cam = USBCamera(device_id=args.global_camera, width=640, height=480, fps=30)
        print(f"  Global: OK")
    if not needs_policy:
        # For scripted stages, still try to open cameras for preview
        try:
            rs_serials = find_realsense_devices()
            if rs_serials:
                wrist_cam = RealSenseCamera(serial=rs_serials[0],
                                            width=640, height=480, fps=30, enable_depth=False)
                print(f"  Wrist: {rs_serials[0]}")
        except Exception:
            pass
        try:
            global_cam = USBCamera(device_id=args.global_camera, width=640, height=480, fps=30)
            print(f"  Global: OK")
        except Exception:
            pass

    # ── Robot ──
    step_label = "[3/4]" if needs_policy else "[2/2]"
    print(f"\n{step_label} Connecting Piper ({args.can_port}) ...")
    robot = PiperRobot(can_port=args.can_port, disable_torque_on_disconnect=False)
    robot.connect()
    print("  Connected.")

    if needs_policy:
        print("\n[4/4] Ready.")
    print("\n" + "-" * 60)
    print("  SPACE = trigger episode    Q/ESC = quit")
    if not is_shadow:
        print("  During episode: SPACE = pause/resume    Q = emergency stop")
    print("-" * 60 + "\n")

    try:
        while True:
            # Wait for trigger
            if args.no_gui:
                cmd = input("  Press ENTER to run, Q to quit: ").strip().lower()
                if cmd == "q":
                    break
            else:
                preview = build_preview(
                    wrist_cam.read() if wrist_cam else None,
                    global_cam.read() if global_cam else None,
                    "READY - SPACE to run", STAGE_LABELS[stage])
                cv2.imshow("Staged Dry-Run", preview)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q'), ord('Q')):
                    break
                if key != ord(' '):
                    continue

            # Dispatch
            if is_shadow:
                if args.shadow_mode == "single":
                    run_shadow_single(policy, preprocessor, postprocessor,
                                      robot, wrist_cam, global_cam,
                                      needs_wrist, needs_global, device, record_dir)
                else:
                    run_shadow_closed(policy, preprocessor, postprocessor,
                                      robot, wrist_cam, global_cam,
                                      needs_wrist, needs_global, device,
                                      record_dir, args.hz, args.max_steps)
            elif stage == "A_approach":
                run_approach(policy, preprocessor, postprocessor,
                             robot, wrist_cam, global_cam,
                             needs_wrist, needs_global, device,
                             record_dir, args.hz, args.max_steps, args.no_gui)
            elif stage == "B_close":
                run_scripted_close(robot, wrist_cam, global_cam,
                                   record_dir, args.no_gui)
            elif stage == "C_lift_move":
                run_scripted_lift(robot, wrist_cam, global_cam,
                                  record_dir, args.no_gui)
            elif stage == "D_place_release":
                run_scripted_place_release(robot, wrist_cam, global_cam,
                                           record_dir, args.no_gui)

            print("  Episode complete. Waiting for next trigger ...\n")

    except KeyboardInterrupt:
        print("\n  Interrupted.")
    finally:
        try:
            cur = robot.get_joint_positions()
            robot.set_joint_positions(cur, velocity_pct=50)
        except Exception:
            pass
        print("  Stopped. Arm stays ENABLED.")
        if wrist_cam:
            wrist_cam.close()
        if global_cam:
            global_cam.close()
        if not args.no_gui:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
