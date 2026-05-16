#!/usr/bin/env python3
"""
Deploy Diffusion Policy on Piper arm for bottle grasping.

Usage:
  conda activate piper_act
  python3 inference/deploy.py \
    --checkpt outputs/train/piper_bottle_grasp/checkpoints/last/pretrained_model

Controls:
  SPACE  = run one grasp attempt
  Q/ESC  = quit
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hardware.piper_wrapper import PiperRobot
from camera.rs_camera import RealSenseCamera, USBCamera, find_realsense_devices

PIPER_GRIPPER_MAX_M = 0.101
GRIPPER_OPEN = 0.08
GRIPPER_CLOSE = 0.0
MAX_DELTA_PER_JOINT = np.array([0.03, 0.03, 0.03, 0.012, 0.012, 0.012], dtype=np.float32)
ACTION_SMOOTH_ALPHA = 0.5
APPROACH_STEPS_DEFAULT = 200
WRIST_FREEZE_J2 = 1.45
READY_J2 = 1.50
READY_COUNT_MIN = 5
STAGNATION_STEPS = 20
STAGNATION_THRESHOLD = 0.0008


def load_policy_processors(policy, checkpt: str, device: torch.device):
    from lerobot.policies.factory import make_pre_post_processors

    pre = {"device_processor": {"device": device.type}, "normalizer_processor": {"device": device.type}}
    post = {"unnormalizer_processor": {"device": device.type}, "device_processor": {"device": "cpu"}}
    return make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=checkpt,
        preprocessor_overrides=pre, postprocessor_overrides=post,
    )


def prepare_observation(state, wrist_img, global_img, device):
    obs = {}
    obs["observation.state"] = torch.from_numpy(
        np.asarray(state, dtype=np.float32)
    ).unsqueeze(0).to(device)

    if wrist_img is not None:
        t = torch.from_numpy(wrist_img).float() / 255.0
        obs["observation.images.wrist_rgb"] = t.permute(2, 0, 1).unsqueeze(0).to(device)

    if global_img is not None:
        t = torch.from_numpy(global_img).float() / 255.0
        obs["observation.images.global_rgb"] = t.permute(2, 0, 1).unsqueeze(0).to(device)

    return obs


def build_preview(wrist_frame, global_frame, text: str, color=(0, 255, 0)):
    preview = cv2.cvtColor(wrist_frame.rgb, cv2.COLOR_RGB2BGR)
    cv2.putText(preview, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    if global_frame is not None:
        g = cv2.cvtColor(global_frame.rgb, cv2.COLOR_RGB2BGR)
        g = cv2.resize(g, (preview.shape[1], preview.shape[0]))
        preview = np.hstack([preview, g])
    return preview


def should_quit(key: int) -> bool:
    return key in (27, ord('q'), ord('Q'))


def fmt_vec(values, precision=3):
    return "[" + ", ".join(f"{float(v):.{precision}f}" for v in values) + "]"


def max_abs_diff(cur, prev) -> float:
    if prev is None:
        return float("nan")
    return float(np.max(np.abs(np.asarray(cur, dtype=np.float32) - np.asarray(prev, dtype=np.float32))))


def interpolate_joint_path(start, target, max_step_rad, max_step_gripper):
    start = np.asarray(start, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    diff = target - start
    arm_steps = int(np.ceil(np.max(np.abs(diff[:6])) / max_step_rad)) if max_step_rad > 0 else 1
    grip_steps = int(np.ceil(abs(diff[6]) / max_step_gripper)) if max_step_gripper > 0 else 1
    n_steps = max(arm_steps, grip_steps, 1)
    waypoints = []
    for i in range(1, n_steps + 1):
        waypoints.append(start + diff * (i / n_steps))
    return waypoints


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpt", type=str, required=True,
                        help="Path to trained Diffusion Policy checkpoint")
    parser.add_argument("--can-port", type=str, default="can0")
    parser.add_argument("--velocity-pct", type=int, default=25)
    parser.add_argument("--hz", type=float, default=30.0,
                        help="Control loop frequency.")
    parser.add_argument("--approach-steps", type=int, default=APPROACH_STEPS_DEFAULT)
    parser.add_argument("--test-mode", choices=("A", "B", "C", "D"), default="A",
                        help="A: approach only. B: approach+close+lift. C: approach+descend. D: full grasp.")
    parser.add_argument("--descend-j2-delta", type=float, default=0.04,
                        help="J2 increment for test mode C.")
    parser.add_argument("--place-j1-offset", type=float, default=0.30,
                        help="J1 offset for place phase in test mode D.")
    parser.add_argument("--action-smooth", type=float, default=ACTION_SMOOTH_ALPHA)
    parser.add_argument("--no-global", action="store_true",
                        help="Disable global camera.")
    parser.add_argument("--global-camera", type=str, default="auto")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview without sending robot commands.")
    parser.add_argument("--debug-actions", action="store_true",
                        help="Print action details at debug-every steps.")
    parser.add_argument("--debug-every", type=int, default=10)
    parser.add_argument("--no-gui", action="store_true",
                        help="Use terminal input instead of GUI.")
    parser.add_argument("--save-rollout", action="store_true",
                        help="Save rollout frames to disk.")
    parser.add_argument("--no-return-to-start", action="store_true")
    parser.add_argument("--wrist-freeze-j2", type=float, default=WRIST_FREEZE_J2)
    parser.add_argument("--ready-j2", type=float, default=READY_J2)
    parser.add_argument("--ready-count-min", type=int, default=READY_COUNT_MIN)
    args = parser.parse_args()

    WRIST_FREEZE_J2_ = args.wrist_freeze_j2
    READY_J2_ = args.ready_j2
    READY_COUNT_MIN_ = args.ready_count_min

    print("=" * 60)
    print("  Piper Diffusion Policy — Bottle Grasp")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # --- Load Diffusion Policy ---
    print(f"\n[1/4] Loading Diffusion Policy from {args.checkpt} ...")
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    policy = DiffusionPolicy.from_pretrained(args.checkpt)
    policy.to(device)
    policy.eval()
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"  Policy loaded: {n_params:,} params")
    print(f"  horizon={policy.config.horizon}  n_action_steps={policy.config.n_action_steps}  "
          f"n_obs_steps={policy.config.n_obs_steps}")

    print("\n[2/4] Loading pre/post processors ...")
    preprocessor, postprocessor = load_policy_processors(policy, args.checkpt, device)
    print("  Processors ready.")

    # --- Connect robot ---
    print(f"\n[3/4] Connecting Piper ({args.can_port}) ...")
    robot = PiperRobot(can_port=args.can_port, disable_torque_on_disconnect=False)
    robot.connect()
    print("  Robot connected and enabled.")

    # --- Init cameras ---
    print("\n[4/4] Initializing cameras ...")
    rs_serials = find_realsense_devices()
    wrist_serial = rs_serials[0] if rs_serials else ""
    wrist_cam = RealSenseCamera(serial=wrist_serial, width=640, height=480, fps=30, enable_depth=False)

    global_cam = None
    requires_global = "observation.images.global_rgb" in policy.config.input_features
    if args.no_global and requires_global:
        raise ValueError("Policy requires global_rgb; cannot use --no-global.")
    if not args.no_global:
        try:
            global_cam = USBCamera(device_id=args.global_camera, width=640, height=480, fps=30)
        except IOError as e:
            if requires_global:
                raise
            print(f"  Global camera skipped: {e}")
    print("  Cameras ready.")

    print("\n" + "-" * 60)
    print("  SPACE = run approach    Q/ESC = quit")
    print(f"  TEST MODE: {args.test_mode}  |  STEPS: {args.approach_steps}  |  SMOOTH: α={args.action_smooth}")
    print(f"  Per-joint max_delta: J1-J3={MAX_DELTA_PER_JOINT[0]:.3f}  J4-J6={MAX_DELTA_PER_JOINT[3]:.3f}")
    print(f"  Wrist freeze @ J2 > {WRIST_FREEZE_J2_:.2f}  |  Ready stop @ J2 > {READY_J2_:.2f} ×{READY_COUNT_MIN_}")
    if args.dry_run:
        print("  DRY RUN: robot commands will not be sent.")
    print("-" * 60 + "\n")

    try:
        while True:
            # --- Wait for trigger ---
            if args.no_gui:
                cmd = input("  Press ENTER to run, Q then ENTER to quit: ").strip().lower()
                if cmd == "q":
                    break
            else:
                wrist_frame = wrist_cam.read()
                global_frame = global_cam.read() if global_cam else None
                preview = build_preview(wrist_frame, global_frame, "READY - SPACE run")
                cv2.imshow("Diffusion Policy Deploy", preview)
                key = cv2.waitKey(1) & 0xFF
                if should_quit(key):
                    break
                if key != ord(' '):
                    continue

            # ============================================================
            #  APPROACH PHASE (Diffusion Policy)
            # ============================================================
            print(f"  >>> Approach ({args.test_mode}, {args.approach_steps} steps) ...")

            rollout_dir = None
            if args.save_rollout:
                import datetime
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                rollout_dir = PROJECT_ROOT / "logs" / "rollouts" / ts
                rollout_dir.mkdir(parents=True, exist_ok=True)
                print(f"  Saving rollout to {rollout_dir}")

            start_robot_state = np.asarray(robot.get_joint_positions(), dtype=np.float32)

            # Reset policy state between trajectories
            policy.reset()
            preprocessor.reset()
            postprocessor.reset()

            last_smoothed = None
            last_state = None
            raw_actions = []
            paused = False
            user_quit = False
            stagnation_count = 0
            ready_count = 0
            stop_reason = "completed"

            for step in range(args.approach_steps):
                loop_start = time.time()

                # Capture observation
                wrist_frame = wrist_cam.read()
                global_frame = global_cam.read() if global_cam else None
                robot_state = robot.get_joint_positions()

                # Build observation and run inference
                obs = prepare_observation(robot_state, wrist_frame.rgb,
                                          global_frame.rgb if global_frame else None, device)

                with torch.inference_mode():
                    normalized_obs = preprocessor(obs)
                    action = policy.select_action(normalized_obs)
                    action = postprocessor(action)

                if action.dim() == 2:
                    action = action.squeeze(0)
                model_action = action.cpu().numpy()
                raw_actions.append(model_action.copy())

                robot_state_arr = np.asarray(robot_state, dtype=np.float32)

                # ── Per-joint delta clamp ──
                raw_delta = model_action - robot_state_arr
                for j in range(6):
                    raw_delta[j] = np.clip(raw_delta[j], -MAX_DELTA_PER_JOINT[j], MAX_DELTA_PER_JOINT[j])
                clipped = robot_state_arr + raw_delta
                clipped[6] = GRIPPER_OPEN

                # ── Wrist freeze ──
                wrist_frozen = False
                if robot_state_arr[1] > WRIST_FREEZE_J2_:
                    clipped[3:6] = robot_state_arr[3:6]
                    wrist_frozen = True

                # ── EMA smoothing ──
                alpha = args.action_smooth
                if last_smoothed is not None and alpha > 0:
                    smoothed_arm = alpha * clipped[:6] + (1.0 - alpha) * last_smoothed
                else:
                    smoothed_arm = clipped[:6].copy()
                sent_target = np.concatenate([smoothed_arm, [GRIPPER_OPEN]])

                # ── Safety clamp ──
                sent_target[:6] = np.clip(sent_target[:6], -3.14, 3.14)
                sent_target[6] = np.clip(sent_target[6], 0.0, PIPER_GRIPPER_MAX_M)

                # ── Ready stop ──
                if robot_state_arr[1] > READY_J2_ and step > 160:
                    ready_count += 1
                else:
                    ready_count = 0
                stop_act = (ready_count >= READY_COUNT_MIN_) or (step + 1 >= args.approach_steps)

                # ── Debug ──
                if args.debug_actions and (
                    step == 0 or step == args.approach_steps - 1 or (step + 1) % args.debug_every == 0
                    or wrist_frozen or stop_act
                ):
                    print(f"  --- step {step+1:03d}/{args.approach_steps} ---")
                    print(f"    robot_state : {fmt_vec(robot_state_arr)}")
                    print(f"    model_action: {fmt_vec(model_action)}")
                    print(f"    clipped     : {fmt_vec(clipped)}")
                    print(f"    sent_target : {fmt_vec(sent_target)}")
                    delta = sent_target - robot_state_arr
                    print(f"    delta       : {fmt_vec(delta)}")
                    print(f"    J2={robot_state_arr[1]:.4f}  wrist_frozen={wrist_frozen}  "
                          f"ready={ready_count}/{READY_COUNT_MIN_}  stop={stop_act}")

                # ── Safety: joint limit ──
                if np.any(np.abs(sent_target[:6]) > 3.0):
                    print(f"\n  [HALT] Joint limit: {fmt_vec(sent_target)}")
                    stop_reason = "joint_limit"
                    break

                # ── Safety: stagnation ──
                state_diff = max_abs_diff(robot_state, last_state)
                near_end = step > args.approach_steps * 0.7
                if not near_end and last_state is not None and state_diff < STAGNATION_THRESHOLD:
                    stagnation_count += 1
                else:
                    stagnation_count = 0
                if stagnation_count >= STAGNATION_STEPS:
                    print(f"\n  [HALT] Stagnation: {STAGNATION_STEPS} steps"
                          f" with diff < {STAGNATION_THRESHOLD}")
                    stop_reason = "stagnation"
                    break

                # ── Ready stop → send and break ──
                if stop_act:
                    stop_reason = "ready" if ready_count >= READY_COUNT_MIN_ else "max_steps"
                    if not args.dry_run:
                        robot.set_joint_positions(sent_target.tolist(), velocity_pct=args.velocity_pct)
                    last_smoothed = smoothed_arm.copy()
                    last_state = robot_state_arr.copy()
                    print(f"\n  [STOP] Approach done ({stop_reason})  J2={robot_state_arr[1]:.4f}  "
                          f"step={step+1}")
                    break

                # ── Save rollout ──
                if rollout_dir is not None and (step % 5 == 0 or step < 5):
                    np.savez_compressed(
                        rollout_dir / f"step_{step:04d}.npz",
                        robot_state=robot_state_arr.copy(),
                        model_action=model_action.copy(),
                        sent_target=sent_target.copy(),
                    )
                    if step % 20 == 0:
                        cv2.imwrite(str(rollout_dir / f"wrist_{step:04d}.jpg"),
                                    cv2.cvtColor(wrist_frame.rgb, cv2.COLOR_RGB2BGR))

                # ── Send to robot ──
                if not args.dry_run:
                    robot.set_joint_positions(sent_target.tolist(), velocity_pct=args.velocity_pct)

                # ── Preview + keyboard ──
                if not args.no_gui:
                    label = f"PAUSED {step+1}/{args.approach_steps}" if paused else f"APPROACH {step+1}/{args.approach_steps}"
                    color = (0, 165, 255) if paused else (0, 0, 255)
                    preview = build_preview(wrist_frame, global_frame, label, color=color)
                    cv2.imshow("Diffusion Policy Deploy", preview)
                    key = cv2.waitKey(1) & 0xFF
                    if should_quit(key):
                        user_quit = True
                        stop_reason = "user_quit"
                        break
                    if key == ord(' '):
                        paused = not paused
                        if paused:
                            print("  PAUSED — SPACE resume, Q quit")
                        else:
                            print("  RESUMED")

                # ── Timing ──
                elapsed = time.time() - loop_start
                step_time = 1.0 / args.hz
                if elapsed < step_time:
                    time.sleep(step_time - elapsed)

                # ── Pause loop ──
                while paused:
                    if not args.no_gui:
                        preview = build_preview(wrist_frame, global_frame,
                                                f"PAUSED {step+1}/{args.approach_steps}", color=(0, 165, 255))
                        cv2.imshow("Diffusion Policy Deploy", preview)
                    if last_smoothed is not None and not args.dry_run:
                        hold_pos = np.concatenate([last_smoothed, [GRIPPER_OPEN]])
                        robot.set_joint_positions(hold_pos.tolist(), velocity_pct=args.velocity_pct)
                    time.sleep(1.0 / args.hz)
                    if not args.no_gui:
                        key = cv2.waitKey(1) & 0xFF
                        if should_quit(key):
                            paused = False
                            user_quit = True
                            stop_reason = "user_quit"
                            break
                        if key == ord(' '):
                            paused = False
                            print("  RESUMED")
                    else:
                        import select
                        if select.select([sys.stdin], [], [], 0.1)[0]:
                            line = sys.stdin.readline().strip().lower()
                            if line == 'q':
                                paused = False
                                user_quit = True
                                stop_reason = "user_quit"
                                break
                            if line == '':
                                paused = False
                                print("  RESUMED")
                if user_quit:
                    break

                last_smoothed = smoothed_arm.copy()
                last_state = robot_state_arr.copy()

            # ── Action stats ──
            if raw_actions:
                ra = np.array(raw_actions)
                jnames = ["J1", "J2", "J3", "J4", "J5", "J6", "Grip"]
                print(f"\n  Action stats ({len(raw_actions)} steps):")
                print(f"  {'Dim':>6}  {'mean':>12}  {'min':>12}  {'max':>12}")
                for d in range(ra.shape[1]):
                    print(f"  {jnames[d]:>6}  {ra[:, d].mean():12.6f}  "
                          f"{ra[:, d].min():12.6f}  {ra[:, d].max():12.6f}")

            print(f"\n  Approach finished ({stop_reason}, {len(raw_actions)} steps).")

            # ============================================================
            #  HANDOVER: hold position
            # ============================================================
            if not user_quit and not args.dry_run:
                print("  Hold position (0.3s) ...")
                cur = robot.get_joint_positions()
                hold_start = time.time()
                while time.time() - hold_start < 0.3:
                    robot.set_joint_positions(cur, velocity_pct=args.velocity_pct)
                    time.sleep(1.0 / args.hz)

            # ============================================================
            #  TEST MODE A: approach only — done
            # ============================================================
            if args.test_mode == "A":
                if not user_quit:
                    final_state = robot.get_joint_positions()
                    print(f"  [TEST-A] Done. Gripper open. Final J2 = {final_state[1]:.5f} rad")

            # ============================================================
            #  TEST MODE C: approach → descend (no close)
            # ============================================================
            if args.test_mode == "C" and not user_quit:
                print(f"  [TEST-C] Descending J2 += {args.descend_j2_delta:.3f} ...")
                cur = np.asarray(robot.get_joint_positions(), dtype=np.float32)
                desc = cur.copy()
                desc[1] = np.clip(desc[1] + args.descend_j2_delta, -3.14, 3.14)
                desc[6] = GRIPPER_OPEN
                path = interpolate_joint_path(cur, desc, max_step_rad=0.015, max_step_gripper=0.002)
                for di, dt in enumerate(path):
                    dt[6] = GRIPPER_OPEN
                    if not args.dry_run:
                        robot.set_joint_positions(dt.tolist(), velocity_pct=args.velocity_pct)
                    if di == 0 or di == len(path) - 1:
                        print(f"    descend {di+1:3d}/{len(path):3d}  {fmt_vec(dt, 3)}")
                    time.sleep(1.0 / args.hz)
                final = robot.get_joint_positions()
                print(f"  Descent done. Final J2 = {final[1]:.5f} rad")

            # ============================================================
            #  TEST MODE B: close + lift
            # ============================================================
            if args.test_mode == "B" and not user_quit:
                # Close gripper
                print(f"  [TEST-B] Closing gripper ({GRIPPER_CLOSE:.3f}m) ...")
                cur = np.asarray(robot.get_joint_positions(), dtype=np.float32)
                close = cur.copy()
                close[6] = GRIPPER_CLOSE
                path = interpolate_joint_path(cur, close, max_step_rad=0.02, max_step_gripper=0.002)
                for ci, ct in enumerate(path):
                    if not args.dry_run:
                        robot.set_joint_positions(ct.tolist(), velocity_pct=args.velocity_pct)
                time.sleep(0.6)
                print("  Closed.")

                # Lift
                print("  [TEST-B] Lifting J3 -= 0.06 ...")
                cur = np.asarray(robot.get_joint_positions(), dtype=np.float32)
                lift = cur.copy()
                lift[2] = np.clip(lift[2] - 0.06, -3.14, 3.14)
                lift[6] = GRIPPER_CLOSE
                path = interpolate_joint_path(cur, lift, max_step_rad=0.02, max_step_gripper=0.002)
                for li, lt in enumerate(path):
                    if not args.dry_run:
                        robot.set_joint_positions(lt.tolist(), velocity_pct=args.velocity_pct)
                time.sleep(0.5)
                print("  Lift done.")

            # ============================================================
            #  TEST MODE D: full grasp + place + release
            # ============================================================
            if args.test_mode == "D" and not user_quit:
                # Close
                print(f"  [TEST-D] Closing gripper ({GRIPPER_CLOSE:.3f}m) ...")
                cur = np.asarray(robot.get_joint_positions(), dtype=np.float32)
                close = cur.copy()
                close[6] = GRIPPER_CLOSE
                path = interpolate_joint_path(cur, close, max_step_rad=0.02, max_step_gripper=0.002)
                for ci, ct in enumerate(path):
                    if not args.dry_run:
                        robot.set_joint_positions(ct.tolist(), velocity_pct=args.velocity_pct)
                time.sleep(0.6)

                # Lift
                print("  [TEST-D] Lifting J3 -= 0.06 ...")
                cur = np.asarray(robot.get_joint_positions(), dtype=np.float32)
                lift = cur.copy()
                lift[2] = np.clip(lift[2] - 0.06, -3.14, 3.14)
                lift[6] = GRIPPER_CLOSE
                path = interpolate_joint_path(cur, lift, max_step_rad=0.02, max_step_gripper=0.002)
                for li, lt in enumerate(path):
                    if not args.dry_run:
                        robot.set_joint_positions(lt.tolist(), velocity_pct=args.velocity_pct)
                time.sleep(0.5)

                # Place
                print(f"  [TEST-D] Placing J1 += {args.place_j1_offset:.2f} ...")
                cur = np.asarray(robot.get_joint_positions(), dtype=np.float32)
                place = cur.copy()
                place[0] = np.clip(place[0] + args.place_j1_offset, -3.14, 3.14)
                place[6] = GRIPPER_CLOSE
                path = interpolate_joint_path(cur, place, max_step_rad=0.03, max_step_gripper=0.002)
                for pi, pt in enumerate(path):
                    if not args.dry_run:
                        robot.set_joint_positions(pt.tolist(), velocity_pct=args.velocity_pct)

                # Release
                print(f"  [TEST-D] Releasing gripper ({GRIPPER_OPEN:.3f}m) ...")
                cur = np.asarray(robot.get_joint_positions(), dtype=np.float32)
                release = cur.copy()
                release[6] = GRIPPER_OPEN
                path = interpolate_joint_path(cur, release, max_step_rad=0.02, max_step_gripper=0.004)
                for ri, rt in enumerate(path):
                    if not args.dry_run:
                        robot.set_joint_positions(rt.tolist(), velocity_pct=args.velocity_pct)
                time.sleep(0.5)
                print("  [TEST-D] Full grasp + place + release done.")

            # ============================================================
            #  RETURN TO START
            # ============================================================
            if not args.no_return_to_start and not user_quit and not args.dry_run:
                print("  Returning to start ...")
                cur = np.asarray(robot.get_joint_positions(), dtype=np.float32)
                target = start_robot_state.copy()
                target[6] = cur[6]
                path = interpolate_joint_path(cur, target, max_step_rad=0.03, max_step_gripper=0.004)
                for ri, rt in enumerate(path):
                    rt[:6] = np.clip(rt[:6], -3.14, 3.14)
                    rt[6] = np.clip(rt[6], 0.0, PIPER_GRIPPER_MAX_M)
                    if not args.dry_run:
                        robot.set_joint_positions(rt.tolist(), velocity_pct=args.velocity_pct)
                    elapsed = time.time()
                    s_time = 1.0 / args.hz
                    if elapsed < s_time:
                        time.sleep(s_time - elapsed)
                print("  Returned.")

            print("  Trajectory complete.\n")

    except KeyboardInterrupt:
        print("\n  Interrupted.")
    finally:
        try:
            cur = robot.get_joint_positions()
            robot.set_joint_positions(cur, velocity_pct=50)
        except Exception:
            pass
        print("  Stopped. Arm stays ENABLED.")
        wrist_cam.close()
        if global_cam:
            global_cam.close()
        if not args.no_gui:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
