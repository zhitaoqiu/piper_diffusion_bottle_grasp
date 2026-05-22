#!/usr/bin/env python3
"""
Deploy Diffusion Policy on Piper arm (follower) for bottle pick & place aside.

Usage:
  conda activate piper_act
  python3 inference/deploy.py \
    --checkpt outputs/train/piper_bottle_pick_place_aside/checkpoints/last/pretrained_model

Controls:
  SPACE  = run one grasp episode
  Q/ESC  = quit

Hardware: only the follower arm is connected (can0). The Diffusion Policy
directly outputs joint targets for the follower.
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
MAX_DELTA_PER_JOINT = np.array([0.03, 0.03, 0.03, 0.012, 0.012, 0.012], dtype=np.float32)
ACTION_SMOOTH_ALPHA = 0.5
MAX_STEPS_DEFAULT = 200
WRIST_FREEZE_J2 = 1.45
READY_J2 = 1.50
READY_COUNT_MIN = 5
STAGNATION_STEPS = 20
STAGNATION_THRESHOLD = 0.0008
WRIST_IMAGE_KEY = "observation.images.wrist_rgb"
GLOBAL_IMAGE_KEY = "observation.images.global_rgb"
STATE_KEY = "observation.state"


def load_policy_processors(policy, checkpt: str, device: torch.device):
    from lerobot.policies.factory import make_pre_post_processors

    device_name = str(device)
    pre = {"device_processor": {"device": device_name}, "normalizer_processor": {"device": device_name}}
    post = {"unnormalizer_processor": {"device": device.type}, "device_processor": {"device": "cpu"}}
    return make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=checkpt,
        preprocessor_overrides=pre, postprocessor_overrides=post,
    )


def feature_shape(feature):
    if hasattr(feature, "shape"):
        return tuple(feature.shape)
    return tuple(feature["shape"])


def policy_input_features(policy):
    return getattr(policy.config, "input_features", {}) or {}


def required_image_inputs(policy):
    features = policy_input_features(policy)
    return {
        "wrist": WRIST_IMAGE_KEY in features,
        "global": GLOBAL_IMAGE_KEY in features,
    }


def image_to_tensor(image, device):
    return torch.from_numpy(image).float().div(255.0).permute(2, 0, 1).unsqueeze(0).to(device)


def prepare_observation(state, wrist_img, global_img, device, image_inputs):
    obs = {}
    obs[STATE_KEY] = torch.from_numpy(
        np.asarray(state, dtype=np.float32)
    ).unsqueeze(0).to(device)

    if image_inputs["wrist"]:
        if wrist_img is None:
            raise RuntimeError("Policy requires wrist_rgb, but wrist camera/frame is unavailable.")
        obs[WRIST_IMAGE_KEY] = image_to_tensor(wrist_img, device)

    if image_inputs["global"]:
        if global_img is None:
            raise RuntimeError("Policy requires global_rgb, but global camera/frame is unavailable.")
        obs[GLOBAL_IMAGE_KEY] = image_to_tensor(global_img, device)

    return obs


def make_zero_observation(policy, device):
    obs = {}
    for key, feature in policy_input_features(policy).items():
        shape = feature_shape(feature)
        obs[key] = torch.zeros((1, *shape), dtype=torch.float32, device=device)
    return obs


def warm_up_policy(policy, preprocessor, postprocessor, device, steps: int):
    if steps <= 0:
        return
    print(f"  Warming up policy ({steps} step{'s' if steps != 1 else ''}) ...")
    dummy_obs = make_zero_observation(policy, device)
    with torch.inference_mode():
        for _ in range(steps):
            normalized_obs = preprocessor(dummy_obs)
            action = policy.select_action(normalized_obs)
            _ = postprocessor(action)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()
    print("  Warmup complete.")


def resolve_device(device_arg: str, allow_cpu: bool) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if allow_cpu:
            return torch.device("cpu")
        raise RuntimeError(
            "CUDA is not available, and CPU inference is disabled to avoid freezing the machine. "
            "Fix the NVIDIA driver/CUDA runtime, or pass --allow-cpu --device cpu for a slow dry-run/debug run."
        )

    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device_arg}, but torch.cuda.is_available() is false.")
    if device.type == "cpu" and not allow_cpu:
        raise RuntimeError("CPU inference is disabled by default. Pass --allow-cpu --device cpu for debugging only.")
    return device


def configure_torch_runtime(torch_threads: int, device: torch.device):
    if torch_threads > 0:
        torch.set_num_threads(torch_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True


def frame_image(frame):
    return frame.rgb if frame is not None else None


def build_preview(wrist_frame, global_frame, text: str, color=(0, 255, 0)):
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

    cv2.putText(preview, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
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
    parser.add_argument("--hz", type=float, default=10.0,
                        help="Control loop frequency. Keep this close to the dataset FPS.")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS_DEFAULT,
                        help="Max steps per episode.")
    parser.add_argument("--action-smooth", type=float, default=ACTION_SMOOTH_ALPHA)
    parser.add_argument("--device", type=str, default="auto",
                        help="Inference device: auto, cuda, cuda:0, or cpu.")
    parser.add_argument("--allow-cpu", action="store_true",
                        help="Allow CPU inference. Slow; intended for dry-run/debug only.")
    parser.add_argument("--torch-threads", type=int, default=2,
                        help="CPU threads used by PyTorch. Keeps CPU fallback/debug runs responsive.")
    parser.add_argument("--num-inference-steps", type=int, default=None,
                        help="Override diffusion sampling steps at deploy time, e.g. 16 for faster smoke tests.")
    parser.add_argument("--warmup-steps", type=int, default=1,
                        help="Run dummy inference before connecting cameras/robot to catch OOM/slow init early.")
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

    if args.hz <= 0:
        parser.error("--hz must be > 0.")
    if args.num_inference_steps is not None and args.num_inference_steps <= 0:
        parser.error("--num-inference-steps must be > 0.")

    try:
        device = resolve_device(args.device, args.allow_cpu)
    except RuntimeError as exc:
        parser.exit(2, f"error: {exc}\n")
    configure_torch_runtime(args.torch_threads, device)
    print(f"  Device: {device}")
    print(f"  Torch threads: {torch.get_num_threads()}")

    # --- Load Diffusion Policy ---
    print(f"\n[1/4] Loading Diffusion Policy from {args.checkpt} ...")
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    policy = DiffusionPolicy.from_pretrained(args.checkpt)
    policy.to(device)
    policy.eval()
    if args.num_inference_steps is not None:
        policy.diffusion.num_inference_steps = args.num_inference_steps
        policy.config.num_inference_steps = args.num_inference_steps
    elif device.type == "cpu" and args.allow_cpu:
        policy.diffusion.num_inference_steps = min(policy.diffusion.num_inference_steps, 8)
        policy.config.num_inference_steps = policy.diffusion.num_inference_steps
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"  Policy loaded: {n_params:,} params")
    print(f"  horizon={policy.config.horizon}  n_action_steps={policy.config.n_action_steps}  "
          f"n_obs_steps={policy.config.n_obs_steps}")
    print(f"  diffusion inference steps={policy.diffusion.num_inference_steps}")
    image_inputs = required_image_inputs(policy)
    print(f"  image inputs: wrist={image_inputs['wrist']}  global={image_inputs['global']}")

    print("\n[2/4] Loading pre/post processors ...")
    preprocessor, postprocessor = load_policy_processors(policy, args.checkpt, device)
    print("  Processors ready.")
    warm_up_policy(policy, preprocessor, postprocessor, device, args.warmup_steps)

    # --- Connect robot ---
    print(f"\n[3/4] Connecting Piper ({args.can_port}) ...")
    robot = PiperRobot(can_port=args.can_port, disable_torque_on_disconnect=False)
    robot.connect()
    print("  Robot connected and enabled.")

    # --- Init cameras ---
    print("\n[4/4] Initializing cameras ...")
    wrist_cam = None
    if image_inputs["wrist"]:
        rs_serials = find_realsense_devices()
        wrist_serial = rs_serials[0] if rs_serials else ""
        wrist_cam = RealSenseCamera(serial=wrist_serial, width=640, height=480, fps=30, enable_depth=False)
    else:
        print("  Wrist camera skipped: policy does not use wrist_rgb.")

    global_cam = None
    requires_global = image_inputs["global"]
    if args.no_global and requires_global:
        raise ValueError("Policy requires global_rgb; cannot use --no-global.")
    if not args.no_global and requires_global:
        try:
            global_cam = USBCamera(device_id=args.global_camera, width=640, height=480, fps=30)
        except IOError as e:
            raise
    elif not requires_global:
        print("  Global camera skipped: policy does not use global_rgb.")
    print("  Cameras ready.")

    print("\n" + "-" * 60)
    print("  SPACE = run episode    Q/ESC = quit")
    print(f"  MAX STEPS: {args.max_steps}  |  SMOOTH: α={args.action_smooth}")
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
                wrist_frame = wrist_cam.read() if wrist_cam else None
                global_frame = global_cam.read() if global_cam else None
                preview = build_preview(wrist_frame, global_frame, "READY - SPACE run")
                cv2.imshow("Diffusion Policy Deploy", preview)
                key = cv2.waitKey(1) & 0xFF
                if should_quit(key):
                    break
                if key != ord(' '):
                    continue

            # ============================================================
            #  Diffusion Policy inference loop
            # ============================================================
            print(f"  >>> Episode start ({args.max_steps} max steps) ...")

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

            for step in range(args.max_steps):
                loop_start = time.time()

                # Capture observation
                wrist_frame = wrist_cam.read() if wrist_cam else None
                global_frame = global_cam.read() if global_cam else None
                robot_state = robot.get_joint_positions()

                # Build observation and run inference
                obs = prepare_observation(
                    robot_state,
                    frame_image(wrist_frame),
                    frame_image(global_frame),
                    device,
                    image_inputs,
                )

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
                if robot_state_arr[1] > READY_J2_ and step > args.max_steps * 0.65:
                    ready_count += 1
                else:
                    ready_count = 0
                stop_act = (ready_count >= READY_COUNT_MIN_) or (step + 1 >= args.max_steps)

                # ── Debug ──
                if args.debug_actions and (
                    step == 0 or step == args.max_steps - 1 or (step + 1) % args.debug_every == 0
                    or wrist_frozen or stop_act
                ):
                    print(f"  --- step {step+1:03d}/{args.max_steps} ---")
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
                near_end = step > args.max_steps * 0.7
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
                    print(f"\n  [STOP] Episode done ({stop_reason})  J2={robot_state_arr[1]:.4f}  "
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
                        if wrist_frame is not None:
                            cv2.imwrite(str(rollout_dir / f"wrist_{step:04d}.jpg"),
                                        cv2.cvtColor(wrist_frame.rgb, cv2.COLOR_RGB2BGR))
                        if global_frame is not None:
                            cv2.imwrite(str(rollout_dir / f"global_{step:04d}.jpg"),
                                        cv2.cvtColor(global_frame.rgb, cv2.COLOR_RGB2BGR))

                # ── Send to robot ──
                if not args.dry_run:
                    robot.set_joint_positions(sent_target.tolist(), velocity_pct=args.velocity_pct)

                # ── Preview + keyboard ──
                if not args.no_gui:
                    label = f"PAUSED {step+1}/{args.max_steps}" if paused else f"INFERENCE {step+1}/{args.max_steps}"
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
                                                f"PAUSED {step+1}/{args.max_steps}", color=(0, 165, 255))
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

            print(f"\n  Episode finished ({stop_reason}, {len(raw_actions)} steps).")

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
                    return_step_start = time.time()
                    rt[:6] = np.clip(rt[:6], -3.14, 3.14)
                    rt[6] = np.clip(rt[6], 0.0, PIPER_GRIPPER_MAX_M)
                    if not args.dry_run:
                        robot.set_joint_positions(rt.tolist(), velocity_pct=args.velocity_pct)
                    elapsed = time.time() - return_step_start
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
        if wrist_cam:
            wrist_cam.close()
        if global_cam:
            global_cam.close()
        if not args.no_gui:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
