#!/usr/bin/env python3
"""Deploy a Diffusion Policy for push-like manipulation tasks.

Gripper mode design
-------------------
--gripper-mode policy       Use the policy's 7th action dimension.
--gripper-mode fixed_open   Force gripper to the configured open value.
--gripper-mode fixed_value  Force gripper to --gripper-fixed-value.
--gripper-mode hold_initial Read the initial gripper width and hold it
                            throughout the trajectory (DEFAULT).

Default rationale: hold_initial does not assume open/closed, making it
safe for push-like tasks whose physical gripper strategy is undecided.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from piper_control.piper_bus import PiperMotorsBus, PiperMotorsBusConfig
from piper_control.reset import interpolate_qpos_path
from piper_control.schema import (
    GRIPPER_OPEN_M,
    PIPER_GRIPPER_MAX_M,
    QposTolerance,
    STANDARD_START_QPOS,
    STATE_DIM,
    as_qpos,
)
from piper_control.start_pose import describe_guard_result, start_pose_guard
from camera.rs_camera import RealSenseCamera, USBCamera, find_realsense_devices

# Reuse task-agnostic utility functions from the existing deploy.py.
# These are pure helpers with no grasp-specific logic.
from inference.deploy import (
    add_temporal_predictions,
    build_preview,
    choose_action_chunk,
    choose_action_vector,
    clip_step_target,
    close_camera,
    configure_torch_runtime,
    ease_alpha,
    ease_derivative,
    feature_shape,
    fmt_vec,
    frame_rgb,
    GlobalVideoRecorder,
    guard_passes,
    image_to_tensor,
    load_policy_processors,
    load_start_pose,
    make_fixed_diffusion_noise,
    make_zero_observation,
    policy_input_features,
    prepare_observation,
    prune_temporal_predictions,
    required_image_inputs,
    reseed_sampling,
    resolve_device,
    select_action_chunk,
    should_quit,
    should_replan_actions,
    smooth_action_chunk,
    stream_servo_segment,
    temporal_ensemble_action,
    warm_up_policy,
    write_servo_qpos,
)

WRIST_IMAGE_KEY = "observation.images.wrist_rgb"
GLOBAL_IMAGE_KEY = "observation.images.global_rgb"
STATE_KEY = "observation.state"

GRIPPER_MODES = ("policy", "fixed_open", "fixed_value", "hold_initial")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Deploy Diffusion Policy for push-like tasks."
    )
    # --- Required ---
    parser.add_argument("--checkpt", required=True)

    # --- Robot ---
    parser.add_argument("--can-port", default="can0")
    parser.add_argument("--velocity-pct", type=int, default=25)
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--servo-hz", type=float, default=50.0)
    parser.add_argument("--servo-ease", choices=("linear", "smoothstep", "cosine"),
                        default="smoothstep")
    parser.add_argument("--max-steps", type=int, default=320)

    # --- Safety ---
    parser.add_argument("--max-delta-arm", type=float, default=0.030)
    parser.add_argument("--max-delta-wrist", type=float, default=0.012)
    parser.add_argument("--max-gripper-step", type=float, default=0.004)
    parser.add_argument("--action-smooth", type=float, default=0.5,
                        help="EMA weight on arm target [0,1]. 0 = no smoothing.")
    parser.add_argument("--stop-noop-steps", type=int, default=40,
                        help="Stop after N nearly-stationary commands. 0 disables.")

    # --- Gripper mode ---
    parser.add_argument("--gripper-mode", choices=GRIPPER_MODES,
                        default="hold_initial",
                        help="How gripper targets are produced. "
                             "policy: use the policy's 7th dim. "
                             "fixed_open: gripper open value. "
                             "fixed_value: --gripper-fixed-value. "
                             "hold_initial: freeze the initial gripper reading.")
    parser.add_argument("--gripper-fixed-value", type=float, default=GRIPPER_OPEN_M,
                        help="Gripper target (metres) when --gripper-mode fixed_value.")

    # --- Cameras ---
    parser.add_argument("--global-camera", default="auto")
    parser.add_argument("--wrist-serial", default="")

    # --- Device ---
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--torch-threads", type=int, default=2)

    # --- Inference ---
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=1)

    # --- Execution mode ---
    parser.add_argument("--exec-mode", choices=("step", "chunk"), default="step")
    parser.add_argument("--chunk-exec-steps", type=int, default=0)
    parser.add_argument("--chunk-smooth-window", type=int, default=3)
    parser.add_argument("--chunk-blend-steps", type=int, default=2)
    parser.add_argument("--chunk-temporal-ensemble", action="store_true")
    parser.add_argument("--ensemble-every", type=int, default=4)
    parser.add_argument("--ensemble-decay", type=float, default=0.35)
    parser.add_argument("--replan-every-step", action="store_true")
    parser.add_argument("--replan-interval", type=int, default=0)
    parser.add_argument("--fixed-noise-seed", type=int, default=None)

    # --- Optional post-trajectory retreat ---
    parser.add_argument("--retreat-after-trajectory", action="store_true",
                        default=False,
                        help="Move arm back to start pose after trajectory. "
                             "Gripper is preserved as-is during retreat.")
    parser.add_argument("--retreat-velocity-pct", type=int, default=20)
    parser.add_argument("--retreat-hz", type=float, default=30.0)
    parser.add_argument("--retreat-max-arm-step", type=float, default=0.02)
    parser.add_argument("--retreat-max-gripper-step", type=float, default=0.003)

    # --- Debug / output ---
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument("--save-global-video", action="store_true")
    parser.add_argument("--global-video-dir", type=Path,
                        default=PROJECT_ROOT / "logs" / "videos")
    parser.add_argument("--global-video-fps", type=float, default=20.0)
    parser.add_argument("--debug-actions", action="store_true")
    parser.add_argument("--debug-every", type=int, default=10)

    # --- Start pose guard ---
    parser.add_argument("--start-pose-file", type=Path,
                        default=PROJECT_ROOT / "config" / "start_pose.json")
    parser.add_argument("--start-guard-mode", choices=("strict", "zone"),
                        default="zone")
    parser.add_argument("--arm-start-tol", type=float, default=0.05)
    parser.add_argument("--gripper-start-tol", type=float, default=0.01)
    parser.add_argument("--skip-start-guard", action="store_true")

    args = parser.parse_args()

    # --- Validation ---
    if args.hz <= 0:
        parser.error("--hz must be > 0.")
    if args.servo_hz <= 0:
        parser.error("--servo-hz must be > 0.")
    if args.max_steps <= 0:
        parser.error("--max-steps must be > 0.")
    if args.max_delta_arm <= 0 or args.max_delta_wrist <= 0 or args.max_gripper_step <= 0:
        parser.error("step limits must be > 0.")
    if not 0 <= args.action_smooth <= 1:
        parser.error("--action-smooth must be in [0, 1].")
    if not 0 <= args.gripper_fixed_value <= PIPER_GRIPPER_MAX_M:
        parser.error(f"--gripper-fixed-value must be in [0, {PIPER_GRIPPER_MAX_M}].")
    if args.num_inference_steps is not None and args.num_inference_steps <= 0:
        parser.error("--num-inference-steps must be > 0.")
    if args.chunk_exec_steps < 0:
        parser.error("--chunk-exec-steps must be >= 0.")
    if args.chunk_smooth_window < 1:
        parser.error("--chunk-smooth-window must be >= 1.")
    if args.chunk_blend_steps < 0:
        parser.error("--chunk-blend-steps must be >= 0.")
    if args.ensemble_every <= 0:
        parser.error("--ensemble-every must be > 0.")
    if args.ensemble_decay < 0:
        parser.error("--ensemble-decay must be >= 0.")
    if args.replan_interval < 0:
        parser.error("--replan-interval must be >= 0.")
    if args.global_video_fps <= 0:
        parser.error("--global-video-fps must be > 0.")
    return args


# ---------------------------------------------------------------------------
#  Gripper mode
# ---------------------------------------------------------------------------

def resolve_gripper_target(policy_gripper: float, current_gripper: float,
                           mode: str, fixed_value: float,
                           initial_gripper: float) -> float:
    """Return the gripper target value for the given mode.

    Args:
        policy_gripper:  The 7th dim from the policy action (ignored for
                         non-policy modes).
        current_gripper: Current measured gripper position (for clamping).
        mode:            One of GRIPPER_MODES.
        fixed_value:     Used when mode == "fixed_value".
        initial_gripper: Used when mode == "hold_initial".

    Returns:
        Gripper target in metres, clamped to [0, PIPER_GRIPPER_MAX_M].
    """
    if mode == "policy":
        value = float(policy_gripper)
    elif mode == "fixed_open":
        value = GRIPPER_OPEN_M
    elif mode == "fixed_value":
        value = float(fixed_value)
    elif mode == "hold_initial":
        value = float(initial_gripper)
    else:
        raise ValueError(f"Unknown gripper mode: {mode!r}")

    return max(0.0, min(value, PIPER_GRIPPER_MAX_M))


# ---------------------------------------------------------------------------
#  Simplified target building (no grasp gates)
# ---------------------------------------------------------------------------

def build_push_target(raw_target: np.ndarray, current: np.ndarray,
                      last_smoothed_arm: np.ndarray | None,
                      gripper_mode: str, gripper_fixed_value: float,
                      initial_gripper: float, args) -> tuple[np.ndarray, np.ndarray]:
    """Build a safe target from the raw policy output.

    Steps:
      1. Per-step delta clipping + joint-range clamping.
      2. EMA smoothing on arm joints only.
      3. Apply the configured gripper mode.

    Returns (target, next_smoothed_arm).
    """
    target = clip_step_target(raw_target, current, args)

    if last_smoothed_arm is not None and args.action_smooth > 0:
        target[:6] = (args.action_smooth * target[:6] +
                      (1.0 - args.action_smooth) * last_smoothed_arm)
    next_smoothed_arm = target[:6].copy()

    target[6] = resolve_gripper_target(
        policy_gripper=float(raw_target[6]),
        current_gripper=float(current[6]),
        mode=gripper_mode,
        fixed_value=gripper_fixed_value,
        initial_gripper=initial_gripper,
    )

    return target, next_smoothed_arm


# ---------------------------------------------------------------------------
#  Optional retreat (task-agnostic, gripper-preserving)
# ---------------------------------------------------------------------------

def run_retreat(bus, start_pose: np.ndarray, global_cam,
                video_recorder, args) -> None:
    """Move arm back to start pose, preserving current gripper width."""
    if not args.retreat_after_trajectory:
        return

    if args.dry_run:
        print("  [DRY RUN] Would retreat to start pose.")
        return

    print("\n  >>> Retreat: returning to start pose ...")
    cur = bus.read_qpos()
    target = start_pose.copy()
    target[6] = float(cur[6])  # preserve current gripper

    path = interpolate_qpos_path(
        cur, target,
        max_arm_step=np.full(6, args.retreat_max_arm_step, dtype=np.float32),
        max_gripper_step_m=args.retreat_max_gripper_step,
    )
    print(f"    current: {fmt_vec(cur)}")
    print(f"    target : {fmt_vec(target)}")
    print(f"    waypoints: {len(path)}")

    for i, qpos in enumerate(path, start=1):
        bus.write_qpos(qpos, velocity_pct=args.retreat_velocity_pct)
        if video_recorder is not None:
            video_recorder.maybe_record(global_cam)
        if i == 1 or i == len(path) or i % 10 == 0:
            print(f"    retreat {i:03d}/{len(path):03d}  {fmt_vec(qpos)}")
        time.sleep(1.0 / args.retreat_hz)

    print(f"    Retreat complete. {fmt_vec(bus.read_qpos())}")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    try:
        device = resolve_device(args.device, args.allow_cpu)
    except RuntimeError as exc:
        raise SystemExit(f"error: {exc}") from exc
    configure_torch_runtime(args.torch_threads, device)
    expected_start = load_start_pose(args.start_pose_file)

    print("=" * 72)
    print("Diffusion Policy — push-like task deploy")
    print(f"  checkpoint   : {args.checkpt}")
    print(f"  device       : {device}")
    print(f"  hz / servo   : {args.hz} / {args.servo_hz} ({args.servo_ease})")
    print(f"  exec mode    : {args.exec_mode}")
    print(f"  gripper mode : {args.gripper_mode}")
    if args.gripper_mode == "fixed_value":
        print(f"    fixed value: {args.gripper_fixed_value:.5f} m")
    print(f"  start guard  : {'SKIPPED' if args.skip_start_guard else args.start_guard_mode}")
    print(f"  dry run      : {args.dry_run}")
    print(f"  retreat      : {'on' if args.retreat_after_trajectory else 'off'}")
    print("=" * 72)

    # ---- Load policy ----
    print("\n[1/4] Loading policy ...")
    from lerobot.policies.diffusion.modeling_diffusion import ACTION, DiffusionPolicy

    policy = DiffusionPolicy.from_pretrained(args.checkpt)
    policy.to(device)
    policy.eval()
    if args.num_inference_steps is not None:
        policy.diffusion.num_inference_steps = args.num_inference_steps
        policy.config.num_inference_steps = args.num_inference_steps
    fixed_diffusion_noise = make_fixed_diffusion_noise(
        policy, device, args.fixed_noise_seed,
    )
    image_inputs = required_image_inputs(policy)
    print(
        f"  horizon={policy.config.horizon} "
        f"n_action_steps={policy.config.n_action_steps} "
        f"n_obs_steps={policy.config.n_obs_steps}"
    )
    if args.exec_mode == "chunk":
        chunk_exec_steps = args.chunk_exec_steps or policy.config.n_action_steps
        print(
            f"  chunk executor: exec_steps={chunk_exec_steps} "
            f"smooth_window={args.chunk_smooth_window} blend={args.chunk_blend_steps}"
        )
    print(f"  image inputs: wrist={image_inputs['wrist']} global={image_inputs['global']}")

    # ---- Load processors ----
    print("\n[2/4] Loading processors ...")
    preprocessor, postprocessor = load_policy_processors(
        policy, args.checkpt, device,
    )
    warm_up_policy(policy, preprocessor, postprocessor, device, args.warmup_steps)

    # ---- Connect Piper ----
    print(f"\n[3/4] Connecting Piper on {args.can_port} ...")
    bus = PiperMotorsBus(
        PiperMotorsBusConfig(
            can_port=args.can_port,
            velocity_pct=args.velocity_pct,
            disable_torque_on_disconnect=False,
        )
    )
    bus.connect()
    print(f"  qpos: {fmt_vec(bus.read_qpos())}")

    # ---- Cameras ----
    print("\n[4/4] Initializing cameras ...")
    wrist_cam = None
    global_cam = None
    if image_inputs["wrist"]:
        serials = find_realsense_devices()
        wrist_serial = args.wrist_serial or (serials[0] if serials else "")
        wrist_cam = RealSenseCamera(
            serial=wrist_serial, width=640, height=480, fps=30, enable_depth=False,
        )
    else:
        print("  Wrist camera skipped (not required by policy).")
    if image_inputs["global"]:
        global_cam = USBCamera(device_id=args.global_camera, width=640, height=480, fps=30)
    else:
        print("  Global camera skipped (not required by policy).")

    # ---- Resolve initial gripper ----
    initial_gripper: float = GRIPPER_OPEN_M  # fallback
    if args.gripper_mode == "hold_initial":
        initial_gripper = float(bus.read_qpos()[6])
        print(f"\n  Gripper hold_initial: will keep gripper at "
              f"{initial_gripper:.5f} m for the full trajectory.")
    elif args.gripper_mode == "fixed_open":
        print(f"\n  Gripper fixed_open: holding gripper at "
              f"{GRIPPER_OPEN_M:.5f} m.")
    elif args.gripper_mode == "fixed_value":
        print(f"\n  Gripper fixed_value: holding gripper at "
              f"{args.gripper_fixed_value:.5f} m.")
    elif args.gripper_mode == "policy":
        print(f"\n  Gripper policy: using policy output for gripper control.")

    print()
    print("SPACE = run one full trajectory, Q/ESC = quit")
    if args.no_gui:
        print("Terminal mode: ENTER runs one full trajectory.")
    print()

    active_video_recorder: GlobalVideoRecorder | None = None
    try:
        while True:
            # ---- Wait for trigger ----
            if args.no_gui:
                command = input(
                    "Press ENTER to run, or Q then ENTER to quit: "
                ).strip().lower()
                if command == "q":
                    break
            else:
                wrist_frame = wrist_cam.read() if wrist_cam else None
                global_frame = global_cam.read() if global_cam else None
                preview = build_preview(wrist_frame, global_frame, "READY - SPACE")
                cv2.imshow("Diffusion push deploy", preview)
                key = cv2.waitKey(1) & 0xFF
                if should_quit(key):
                    break
                if key != ord(" "):
                    continue

            if not guard_passes(bus, expected_start, args):
                print("  Move the arm back into the start zone before running.")
                continue

            # ---- Per-trajectory setup ----
            active_video_recorder = GlobalVideoRecorder(
                args.save_global_video, args.global_video_dir,
                args.global_video_fps,
            )
            active_video_recorder.start()

            policy.reset()
            preprocessor.reset()
            postprocessor.reset()
            last_smoothed_arm = None
            noop_count = 0
            stop_reason = "max_steps"
            trajectory_start = bus.read_qpos()

            # Re-read initial gripper if holding it
            if args.gripper_mode == "hold_initial":
                initial_gripper = float(trajectory_start[6])

            print(f"\n  >>> trajectory start ({args.max_steps} max steps)")

            step = 0
            chunk_id = 0
            prediction_bank: dict[int, list[tuple[int, np.ndarray]]] = {}

            while step < args.max_steps:
                # ---- Read sensors ----
                loop_start = time.time()
                wrist_frame = wrist_cam.read() if wrist_cam else None
                global_frame = global_cam.read() if global_cam else None
                active_video_recorder.write_rgb(frame_rgb(global_frame))
                current = bus.read_qpos()

                obs = prepare_observation(
                    current,
                    frame_rgb(wrist_frame),
                    frame_rgb(global_frame),
                    device,
                    image_inputs,
                )

                # ---- Policy inference ----
                if args.exec_mode == "step":
                    with torch.inference_mode():
                        normalized_obs = preprocessor(obs)
                        if should_replan_actions(step, args):
                            action_queue = getattr(policy, "_queues", {}).get(ACTION)
                            if action_queue is not None:
                                action_queue.clear()
                        reseed_sampling(args.fixed_noise_seed, device)
                        action = postprocessor(
                            policy.select_action(
                                normalized_obs, noise=fixed_diffusion_noise,
                            )
                        )
                    raw_actions = [choose_action_vector(action)]

                elif args.exec_mode == "chunk":
                    if args.chunk_temporal_ensemble:
                        # Temporal-ensemble chunk path
                        if step % args.ensemble_every == 0 or step not in prediction_bank:
                            with torch.inference_mode():
                                normalized_obs = preprocessor(obs)
                                action_queue = getattr(policy, "_queues", {}).get(ACTION)
                                if action_queue is not None:
                                    action_queue.clear()
                                reseed_sampling(args.fixed_noise_seed, device)
                                raw_chunk = select_action_chunk(
                                    policy, postprocessor, normalized_obs,
                                    ACTION, fixed_diffusion_noise,
                                )

                            exec_steps = args.chunk_exec_steps or len(raw_chunk)
                            raw_chunk = smooth_action_chunk(
                                raw_chunk[:exec_steps], last_smoothed_arm, args,
                            )
                            add_temporal_predictions(
                                prediction_bank, step, raw_chunk, chunk_id,
                            )
                            chunk_id += 1

                        temporal_action = temporal_ensemble_action(
                            prediction_bank, step, args.ensemble_decay,
                        )
                        if temporal_action is None:
                            stop_reason = "empty_temporal_ensemble"
                            break
                        raw_actions = [temporal_action]
                        prune_temporal_predictions(prediction_bank, step)
                    else:
                        # Simple chunk path
                        with torch.inference_mode():
                            normalized_obs = preprocessor(obs)
                            action_queue = getattr(policy, "_queues", {}).get(ACTION)
                            if action_queue is not None:
                                action_queue.clear()
                            reseed_sampling(args.fixed_noise_seed, device)
                            raw_chunk = select_action_chunk(
                                policy, postprocessor, normalized_obs,
                                ACTION, fixed_diffusion_noise,
                            )

                        exec_steps = args.chunk_exec_steps or len(raw_chunk)
                        raw_actions = smooth_action_chunk(
                            raw_chunk[:exec_steps], last_smoothed_arm, args,
                        )
                        chunk_id += 1
                else:
                    raise ValueError(f"Unknown exec_mode: {args.exec_mode}")

                # ---- Execute each action ----
                for raw_target in raw_actions:
                    if step >= args.max_steps:
                        break

                    loop_start = time.time()
                    current = bus.read_qpos()

                    target, last_smoothed_arm = build_push_target(
                        raw_target, current, last_smoothed_arm,
                        args.gripper_mode, args.gripper_fixed_value,
                        initial_gripper, args,
                    )

                    # No-op detection
                    command_delta = target - current
                    arm_delta = float(np.max(np.abs(command_delta[:6])))
                    gripper_delta = float(abs(command_delta[6]))
                    if arm_delta < 0.0015 and gripper_delta < 0.0008:
                        noop_count += 1
                    else:
                        noop_count = 0

                    if args.debug_actions and (
                        step == 0 or step == args.max_steps - 1
                        or step % max(1, args.debug_every) == 0
                    ):
                        print(
                            f"  step {step + 1:03d}: arm_delta={arm_delta:.4f} "
                            f"grip_delta={gripper_delta:.4f} noop={noop_count}"
                        )
                        print(f"    state : {fmt_vec(current)}")
                        print(f"    model : {fmt_vec(raw_target)}")
                        print(f"    target: {fmt_vec(target)}")

                    preview = None
                    if not args.no_gui:
                        preview = build_preview(
                            wrist_frame, global_frame,
                            f"EXEC {step + 1}/{args.max_steps}",
                            color=(0, 0, 255),
                        )

                    elapsed = time.time() - loop_start
                    delay = 1.0 / args.hz - elapsed

                    if not args.dry_run:
                        user_stop = stream_servo_segment(
                            bus, current, target,
                            duration=max(0.0, delay),
                            args=args,
                            preview=preview,
                            video_recorder=active_video_recorder,
                            record_camera=global_cam,
                        )
                        if user_stop:
                            stop_reason = "user_stop"
                            break
                    else:
                        if preview is not None:
                            cv2.imshow("Diffusion push deploy", preview)
                            if should_quit(cv2.waitKey(1) & 0xFF):
                                stop_reason = "user_stop"
                                break
                        if delay > 0:
                            time.sleep(delay)

                    if (args.stop_noop_steps > 0
                            and noop_count >= args.stop_noop_steps):
                        stop_reason = f"noop_{noop_count}"
                        break

                    step += 1

                if stop_reason != "max_steps":
                    break

            print(f"  <<< trajectory stop ({stop_reason})")
            final_qpos = bus.read_qpos()
            print(f"  Final qpos: {fmt_vec(final_qpos)}")
            print(f"  Steps executed: {step}")

            run_retreat(bus, expected_start, global_cam,
                        active_video_recorder, args)
            active_video_recorder.close()
            active_video_recorder = None

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        close_camera(wrist_cam)
        close_camera(global_cam)
        if active_video_recorder is not None:
            active_video_recorder.close()
        bus.disconnect()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
