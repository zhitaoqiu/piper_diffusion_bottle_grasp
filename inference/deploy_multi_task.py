#!/usr/bin/env python3
"""Deploy a Multi-Task DiT (CLIP-based Diffusion Transformer) policy.

Differences from deploy.py (single-task ResNet18 diffusion):
- CLIP ViT vision encoder instead of ResNet18
- Text-conditioned: requires --task "pick green object..."
- MultiTaskDiTPolicy instead of DiffusionPolicy
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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
    QposTolerance,
    STANDARD_START_QPOS,
    STATE_DIM,
    as_qpos,
)
from piper_control.start_pose import describe_guard_result, start_pose_guard
from camera.rs_camera import RealSenseCamera, USBCamera, find_realsense_devices

WRIST_IMAGE_KEY = "observation.images.wrist_rgb"
GLOBAL_IMAGE_KEY = "observation.images.global_rgb"
STATE_KEY = "observation.state"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpt", required=True)
    parser.add_argument("--task", required=True,
                        help="Language task description, e.g. 'Pick up the green object and put it into the box.'")
    parser.add_argument("--can-port", default="can0")
    parser.add_argument("--velocity-pct", type=int, default=25)
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--servo-hz", type=float, default=50.0)
    parser.add_argument("--servo-ease", choices=("linear", "smoothstep", "cosine"),
                        default="smoothstep")
    parser.add_argument("--arm-control-mode", choices=("position", "mit"), default="position")
    parser.add_argument("--allow-mit-control", action="store_true")
    parser.add_argument("--mit-kp", type=float, default=8.0)
    parser.add_argument("--mit-kd", type=float, default=0.8)
    parser.add_argument("--mit-max-vel-ref", type=float, default=1.2)
    parser.add_argument("--mit-hold-sec", type=float, default=0.25)
    parser.add_argument("--max-steps", type=int, default=320)
    parser.add_argument("--max-delta-arm", type=float, default=0.030)
    parser.add_argument("--max-delta-wrist", type=float, default=0.012)
    parser.add_argument("--max-gripper-step", type=float, default=0.004)
    parser.add_argument("--hold-open-steps", type=int, default=35)
    parser.add_argument("--hold-open-min-arm-motion", type=float, default=0.08)
    parser.add_argument("--hold-open-gripper", type=float, default=GRIPPER_OPEN_M)
    parser.add_argument("--disable-close-gate", action="store_true")
    parser.add_argument("--close-gate-j2", type=float, default=1.70)
    parser.add_argument("--close-gate-j3", type=float, default=-0.62)
    parser.add_argument("--close-gate-gripper", type=float, default=0.09)
    parser.add_argument("--action-smooth", type=float, default=0.5)
    parser.add_argument("--stop-noop-steps", type=int, default=40)
    parser.add_argument("--global-camera", default="auto")
    parser.add_argument("--wrist-serial", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--exec-mode", choices=("step", "chunk"), default="step")
    parser.add_argument("--chunk-exec-steps", type=int, default=0)
    parser.add_argument("--chunk-smooth-window", type=int, default=3)
    parser.add_argument("--chunk-blend-steps", type=int, default=2)
    parser.add_argument("--chunk-temporal-ensemble", action="store_true")
    parser.add_argument("--ensemble-every", type=int, default=4)
    parser.add_argument("--ensemble-decay", type=float, default=0.35)
    parser.add_argument("--replan-every-step", action="store_true")
    parser.add_argument("--replan-interval", type=int, default=0)
    parser.add_argument("--release-after-trajectory", action="store_true", default=True)
    parser.add_argument("--no-release-after-trajectory", action="store_false",
                        dest="release_after_trajectory")
    parser.add_argument("--release-hold-sec", type=float, default=1.0)
    parser.add_argument("--retreat-after-trajectory", action="store_true", default=True)
    parser.add_argument("--no-retreat-after-trajectory", action="store_false",
                        dest="retreat_after_trajectory")
    parser.add_argument("--retreat-velocity-pct", type=int, default=20)
    parser.add_argument("--retreat-hz", type=float, default=30.0)
    parser.add_argument("--retreat-max-arm-step", type=float, default=0.02)
    parser.add_argument("--retreat-max-gripper-step", type=float, default=0.003)
    parser.add_argument("--post-after-step", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument("--save-global-video", action="store_true")
    parser.add_argument("--global-video-dir", type=Path,
                        default=PROJECT_ROOT / "logs" / "videos")
    parser.add_argument("--global-video-fps", type=float, default=20.0)
    parser.add_argument("--debug-actions", action="store_true")
    parser.add_argument("--debug-preprocessed-keys", action="store_true",
                        help="Print processed keys/shapes on first step and verify language keys are present.")
    parser.add_argument("--save-debug-frames", action="store_true",
                        help="Save step 0 wrist/global frames for visual inspection.")
    parser.add_argument("--debug-frame-dir", type=Path,
                        default=PROJECT_ROOT / "logs" / "debug_frames",
                        help="Directory for --save-debug-frames output.")
    parser.add_argument("--debug-every", type=int, default=10)
    parser.add_argument("--start-pose-file", type=Path,
                        default=PROJECT_ROOT / "config" / "start_pose.json")
    parser.add_argument("--start-guard-mode", choices=("strict", "zone"), default="zone")
    parser.add_argument("--arm-start-tol", type=float, default=0.05)
    parser.add_argument("--gripper-start-tol", type=float, default=0.01)
    parser.add_argument("--skip-start-guard", action="store_true")
    return parser.parse_args()


def fmt_vec(values, precision: int = 3) -> str:
    return "[" + ", ".join(f"{float(value):.{precision}f}" for value in values) + "]"


def resolve_device(device_arg: str, allow_cpu: bool) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if allow_cpu:
            return torch.device("cpu")
        raise RuntimeError("CUDA is unavailable.")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device_arg}, but torch.cuda.is_available() is false.")
    if device.type == "cpu" and not allow_cpu:
        raise RuntimeError("CPU inference is disabled by default. Pass --allow-cpu for debugging.")
    return device


def configure_torch_runtime(torch_threads: int, device: torch.device) -> None:
    if torch_threads > 0:
        torch.set_num_threads(torch_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True


def policy_input_features(policy) -> dict:
    return getattr(policy.config, "input_features", {}) or {}


def feature_shape(feature) -> tuple[int, ...]:
    if hasattr(feature, "shape"):
        return tuple(feature.shape)
    return tuple(feature["shape"])


def required_image_inputs(policy) -> dict[str, bool]:
    features = policy_input_features(policy)
    return {
        "wrist": WRIST_IMAGE_KEY in features,
        "global": GLOBAL_IMAGE_KEY in features,
    }


def image_to_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(image).float().div(255.0).permute(2, 0, 1).to(device)


def prepare_observation(qpos, wrist_image, global_image, device, image_inputs, task: str):
    obs = {
        STATE_KEY: torch.as_tensor(
            as_qpos(qpos, label="observation qpos"),
            dtype=torch.float32,
            device=device,
        ),
        "task": task,
    }
    if image_inputs["wrist"]:
        if wrist_image is None:
            raise RuntimeError("Policy requires wrist_rgb, but no wrist frame is available.")
        obs[WRIST_IMAGE_KEY] = image_to_tensor(wrist_image, device)
    if image_inputs["global"]:
        if global_image is None:
            raise RuntimeError("Policy requires global_rgb, but no global frame is available.")
        obs[GLOBAL_IMAGE_KEY] = image_to_tensor(global_image, device)
    return obs


def make_zero_observation(policy, device: torch.device, task: str = "pick up object") -> dict[str, torch.Tensor]:
    features = policy_input_features(policy)
    obs: dict = {}
    for key in (STATE_KEY, WRIST_IMAGE_KEY, GLOBAL_IMAGE_KEY):
        if key in features:
            obs[key] = torch.zeros(feature_shape(features[key]), dtype=torch.float32, device=device)
    obs["task"] = task
    return obs


def load_policy_processors(policy, checkpt: str, device: torch.device):
    from lerobot.policies.factory import make_pre_post_processors

    device_name = str(device)
    pre = {"device_processor": {"device": device_name}, "normalizer_processor": {"device": device_name}}
    post = {"unnormalizer_processor": {"device": device.type}, "device_processor": {"device": "cpu"}}
    return make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=checkpt,
        preprocessor_overrides=pre,
        postprocessor_overrides=post,
    )


def assert_prompt_plumbed(processed, task: str = "") -> None:
    """Check that task text actually reached the tokenizer/preprocessor."""
    if isinstance(processed, dict):
        keys = list(processed.keys())
    elif hasattr(processed, "keys"):
        keys = list(processed.keys())
    else:
        keys = []

    if task:
        print(f"  [debug-preprocessed] Current task prompt: {task!r}")
    print(f"  [debug-preprocessed] keys ({len(keys)}): {keys}")
    for k in keys:
        val = processed[k] if isinstance(processed, dict) else getattr(processed, k)
        if hasattr(val, "shape"):
            print(f"    {k}: shape={tuple(val.shape)} dtype={val.dtype} device={val.device if hasattr(val, 'device') else 'N/A'}")
        else:
            print(f"    {k}: type={type(val).__name__}")

    # strong signal: these keys indicate tokenizer/CLIP actually processed the text
    strong_keys = {"language", "token", "attention_mask", "input_ids", "text"}
    has_strong = any(
        any(pattern in k.lower() for pattern in strong_keys)
        for k in keys
    )
    has_task = "task" in keys

    if has_strong:
        print(f"  [debug-preprocessed] OK: language/token keys present → text condition is plumbed.")
    elif has_task:
        print(
            "  [debug-preprocessed] WARN: only raw 'task' string found in processed keys. "
            "MultiTaskDiTPolicy may consume it directly, but verify that the text is NOT silently ignored. "
            "Consider checking whether the tokenizer processor ran."
        )
    else:
        raise RuntimeError(
            "Prompt was provided but no language/task/token key exists after preprocessing. "
            "The text condition may not reach MultiTaskDiTPolicy."
        )


def sanitize_action(raw_action, expected_dim: int = STATE_DIM) -> np.ndarray:
    """Validate and sanitize raw action tensor from the policy."""
    action = np.asarray(raw_action, dtype=np.float32)
    if action.ndim == 0:
        raise RuntimeError(f"Action is a scalar (shape={action.shape}), expected 1D array of length {expected_dim}.")
    if action.ndim >= 2:
        print(f"  [sanitize] action has shape {action.shape}, taking first row")
        action = action[0]
    if action.ndim != 1:
        raise RuntimeError(f"Action must be 1D after sanitization, got shape {action.shape}.")
    if action.shape[0] != expected_dim:
        raise RuntimeError(
            f"Action dim mismatch: got {action.shape[0]}, expected {expected_dim}. "
            f"Original raw shape: {np.asarray(raw_action).shape}"
        )
    return action


def warm_up_policy(policy, preprocessor, postprocessor, device: torch.device, steps: int, task: str) -> None:
    if steps <= 0:
        return
    dummy = make_zero_observation(policy, device, task)
    print(f"  Warming up policy ({steps} step{'s' if steps != 1 else ''}) ...")
    with torch.inference_mode():
        for _ in range(steps):
            _ = postprocessor(policy.select_action(preprocessor(dummy)))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()


def load_start_pose(path: Path) -> np.ndarray:
    if not path.exists():
        print(f"  [WARN] start pose file missing: {path}. Using default.")
        return STANDARD_START_QPOS.copy()
    data = json.loads(path.read_text(encoding="utf-8"))
    return as_qpos(data["qpos"], label=f"start pose file {path}")


def frame_rgb(frame):
    return frame.rgb if frame is not None else None


def run_approach(qpos0, policy, preprocessor, postprocessor, bus, wrist_cam, global_cam,
                 image_inputs, device, args, task):
    """Run the diffusion policy approach trajectory."""
    exec_steps = min(args.max_steps, 999)
    stop_noop_steps = args.stop_noop_steps

    qpos = qpos0.copy()
    prev_arm_target = qpos0[:6].copy()

    control_hz = args.hz
    servo_hz = args.servo_hz
    subs = max(1, round(servo_hz / control_hz))

    start_qpos = qpos0.copy()
    prev_qpos_for_noop = qpos.copy()
    noop_count = 0

    if args.exec_mode == "chunk":
        chunk_exec_steps = args.chunk_exec_steps or policy.config.n_action_steps
        chunk_action_queue = []
        last_chunk_tail = None

    print(f"\n--- Trajectory start (max {exec_steps} steps, {control_hz} Hz) ---")

    for step in range(exec_steps):
        loop_start = time.perf_counter()

        wrist_frame = wrist_cam.read() if wrist_cam else None
        global_frame = global_cam.read() if global_cam else None
        wrist_img = frame_rgb(wrist_frame)
        global_img = frame_rgb(global_frame)

        if args.save_debug_frames and step == 0:
            debug_frame_dir = Path(args.debug_frame_dir)
            debug_frame_dir.mkdir(parents=True, exist_ok=True)
            for name, img in (("global", global_img), ("wrist", wrist_img)):
                if img is None:
                    continue
                bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                out_path = debug_frame_dir / f"step000_{name}.png"
                cv2.imwrite(str(out_path), bgr)
                print(f"  [debug-frame] saved {out_path} ({img.shape[1]}x{img.shape[0]})")

        obs = prepare_observation(qpos, wrist_img, global_img, device, image_inputs, task)

        if args.exec_mode == "step":
            with torch.inference_mode():
                processed = preprocessor(obs)
                if args.debug_preprocessed_keys and step == 0:
                    assert_prompt_plumbed(processed, task)
                action_out = policy.select_action(processed)
                action = postprocessor(action_out)
            raw_action = action.cpu().numpy().squeeze(0)
            raw_action = sanitize_action(raw_action, STATE_DIM)

            if args.replan_every_step or (args.replan_interval > 0 and step % args.replan_interval == 0):
                policy.reset()
                preprocessor.reset()
                postprocessor.reset()

        elif args.exec_mode == "chunk":
            if len(chunk_action_queue) == 0:
                with torch.inference_mode():
                    processed = preprocessor(obs)
                    if args.debug_preprocessed_keys and step == 0:
                        assert_prompt_plumbed(processed, task)
                    action_chunk = policy.select_action(processed)
                    chunk_raw = postprocessor(action_chunk).cpu().numpy().squeeze(0)
                if chunk_raw.ndim != 2 or chunk_raw.shape[0] <= 1:
                    raise RuntimeError(
                        f"Postprocessor returned shape {chunk_raw.shape}, not a valid action chunk [T, action_dim]. "
                        "policy.select_action() appears to return a single action, not an action chunk. "
                        "Please use --exec-mode step."
                    )
                chunk = chunk_raw.astype(np.float32)
                exec_len = min(chunk_exec_steps, len(chunk))

                # smoothing
                if args.chunk_smooth_window > 1:
                    n_arm = min(6, chunk.shape[1] - 1)
                    for i in range(exec_len):
                        w_start = max(0, i - args.chunk_smooth_window // 2)
                        w_end = min(exec_len, w_start + args.chunk_smooth_window)
                        chunk[i, :n_arm] = chunk[w_start:w_end, :n_arm].mean(axis=0)

                # blend from previous
                if last_chunk_tail is not None and args.chunk_blend_steps > 0:
                    blend = min(args.chunk_blend_steps, exec_len)
                    n_arm = min(6, chunk.shape[1] - 1)
                    alpha = np.linspace(0, 1, blend).reshape(-1, 1)
                    chunk[:blend, :n_arm] = (
                        last_chunk_tail[:n_arm] * (1 - alpha)
                        + chunk[:blend, :n_arm] * alpha
                    )

                chunk_action_queue = list(chunk[:exec_len])
                if exec_len > 0:
                    last_chunk_tail = chunk[exec_len - 1]

            raw_action = chunk_action_queue.pop(0)

        else:
            raise ValueError(f"Unknown exec_mode: {args.exec_mode}")

        # clamp per-step delta against current qpos
        qpos_before = qpos.copy()
        delta = raw_action - qpos
        for j in range(3):
            delta[j] = np.clip(delta[j], -args.max_delta_arm, args.max_delta_arm)
        for j in range(3, 6):
            delta[j] = np.clip(delta[j], -args.max_delta_wrist, args.max_delta_wrist)
        delta[6] = np.clip(delta[6], -args.max_gripper_step, args.max_gripper_step)
        target = qpos + delta

        # EMA smoothing on arm joints only
        target[:6] = args.action_smooth * prev_arm_target + (1.0 - args.action_smooth) * target[:6]

        # wrist freeze
        if qpos[1] > 1.45:
            target[3:6] = qpos[3:6]

        # hold-open gate
        if step < args.hold_open_steps or np.linalg.norm(target[:3] - start_qpos[:3]) < args.hold_open_min_arm_motion:
            target[6] = max(target[6], args.hold_open_gripper)

        # close gate
        if not args.disable_close_gate:
            if qpos[1] < args.close_gate_j2 or qpos[2] > args.close_gate_j3:
                target[6] = max(target[6], args.close_gate_gripper)

        if args.dry_run:
            if (step + 1) % args.debug_every == 0 or step <= 3:
                print(f"  [dry] step {step + 1:3d}: {fmt_vec(target)}")
        else:
            for sub in range(subs):
                alpha_sub = (sub + 1) / subs
                if args.servo_ease == "linear":
                    interp = qpos[:6] + alpha_sub * (target[:6] - qpos[:6])
                elif args.servo_ease == "cosine":
                    t = alpha_sub
                    w = (1.0 - np.cos(t * np.pi)) / 2.0
                    interp = qpos[:6] + w * (target[:6] - qpos[:6])
                else:
                    t = alpha_sub
                    w = t * t * (3.0 - 2.0 * t)
                    interp = qpos[:6] + w * (target[:6] - qpos[:6])
                gripper_sub = qpos[6] + alpha_sub * (target[6] - qpos[6])
                joint_cmd = np.append(interp, gripper_sub)
                bus.write_qpos(joint_cmd)
                time.sleep(1.0 / servo_hz)

        qpos = bus.read_qpos()
        prev_arm_target = target[:6].copy()

        # stagnation check — per-step motion
        if args.stop_noop_steps > 0:
            step_motion = np.linalg.norm(qpos[:6] - prev_qpos_for_noop[:6])
            if step_motion < 8e-4:
                noop_count += 1
            else:
                noop_count = 0
            prev_qpos_for_noop = qpos.copy()
            if noop_count >= stop_noop_steps:
                print(f"<<< trajectory stop (stagnation after {step + 1} steps)")
                break

        if args.debug_actions and ((step + 1) % args.debug_every == 0 or step <= 3):
            delta_before = raw_action - qpos_before
            print(f"  step {step + 1:3d}:")
            print(f"    qpos_before              = {fmt_vec(qpos_before)}")
            print(f"    raw_action               = {fmt_vec(raw_action)}")
            print(f"    delta_before_clip        = {fmt_vec(delta_before)}")
            print(f"    delta_after_clip         = {fmt_vec(delta)}")
            print(f"    target_after_smooth_gates= {fmt_vec(target)}")
            print(f"    qpos_after               = {fmt_vec(qpos)}")

        elapsed = time.perf_counter() - loop_start
        sleep = 1.0 / control_hz - elapsed
        if sleep > 0:
            time.sleep(sleep)

    return qpos


def scripted_release(bus, qpos, args):
    """Open gripper with scripted interpolation."""
    print("... scripted release (open gripper)")
    qpos_target = qpos.copy()
    qpos_target[6] = GRIPPER_OPEN_M
    n = int(args.release_hold_sec * args.retreat_hz)
    max_grip = args.retreat_max_gripper_step
    waypoints = interpolate_qpos_path(qpos, qpos_target, max_arm_step=0.05, max_gripper_step_m=max_grip)
    if len(waypoints) < n:
        while len(waypoints) < n:
            waypoints.append(qpos_target.copy())
    for wp in waypoints:
        bus.write_qpos(wp)
        time.sleep(1.0 / args.retreat_hz)
    return bus.read_qpos()


def scripted_retreat(bus, qpos, start_pose, args):
    """Move arm back to start pose."""
    print("... scripted retreat to start pose")
    current = qpos.copy()
    target = start_pose.copy()
    target[6] = current[6]
    waypoints = interpolate_qpos_path(current, target,
                                       max_arm_step=args.retreat_max_arm_step,
                                       max_gripper_step_m=0.003)
    for wp in waypoints:
        bus.write_qpos(wp)
        time.sleep(1.0 / args.retreat_hz)
    return bus.read_qpos()


def main() -> int:
    args = parse_args()

    # --- guard unimplemented parameters ---
    if args.arm_control_mode != "position":
        raise SystemExit(
            f"MIT mode (--arm-control-mode={args.arm_control_mode}) is parsed but not implemented in this script."
        )
    if getattr(args, "chunk_temporal_ensemble", False):
        raise SystemExit(
            "Temporal ensemble (--chunk-temporal-ensemble) is parsed but not implemented in this script."
        )

    try:
        device = resolve_device(args.device, args.allow_cpu)
    except RuntimeError as exc:
        raise SystemExit(f"error: {exc}") from exc
    configure_torch_runtime(args.torch_threads, device)
    expected_start = load_start_pose(args.start_pose_file)

    print("=" * 72)
    print("Multi-Task DiT deploy")
    print(f"  checkpoint  : {args.checkpt}")
    print(f"  task        : {args.task}")
    print(f"  device      : {device}")
    print(f"  hz          : {args.hz}")
    print(f"  exec mode   : {args.exec_mode}")
    print(f"  dry run     : {args.dry_run}")
    print("=" * 72)

    print("\n[1/4] Loading MultiTaskDiT policy ...")
    from lerobot.policies.multi_task_dit.modeling_multi_task_dit import MultiTaskDiTPolicy

    policy = MultiTaskDiTPolicy.from_pretrained(args.checkpt)
    policy.to(device)
    policy.eval()
    if args.num_inference_steps is not None:
        if hasattr(policy, 'objective'):
            policy.objective.num_inference_steps = args.num_inference_steps
        policy.config.num_inference_steps = args.num_inference_steps

    image_inputs = required_image_inputs(policy)
    print(
        f"  horizon={policy.config.horizon} n_action_steps={policy.config.n_action_steps} "
        f"n_obs_steps={policy.config.n_obs_steps}"
    )
    print(f"  image inputs: wrist={image_inputs['wrist']} global={image_inputs['global']}")

    print("\n[2/4] Loading processors ...")
    preprocessor, postprocessor = load_policy_processors(policy, args.checkpt, device)
    warm_up_policy(policy, preprocessor, postprocessor, device, args.warmup_steps, args.task)

    bus = None
    wrist_cam = None
    global_cam = None
    try:
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

        print("\n[4/4] Initializing cameras ...")
        if image_inputs["wrist"]:
            serials = find_realsense_devices()
            wrist_serial = args.wrist_serial or (serials[0] if serials else "")
            wrist_cam = RealSenseCamera(
                serial=wrist_serial, width=640, height=480, fps=30, enable_depth=False,
            )
            print(f"  wrist cam: {wrist_serial or 'default'}")
        if image_inputs["global"]:
            global_cam = USBCamera(device_id=args.global_camera, width=640, height=480, fps=30)
            print(f"  global cam: {args.global_camera}")

        # start guard
        qpos = bus.read_qpos()
        if not args.skip_start_guard:
            tolerance = QposTolerance(arm_rad=args.arm_start_tol, gripper_m=args.gripper_start_tol)
            ok = start_pose_guard(qpos, expected_start, mode=args.start_guard_mode, tolerance=tolerance)
            if not ok:
                msg = describe_guard_result(qpos, expected_start, mode=args.start_guard_mode, tolerance=tolerance)
                print(f"[ABORT] {msg}")
                return 1
            msg = describe_guard_result(qpos, expected_start, mode=args.start_guard_mode, tolerance=tolerance)
            print(f"  start guard: OK ({msg})")
        else:
            print("  start guard: SKIPPED")

        # run policy
        final_qpos = run_approach(
            qpos, policy, preprocessor, postprocessor, bus,
            wrist_cam, global_cam, image_inputs, device, args, args.task,
        )

        # scripted post-trajectory
        if not args.dry_run:
            if args.release_after_trajectory:
                final_qpos = scripted_release(bus, final_qpos, args)
            if args.retreat_after_trajectory:
                final_qpos = scripted_retreat(bus, final_qpos, expected_start, args)
        else:
            print("  [DRY RUN] Skipping release/retreat.")

        print(f"\nFinal qpos: {fmt_vec(bus.read_qpos())}")
        print("Done.")
    finally:
        for cam in (wrist_cam, global_cam):
            if cam is not None:
                try:
                    if hasattr(cam, "stop"):
                        cam.stop()
                    elif hasattr(cam, "close"):
                        cam.close()
                except Exception:
                    pass
        if bus is not None:
            try:
                bus.disconnect()
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
