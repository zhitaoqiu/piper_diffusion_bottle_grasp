#!/usr/bin/env python3
"""Deploy a Diffusion Policy trained on ACT adapter-v2 full-trajectory data."""

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

from adapter_v2.piper_bus import PiperMotorsBusV2, PiperMotorsBusV2Config
from adapter_v2.reset import interpolate_qpos_path
from adapter_v2.schema import (
    GRIPPER_OPEN_M,
    PIPER_GRIPPER_MAX_M,
    QposTolerance,
    STANDARD_START_QPOS,
    STATE_DIM,
    as_qpos,
)
from adapter_v2.start_pose import describe_guard_result, start_pose_guard
from camera.rs_camera import RealSenseCamera, USBCamera, find_realsense_devices

WRIST_IMAGE_KEY = "observation.images.wrist_rgb"
GLOBAL_IMAGE_KEY = "observation.images.global_rgb"
STATE_KEY = "observation.state"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpt", required=True)
    parser.add_argument("--can-port", default="can0")
    parser.add_argument("--velocity-pct", type=int, default=25)
    parser.add_argument("--hz", type=float, default=10.0,
                        help="Use the dataset FPS unless intentionally retiming execution.")
    parser.add_argument("--max-steps", type=int, default=320)
    parser.add_argument("--max-delta-arm", type=float, default=0.030,
                        help="Per-step limit for J1-J3 absolute target changes.")
    parser.add_argument("--max-delta-wrist", type=float, default=0.012,
                        help="Per-step limit for J4-J6 absolute target changes.")
    parser.add_argument("--max-gripper-step", type=float, default=0.004)
    parser.add_argument("--hold-open-steps", type=int, default=35,
                        help="Keep the gripper open for at least this many policy steps.")
    parser.add_argument("--hold-open-min-arm-motion", type=float, default=0.08,
                        help="Keep the gripper open until the arm has moved this far from start.")
    parser.add_argument("--hold-open-gripper", type=float, default=GRIPPER_OPEN_M,
                        help="Minimum gripper opening while the hold-open gate is active.")
    parser.add_argument("--disable-close-gate", action="store_true",
                        help="Disable the approach-position gate that prevents early gripper close.")
    parser.add_argument("--close-gate-j2", type=float, default=1.70,
                        help="Allow policy gripper close only after current J2 reaches this value.")
    parser.add_argument("--close-gate-j3", type=float, default=-0.62,
                        help="Allow policy gripper close only after current J3 is below this value.")
    parser.add_argument("--close-gate-gripper", type=float, default=0.09,
                        help="Minimum gripper opening while waiting for close-gate approach pose.")
    parser.add_argument("--action-smooth", type=float, default=0.5,
                        help="EMA weight on current arm target. Gripper is not smoothed.")
    parser.add_argument("--stop-noop-steps", type=int, default=40,
                        help="Stop after this many nearly-stationary commands. 0 disables it.")
    parser.add_argument("--global-camera", default="auto")
    parser.add_argument("--wrist-serial", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--exec-mode", choices=("step", "chunk"), default="step",
                        help="step keeps the legacy per-step executor. chunk executes each "
                             "predicted action chunk as a short smoothed trajectory.")
    parser.add_argument("--chunk-exec-steps", type=int, default=0,
                        help="Number of actions to execute from each generated chunk. "
                             "0 means use the whole policy n_action_steps chunk.")
    parser.add_argument("--chunk-smooth-window", type=int, default=3,
                        help="Moving-average window over arm joints inside each generated chunk.")
    parser.add_argument("--chunk-blend-steps", type=int, default=2,
                        help="Blend the first N arm targets of a new chunk from the previous "
                             "sent target to reduce chunk-boundary jumps.")
    parser.add_argument("--replan-every-step", action="store_true",
                        help="Clear the diffusion action queue before every control step, "
                             "while keeping observation history. This avoids queued future "
                             "phase actions running ahead of the real robot when commands are clamped.")
    parser.add_argument("--replan-interval", type=int, default=0,
                        help="Clear the diffusion action queue every N control steps. "
                             "Use 2-4 to reduce per-step replanning jitter while still "
                             "preventing the action queue from running far ahead. 0 disables it.")
    parser.add_argument("--fixed-noise-seed", type=int, default=None,
                        help="Use a fixed diffusion sampling noise seed for every replan. "
                             "This makes per-step replanning much less jittery.")
    parser.add_argument("--release-after-trajectory", action="store_true", default=True,
                        help="Open gripper after trajectory stops (scripted, not policy).")
    parser.add_argument("--no-release-after-trajectory", action="store_false",
                        dest="release_after_trajectory")
    parser.add_argument("--retreat-after-trajectory", action="store_true", default=True,
                        help="Move arm back to start pose after trajectory/release.")
    parser.add_argument("--no-retreat-after-trajectory", action="store_false",
                        dest="retreat_after_trajectory")
    parser.add_argument("--retreat-velocity-pct", type=int, default=20)
    parser.add_argument("--retreat-hz", type=float, default=30.0)
    parser.add_argument("--retreat-max-arm-step", type=float, default=0.02)
    parser.add_argument("--retreat-max-gripper-step", type=float, default=0.003)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument("--debug-actions", action="store_true")
    parser.add_argument("--debug-every", type=int, default=10)
    parser.add_argument("--start-pose-file", type=Path,
                        default=PROJECT_ROOT / "config" / "adapter_v2_start_pose.json")
    parser.add_argument("--start-guard-mode", choices=("strict", "zone"), default="zone")
    parser.add_argument("--arm-start-tol", type=float, default=0.05)
    parser.add_argument("--gripper-start-tol", type=float, default=0.01)
    parser.add_argument("--skip-start-guard", action="store_true",
                        help="Bypass adapter-v2 fixed-start protection intentionally.")
    args = parser.parse_args()
    if args.hz <= 0:
        parser.error("--hz must be > 0.")
    if args.max_steps <= 0:
        parser.error("--max-steps must be > 0.")
    if args.max_delta_arm <= 0 or args.max_delta_wrist <= 0 or args.max_gripper_step <= 0:
        parser.error("step limits must be > 0.")
    if args.hold_open_steps < 0:
        parser.error("--hold-open-steps must be >= 0.")
    if args.hold_open_min_arm_motion < 0:
        parser.error("--hold-open-min-arm-motion must be >= 0.")
    if not 0 <= args.hold_open_gripper <= PIPER_GRIPPER_MAX_M:
        parser.error(f"--hold-open-gripper must be in [0, {PIPER_GRIPPER_MAX_M}].")
    if not 0 <= args.close_gate_gripper <= PIPER_GRIPPER_MAX_M:
        parser.error(f"--close-gate-gripper must be in [0, {PIPER_GRIPPER_MAX_M}].")
    if not 0 <= args.action_smooth <= 1:
        parser.error("--action-smooth must be in [0, 1].")
    if args.num_inference_steps is not None and args.num_inference_steps <= 0:
        parser.error("--num-inference-steps must be > 0.")
    if args.chunk_exec_steps < 0:
        parser.error("--chunk-exec-steps must be >= 0.")
    if args.chunk_smooth_window < 1:
        parser.error("--chunk-smooth-window must be >= 1.")
    if args.chunk_blend_steps < 0:
        parser.error("--chunk-blend-steps must be >= 0.")
    if args.replan_interval < 0:
        parser.error("--replan-interval must be >= 0.")
    return args


def fmt_vec(values, precision: int = 3) -> str:
    return "[" + ", ".join(f"{float(value):.{precision}f}" for value in values) + "]"


def resolve_device(device_arg: str, allow_cpu: bool) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if allow_cpu:
            return torch.device("cpu")
        raise RuntimeError(
            "CUDA is unavailable. Pass --allow-cpu --device cpu only for a slow dry-run/debug run."
        )
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
    return torch.from_numpy(image).float().div(255.0).permute(2, 0, 1).unsqueeze(0).to(device)


def prepare_observation(qpos, wrist_image, global_image, device, image_inputs):
    obs = {
        STATE_KEY: torch.from_numpy(as_qpos(qpos, label="observation qpos")).unsqueeze(0).to(device)
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


def make_zero_observation(policy, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: torch.zeros((1, *feature_shape(feature)), dtype=torch.float32, device=device)
        for key, feature in policy_input_features(policy).items()
    }


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


def warm_up_policy(policy, preprocessor, postprocessor, device: torch.device, steps: int) -> None:
    if steps <= 0:
        return
    dummy = make_zero_observation(policy, device)
    print(f"  Warming up policy ({steps} step{'s' if steps != 1 else ''}) ...")
    with torch.inference_mode():
        for _ in range(steps):
            _ = postprocessor(policy.select_action(preprocessor(dummy)))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()


def make_fixed_diffusion_noise(policy, device: torch.device, seed: int | None) -> torch.Tensor | None:
    if seed is None:
        return None

    action_dim = int(policy.diffusion.config.action_feature.shape[0])
    dtype = next(policy.diffusion.parameters()).dtype
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return torch.randn(
        (1, policy.config.horizon, action_dim),
        dtype=dtype,
        device=device,
        generator=generator,
    )


def reseed_sampling(seed: int | None, device: torch.device) -> None:
    if seed is None:
        return
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def should_replan_actions(step: int, args) -> bool:
    if args.replan_every_step:
        return True
    return args.replan_interval > 0 and step % args.replan_interval == 0


def load_start_pose(path: Path) -> np.ndarray:
    if not path.exists():
        print(f"  [WARN] start pose file missing: {path}. Using adapter-v2 schema default.")
        return STANDARD_START_QPOS.copy()
    data = json.loads(path.read_text(encoding="utf-8"))
    return as_qpos(data["qpos"], label=f"start pose file {path}")


def frame_rgb(frame):
    return frame.rgb if frame is not None else None


def build_preview(wrist_frame, global_frame, label: str, color=(0, 255, 0)):
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
    cv2.putText(preview, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    return preview


def close_camera(camera) -> None:
    if camera is None:
        return
    close = getattr(camera, "close", None)
    if close is not None:
        close()


def choose_action_vector(action: torch.Tensor) -> np.ndarray:
    while action.dim() > 1 and action.shape[0] == 1:
        action = action.squeeze(0)
    if action.dim() == 2 and action.shape[-1] == STATE_DIM:
        action = action[0]
    vector = action.detach().cpu().numpy().reshape(-1)
    return as_qpos(vector, label="Diffusion action")


def choose_action_chunk(action_chunk: torch.Tensor) -> np.ndarray:
    while action_chunk.dim() > 1 and action_chunk.shape[0] == 1:
        action_chunk = action_chunk.squeeze(0)
    if action_chunk.dim() == 1:
        action_chunk = action_chunk.unsqueeze(0)
    chunk = action_chunk.detach().cpu().numpy().reshape(-1, STATE_DIM)
    return np.stack(
        [as_qpos(action, label=f"Diffusion chunk action {idx}") for idx, action in enumerate(chunk)],
        axis=0,
    )


def select_action_chunk(policy, postprocessor, normalized_obs, action_key, noise) -> np.ndarray:
    first_action = policy.select_action(normalized_obs, noise=noise)
    action_queue = getattr(policy, "_queues", {}).get(action_key)
    normalized_actions = [first_action]
    if action_queue is not None:
        normalized_actions.extend(list(action_queue))
        action_queue.clear()
    normalized_chunk = torch.stack(normalized_actions, dim=1)
    return choose_action_chunk(postprocessor(normalized_chunk))


def smooth_action_chunk(raw_chunk: np.ndarray, last_sent_arm: np.ndarray | None, args) -> np.ndarray:
    chunk = np.asarray(raw_chunk, dtype=np.float32).copy()
    if len(chunk) == 0:
        return chunk

    if last_sent_arm is not None and args.chunk_blend_steps > 0:
        blend_steps = min(int(args.chunk_blend_steps), len(chunk))
        for idx in range(blend_steps):
            weight = (idx + 1) / (blend_steps + 1)
            chunk[idx, :6] = weight * chunk[idx, :6] + (1.0 - weight) * last_sent_arm

    window = int(args.chunk_smooth_window)
    if window > 1 and len(chunk) > 1:
        pad_left = window // 2
        pad_right = window - 1 - pad_left
        padded = np.pad(chunk[:, :6], ((pad_left, pad_right), (0, 0)), mode="edge")
        kernel = np.ones(window, dtype=np.float32) / window
        for joint_idx in range(6):
            chunk[:, joint_idx] = np.convolve(padded[:, joint_idx], kernel, mode="valid")

    return chunk


def clip_step_target(raw_target, current, args) -> np.ndarray:
    target = as_qpos(raw_target, label="raw Diffusion target").copy()
    current_qpos = as_qpos(current, label="current qpos")
    delta = target - current_qpos
    limits = np.asarray(
        [args.max_delta_arm] * 3 + [args.max_delta_wrist] * 3,
        dtype=np.float32,
    )
    delta[:6] = np.clip(delta[:6], -limits, limits)
    delta[6] = np.clip(delta[6], -args.max_gripper_step, args.max_gripper_step)
    target = current_qpos + delta
    target[:6] = np.clip(target[:6], -3.14, 3.14)
    target[6] = np.clip(target[6], 0.0, PIPER_GRIPPER_MAX_M)
    return target


def apply_gripper_hold_open(target, current, trajectory_start, step: int, args) -> tuple[np.ndarray, bool]:
    arm_motion = float(np.max(np.abs(current[:6] - trajectory_start[:6])))
    hold_by_step = step < args.hold_open_steps
    hold_by_motion = arm_motion < args.hold_open_min_arm_motion
    hold_open = hold_by_step or hold_by_motion
    if not hold_open:
        return target, False

    guarded = target.copy()
    guarded[6] = max(float(guarded[6]), float(current[6]), float(args.hold_open_gripper))
    guarded[6] = min(guarded[6], PIPER_GRIPPER_MAX_M)
    return guarded, True


def apply_approach_close_gate(target, current, args) -> tuple[np.ndarray, bool]:
    if args.disable_close_gate:
        return target, False

    approach_ready = (
        float(current[1]) >= float(args.close_gate_j2)
        and float(current[2]) <= float(args.close_gate_j3)
    )
    if approach_ready or float(target[6]) >= float(args.close_gate_gripper):
        return target, False

    guarded = target.copy()
    guarded[6] = max(float(target[6]), float(current[6]), float(args.close_gate_gripper))
    guarded[6] = min(guarded[6], PIPER_GRIPPER_MAX_M)
    return guarded, True


def build_target_from_action(raw_target, current, trajectory_start, step: int, last_smoothed_arm, args):
    target = clip_step_target(raw_target, current, args)

    if last_smoothed_arm is not None and args.action_smooth > 0:
        target[:6] = (
            args.action_smooth * target[:6]
            + (1.0 - args.action_smooth) * last_smoothed_arm
        )
    next_smoothed_arm = target[:6].copy()

    target, gripper_gate_active = apply_gripper_hold_open(
        target, current, trajectory_start, step, args
    )
    target, close_gate_active = apply_approach_close_gate(target, current, args)
    if gripper_gate_active:
        gripper_gate_label = "start"
    elif close_gate_active:
        gripper_gate_label = "approach"
    else:
        gripper_gate_label = "policy"

    return target, next_smoothed_arm, gripper_gate_label


def guard_passes(bus, expected_start, args) -> bool:
    if args.skip_start_guard:
        return True
    current = bus.read_qpos()
    tolerance = QposTolerance(args.arm_start_tol, args.gripper_start_tol)
    passed = start_pose_guard(
        current,
        expected_start,
        mode=args.start_guard_mode,
        tolerance=tolerance,
    )
    summary = describe_guard_result(
        current,
        expected_start,
        mode=args.start_guard_mode,
        tolerance=tolerance,
    )
    print(f"  Start guard {'PASS' if passed else 'FAIL'}: {summary}")
    return passed


def should_quit(key: int) -> bool:
    return key in (27, ord("q"), ord("Q"))


def main() -> int:
    args = parse_args()
    try:
        device = resolve_device(args.device, args.allow_cpu)
    except RuntimeError as exc:
        raise SystemExit(f"error: {exc}") from exc
    configure_torch_runtime(args.torch_threads, device)
    expected_start = load_start_pose(args.start_pose_file)

    print("=" * 72)
    print("Diffusion adapter-v2 full-trajectory deploy")
    print(f"  checkpoint  : {args.checkpt}")
    print(f"  device      : {device}")
    print(f"  hz          : {args.hz}")
    print(f"  exec mode   : {args.exec_mode}")
    print(f"  start guard : {'SKIPPED' if args.skip_start_guard else args.start_guard_mode}")
    print(f"  dry run     : {args.dry_run}")
    print(
        "  grip gate   : "
        f"hold open {args.hold_open_steps} steps and "
        f"until arm moves {args.hold_open_min_arm_motion:.3f} rad"
    )
    print("=" * 72)

    print("\n[1/4] Loading policy ...")
    from lerobot.policies.diffusion.modeling_diffusion import ACTION, DiffusionPolicy

    policy = DiffusionPolicy.from_pretrained(args.checkpt)
    policy.to(device)
    policy.eval()
    if args.num_inference_steps is not None:
        policy.diffusion.num_inference_steps = args.num_inference_steps
        policy.config.num_inference_steps = args.num_inference_steps
    fixed_diffusion_noise = make_fixed_diffusion_noise(policy, device, args.fixed_noise_seed)
    image_inputs = required_image_inputs(policy)
    print(
        f"  horizon={policy.config.horizon} n_action_steps={policy.config.n_action_steps} "
        f"n_obs_steps={policy.config.n_obs_steps}"
    )
    if args.exec_mode == "chunk":
        chunk_exec_steps = args.chunk_exec_steps or policy.config.n_action_steps
        print(
            f"  chunk executor: exec_steps={chunk_exec_steps} "
            f"smooth_window={args.chunk_smooth_window} blend={args.chunk_blend_steps}"
        )
    else:
        if args.replan_every_step:
            print("  replan: every control step")
        elif args.replan_interval > 0:
            print(f"  replan: every {args.replan_interval} control steps")
        else:
            print(f"  replan: policy queue default ({policy.config.n_action_steps} action steps)")
    if args.fixed_noise_seed is not None:
        print(f"  fixed diffusion noise seed: {args.fixed_noise_seed}")
    print(f"  image inputs: wrist={image_inputs['wrist']} global={image_inputs['global']}")

    print("\n[2/4] Loading processors ...")
    preprocessor, postprocessor = load_policy_processors(policy, args.checkpt, device)
    warm_up_policy(policy, preprocessor, postprocessor, device, args.warmup_steps)

    print(f"\n[3/4] Connecting Piper on {args.can_port} ...")
    bus = PiperMotorsBusV2(
        PiperMotorsBusV2Config(
            can_port=args.can_port,
            velocity_pct=args.velocity_pct,
            disable_torque_on_disconnect=False,
        )
    )
    bus.connect()
    print(f"  qpos: {fmt_vec(bus.read_qpos())}")

    print("\n[4/4] Initializing cameras ...")
    wrist_cam = None
    global_cam = None
    if image_inputs["wrist"]:
        serials = find_realsense_devices()
        wrist_serial = args.wrist_serial or (serials[0] if serials else "")
        wrist_cam = RealSenseCamera(
            serial=wrist_serial,
            width=640,
            height=480,
            fps=30,
            enable_depth=False,
        )
    else:
        print("  Wrist camera skipped.")
    if image_inputs["global"]:
        global_cam = USBCamera(device_id=args.global_camera, width=640, height=480, fps=30)
    else:
        print("  Global camera skipped.")

    print()
    print("SPACE = run one full trajectory, Q/ESC = quit")
    if args.no_gui:
        print("Terminal mode: ENTER runs one full trajectory.")
    print()

    try:
        while True:
            if args.no_gui:
                command = input("Press ENTER to run, or Q then ENTER to quit: ").strip().lower()
                if command == "q":
                    break
            else:
                wrist_frame = wrist_cam.read() if wrist_cam else None
                global_frame = global_cam.read() if global_cam else None
                preview = build_preview(wrist_frame, global_frame, "READY - SPACE")
                cv2.imshow("Diffusion adapter-v2 deploy", preview)
                key = cv2.waitKey(1) & 0xFF
                if should_quit(key):
                    break
                if key != ord(" "):
                    continue

            if not guard_passes(bus, expected_start, args):
                print("  Move the arm back into the adapter-v2 start zone before running.")
                continue

            policy.reset()
            preprocessor.reset()
            postprocessor.reset()
            last_smoothed_arm = None
            noop_count = 0
            stop_reason = "max_steps"
            trajectory_start = bus.read_qpos()
            print(f"  >>> trajectory start ({args.max_steps} max steps)")

            step = 0
            chunk_id = 0
            while step < args.max_steps:
                if args.exec_mode == "step":
                    loop_start = time.time()
                    wrist_frame = wrist_cam.read() if wrist_cam else None
                    global_frame = global_cam.read() if global_cam else None
                    current = bus.read_qpos()
                    obs = prepare_observation(
                        current,
                        frame_rgb(wrist_frame),
                        frame_rgb(global_frame),
                        device,
                        image_inputs,
                    )

                    with torch.inference_mode():
                        normalized_obs = preprocessor(obs)
                        if should_replan_actions(step, args):
                            action_queue = getattr(policy, "_queues", {}).get(ACTION)
                            if action_queue is not None:
                                action_queue.clear()
                        reseed_sampling(args.fixed_noise_seed, device)
                        action = postprocessor(
                            policy.select_action(normalized_obs, noise=fixed_diffusion_noise)
                        )
                    raw_actions = [choose_action_vector(action)]
                else:
                    wrist_frame = wrist_cam.read() if wrist_cam else None
                    global_frame = global_cam.read() if global_cam else None
                    current = bus.read_qpos()
                    obs = prepare_observation(
                        current,
                        frame_rgb(wrist_frame),
                        frame_rgb(global_frame),
                        device,
                        image_inputs,
                    )

                    with torch.inference_mode():
                        normalized_obs = preprocessor(obs)
                        action_queue = getattr(policy, "_queues", {}).get(ACTION)
                        if action_queue is not None:
                            action_queue.clear()
                        reseed_sampling(args.fixed_noise_seed, device)
                        raw_chunk = select_action_chunk(
                            policy,
                            postprocessor,
                            normalized_obs,
                            ACTION,
                            fixed_diffusion_noise,
                        )

                    exec_steps = args.chunk_exec_steps or len(raw_chunk)
                    raw_actions = smooth_action_chunk(
                        raw_chunk[:exec_steps],
                        last_smoothed_arm,
                        args,
                    )
                    if args.debug_actions:
                        j2_range = (float(raw_actions[:, 1].min()), float(raw_actions[:, 1].max()))
                        j3_range = (float(raw_actions[:, 2].min()), float(raw_actions[:, 2].max()))
                        grip_range = (
                            float(raw_actions[:, 6].min()),
                            float(raw_actions[:, 6].max()),
                        )
                        print(
                            f"  [CHUNK] id={chunk_id} size={len(raw_actions)} "
                            f"J2=[{j2_range[0]:.3f},{j2_range[1]:.3f}] "
                            f"J3=[{j3_range[0]:.3f},{j3_range[1]:.3f}] "
                            f"Grip=[{grip_range[0]:.3f},{grip_range[1]:.3f}]"
                        )
                    chunk_id += 1

                for raw_target in raw_actions:
                    if step >= args.max_steps:
                        break

                    loop_start = time.time()
                    current = bus.read_qpos()
                    target, last_smoothed_arm, gripper_gate_label = build_target_from_action(
                        raw_target,
                        current,
                        trajectory_start,
                        step,
                        last_smoothed_arm,
                        args,
                    )

                    command_delta = target - current
                    arm_delta = float(np.max(np.abs(command_delta[:6])))
                    gripper_delta = float(abs(command_delta[6]))
                    if arm_delta < 0.0015 and gripper_delta < 0.0008:
                        noop_count += 1
                    else:
                        noop_count = 0

                    if args.debug_actions and (
                        step == 0
                        or step == args.max_steps - 1
                        or step % max(1, args.debug_every) == 0
                    ):
                        print(
                            f"  step {step + 1:03d}: arm_delta={arm_delta:.4f} "
                            f"grip_delta={gripper_delta:.4f} noop={noop_count} "
                            f"grip_gate={gripper_gate_label}"
                        )
                        print(f"    state : {fmt_vec(current)}")
                        print(f"    model : {fmt_vec(raw_target)}")
                        print(f"    target: {fmt_vec(target)}")

                    if not args.dry_run:
                        bus.write_qpos(target, velocity_pct=args.velocity_pct)

                    if not args.no_gui:
                        preview = build_preview(
                            wrist_frame,
                            global_frame,
                            f"EXEC {step + 1}/{args.max_steps}",
                            color=(0, 0, 255),
                        )
                        cv2.imshow("Diffusion adapter-v2 deploy", preview)
                        if should_quit(cv2.waitKey(1) & 0xFF):
                            stop_reason = "user_stop"
                            break

                    if args.stop_noop_steps > 0 and noop_count >= args.stop_noop_steps:
                        stop_reason = f"noop_{noop_count}"
                        break

                    step += 1
                    elapsed = time.time() - loop_start
                    delay = 1.0 / args.hz - elapsed
                    if delay > 0:
                        time.sleep(delay)

                if stop_reason != "max_steps":
                    break

            print(f"  <<< trajectory stop ({stop_reason})")
            final_qpos = bus.read_qpos()
            print(f"  Final qpos: {fmt_vec(final_qpos)}")
            print(f"  Steps executed: {step}")

            # ── Scripted post-trajectory: release + retreat ──
            if not args.dry_run and (args.release_after_trajectory or args.retreat_after_trajectory):
                print("\n  === Post-trajectory scripted actions ===")

            if args.release_after_trajectory:
                if args.dry_run:
                    print("  [DRY RUN] Would release gripper.")
                else:
                    print("\n  >>> Release: opening gripper ...")
                    cur = bus.read_qpos()
                    release_target = cur.copy()
                    release_target[6] = GRIPPER_OPEN_M
                    path = interpolate_qpos_path(
                        cur, release_target,
                        max_arm_step=np.full(6, args.retreat_max_arm_step, dtype=np.float32),
                        max_gripper_step_m=args.retreat_max_gripper_step,
                    )
                    print(f"    grip {cur[6]:.4f} → {GRIPPER_OPEN_M:.4f} ({len(path)} steps)")
                    for i, qpos in enumerate(path, start=1):
                        bus.write_qpos(qpos, velocity_pct=args.retreat_velocity_pct)
                        if i == 1 or i == len(path):
                            print(f"    release {i:03d}/{len(path):03d}  grip={qpos[6]:.4f}")
                        time.sleep(1.0 / args.retreat_hz)
                    # Dwell
                    hold_start = time.time()
                    while time.time() - hold_start < 0.3:
                        bus.write_qpos(release_target, velocity_pct=args.retreat_velocity_pct)
                        time.sleep(1.0 / args.retreat_hz)
                    print(f"    Release complete. grip={bus.read_qpos()[6]:.5f} m")

            if args.retreat_after_trajectory:
                if args.dry_run:
                    print("  [DRY RUN] Would retreat to start pose.")
                else:
                    print("\n  >>> Retreat: returning to start pose ...")
                    target = load_start_pose(args.start_pose_file)
                    cur = bus.read_qpos()
                    # Preserve current gripper so we don't re-close during retreat
                    target[6] = cur[6]
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
                        if i == 1 or i == len(path) or i % 10 == 0:
                            print(f"    retreat {i:03d}/{len(path):03d}  {fmt_vec(qpos)}")
                        time.sleep(1.0 / args.retreat_hz)
                    final = bus.read_qpos()
                    print(f"    Retreat complete. {fmt_vec(final)}")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        close_camera(wrist_cam)
        close_camera(global_cam)
        bus.disconnect()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
