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
from adapter_v2.schema import (
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
    if not 0 <= args.action_smooth <= 1:
        parser.error("--action-smooth must be in [0, 1].")
    if args.num_inference_steps is not None and args.num_inference_steps <= 0:
        parser.error("--num-inference-steps must be > 0.")
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
    print(f"  start guard : {'SKIPPED' if args.skip_start_guard else args.start_guard_mode}")
    print(f"  dry run     : {args.dry_run}")
    print("=" * 72)

    print("\n[1/4] Loading policy ...")
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

    policy = DiffusionPolicy.from_pretrained(args.checkpt)
    policy.to(device)
    policy.eval()
    if args.num_inference_steps is not None:
        policy.diffusion.num_inference_steps = args.num_inference_steps
        policy.config.num_inference_steps = args.num_inference_steps
    image_inputs = required_image_inputs(policy)
    print(
        f"  horizon={policy.config.horizon} n_action_steps={policy.config.n_action_steps} "
        f"n_obs_steps={policy.config.n_obs_steps}"
    )
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
            print(f"  >>> trajectory start ({args.max_steps} max steps)")

            for step in range(args.max_steps):
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
                    action = postprocessor(policy.select_action(normalized_obs))
                raw_target = choose_action_vector(action)
                target = clip_step_target(raw_target, current, args)

                if last_smoothed_arm is not None and args.action_smooth > 0:
                    target[:6] = (
                        args.action_smooth * target[:6]
                        + (1.0 - args.action_smooth) * last_smoothed_arm
                    )
                last_smoothed_arm = target[:6].copy()

                command_delta = target - current
                arm_delta = float(np.max(np.abs(command_delta[:6])))
                gripper_delta = float(abs(command_delta[6]))
                if arm_delta < 0.0015 and gripper_delta < 0.0008:
                    noop_count += 1
                else:
                    noop_count = 0

                if args.debug_actions and (
                    step == 0 or step == args.max_steps - 1 or step % max(1, args.debug_every) == 0
                ):
                    print(
                        f"  step {step + 1:03d}: arm_delta={arm_delta:.4f} "
                        f"grip_delta={gripper_delta:.4f} noop={noop_count}"
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

                elapsed = time.time() - loop_start
                delay = 1.0 / args.hz - elapsed
                if delay > 0:
                    time.sleep(delay)

            print(f"  <<< trajectory stop ({stop_reason})")
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
