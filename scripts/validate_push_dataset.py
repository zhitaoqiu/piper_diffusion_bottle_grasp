#!/usr/bin/env python3
"""Validate a LeRobot dataset for push-like task training readiness.

Checks performed:
  1.  Dataset exists and can be loaded.
  2.  observation.state / action present, shape [N, 7].
  3.  observation.images.global_rgb present.
  4.  No NaN / Inf in state or action.
  5.  Gripper action statistics (min/max/mean/std) with a warning if the
      variance is unexpectedly high.
  6.  Episode length within [--min-frames, --max-frames].
  7.  Per-episode arm motion above a minimal threshold.
  8.  Image black-frame detection (sample-based).
  9.  Per-episode summary table.

This validator does NOT perform object detection, goal recognition, or
task-level semantics checks.
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a push-like LeRobot dataset."
    )
    parser.add_argument("--dataset-path", required=True,
                        help="Path to the dataset root (must contain meta/info.json).")
    parser.add_argument("--repo-id", default="piper/push_task",
                        help="Dataset repo id for LeRobotDataset loading.")
    parser.add_argument("--min-frames", type=int, default=30,
                        help="Minimum frames per episode.")
    parser.add_argument("--max-frames", type=int, default=600,
                        help="Maximum frames per episode.")
    parser.add_argument("--check-images", action="store_true", default=True,
                        help="Check image quality (black frames etc.).")
    parser.add_argument("--no-check-images", action="store_false",
                        dest="check_images")
    parser.add_argument("--check-motion", action="store_true", default=True,
                        help="Check per-episode arm motion.")
    parser.add_argument("--no-check-motion", action="store_false",
                        dest="check_motion")
    parser.add_argument("--require-global", action="store_true", default=True,
                        help="Require observation.images.global_rgb.")
    parser.add_argument("--no-require-global", action="store_false",
                        dest="require_global")
    return parser.parse_args()


def print_header(text: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}")


def check_gripper_variance(gripper_values: np.ndarray) -> list[str]:
    """Return warnings if gripper variance looks unusual."""
    warnings: list[str] = []
    gstd = float(np.std(gripper_values))
    gmin = float(np.min(gripper_values))
    gmax = float(np.max(gripper_values))
    grange = gmax - gmin

    if grange > 0.03:
        warnings.append(
            f"Large gripper range ({grange:.4f} m). "
            f"If this is a push-like task, verify that the gripper "
            f"movement is intentional, not accidental teleop noise."
        )
    if gstd > 0.008:
        warnings.append(
            f"High gripper std ({gstd:.4f} m). "
            f"Check whether gripper was supposed to stay fixed."
        )
    return warnings


def check_black_frame(image: np.ndarray, threshold: float = 5.0) -> bool:
    """Return True if the image appears black (mean pixel < threshold)."""
    if image is None:
        return True
    return bool(np.mean(image) < threshold)


def main() -> int:
    args = parse_args()

    dataset_path = Path(args.dataset_path)
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.exists():
        print(f"[FAIL] Dataset info.json not found: {info_path}")
        return 1

    # ---- Load dataset ----
    print_header("Loading dataset")
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        dataset = LeRobotDataset(args.repo_id, root=str(dataset_path))
    except Exception as exc:
        print(f"[FAIL] Cannot load dataset: {exc}")
        return 1

    meta = dataset.meta
    features = meta.features
    n_episodes = dataset.num_episodes
    n_frames = len(dataset)
    print(f"  episodes : {n_episodes}")
    print(f"  frames   : {n_frames}")

    errors: list[str] = []
    warnings: list[str] = []

    # ---- Check required features ----
    print_header("Feature checks")

    # observation.state
    if "observation.state" not in features:
        errors.append("Missing feature: observation.state")
    else:
        state_shape = features["observation.state"]["shape"]
        print(f"  observation.state : shape={state_shape}")
        if state_shape != [7]:
            errors.append(
                f"observation.state shape is {state_shape}, expected [7]"
            )

    # action
    if "action" not in features:
        errors.append("Missing feature: action")
    else:
        action_shape = features["action"]["shape"]
        print(f"  action             : shape={action_shape}")
        if action_shape != [7]:
            errors.append(
                f"action shape is {action_shape}, expected [7]"
            )

    # global_rgb
    global_key = "observation.images.global_rgb"
    if global_key not in features:
        if args.require_global:
            errors.append(f"Missing feature: {global_key}")
        else:
            warnings.append(f"Feature not present: {global_key}")
    else:
        global_shape = features[global_key]["shape"]
        print(f"  {global_key} : shape={global_shape}")

    # wrist_rgb — informational only
    wrist_key = "observation.images.wrist_rgb"
    if wrist_key in features:
        print(f"  {wrist_key} : present (not required for global-only training)")
    else:
        print(f"  {wrist_key} : not present (OK for global-only training)")

    if errors:
        for err in errors:
            print(f"  [FAIL] {err}")
        print(f"\n  {len(errors)} error(s) found. Aborting.")
        return 1

    # ---- NaN / Inf checks ----
    print_header("NaN / Inf checks")
    all_state = []
    all_action = []
    all_gripper_action = []
    ep_lengths = []
    ep_motions = []

    for ep_idx in range(n_episodes):
        ep_data = dataset.get_episode(ep_idx)
        state = ep_data["observation.state"].numpy() if hasattr(ep_data["observation.state"], "numpy") else np.asarray(ep_data["observation.state"])
        action = ep_data["action"].numpy() if hasattr(ep_data["action"], "numpy") else np.asarray(ep_data["action"])
        ep_len = len(state)
        ep_lengths.append(ep_len)

        nan_state = np.any(~np.isfinite(state))
        nan_action = np.any(~np.isfinite(action))
        if nan_state:
            errors.append(f"Episode {ep_idx}: NaN/Inf in observation.state")
        if nan_action:
            errors.append(f"Episode {ep_idx}: NaN/Inf in action")

        all_state.append(state)
        all_action.append(action)
        all_gripper_action.append(action[:, 6])

        # Motion check
        if args.check_motion and ep_len >= 2:
            arm_deltas = np.diff(state[:, :6], axis=0)
            total_motion = float(np.sum(np.abs(arm_deltas)))
            ep_motions.append(total_motion)

    if errors:
        for err in errors:
            print(f"  [FAIL] {err}")
        print(f"\n  {len(errors)} NaN/Inf error(s) found.")
        return 1
    print("  All state/action values are finite.  OK")

    # ---- Gripper analysis ----
    print_header("Gripper action analysis")
    all_grip = np.concatenate(all_gripper_action)
    gmin, gmax = float(np.min(all_grip)), float(np.max(all_grip))
    gmean, gstd = float(np.mean(all_grip)), float(np.std(all_grip))
    print(f"  min / max     : {gmin:.5f} / {gmax:.5f} m")
    print(f"  mean / std    : {gmean:.5f} / {gstd:.5f} m")
    print(f"  range         : {gmax - gmin:.5f} m")

    grip_warnings = check_gripper_variance(all_grip)
    for w in grip_warnings:
        warnings.append(w)
        print(f"  [WARN] {w}")

    if not grip_warnings:
        print("  Gripper variance looks normal (std < 0.008 m).")

    # ---- Episode length ----
    print_header("Episode length check")
    ep_lengths_arr = np.array(ep_lengths)
    too_short = np.sum(ep_lengths_arr < args.min_frames)
    too_long = np.sum(ep_lengths_arr > args.max_frames)
    print(f"  min / max / mean : {ep_lengths_arr.min()} / "
          f"{ep_lengths_arr.max()} / {ep_lengths_arr.mean():.1f}")
    print(f"  range required   : [{args.min_frames}, {args.max_frames}]")
    if too_short:
        warnings.append(
            f"{too_short} episode(s) shorter than {args.min_frames} frames."
        )
    if too_long:
        warnings.append(
            f"{too_long} episode(s) longer than {args.max_frames} frames."
        )
    if too_short or too_long:
        for w in warnings[-max(too_short, too_long, 1):]:
            print(f"  [WARN] {w}")
    else:
        print("  All episode lengths in range.  OK")

    # ---- Motion check ----
    if args.check_motion and ep_motions:
        print_header("Arm motion check")
        ep_motions_arr = np.array(ep_motions)
        min_motion = float(np.min(ep_motions_arr))
        print(f"  min / max / mean total arm motion : "
              f"{min_motion:.3f} / {float(np.max(ep_motions_arr)):.3f} / "
              f"{float(np.mean(ep_motions_arr)):.3f} rad")
        still_eps = np.where(ep_motions_arr < 0.05)[0]
        if len(still_eps) > 0:
            warnings.append(
                f"{len(still_eps)} episode(s) have very low arm motion "
                f"(< 0.05 rad total): {still_eps.tolist()}"
            )
            print(f"  [WARN] Low-motion episodes: {still_eps.tolist()}")
        else:
            print("  All episodes have sufficient arm motion.  OK")

    # ---- Image check (sample-based) ----
    if args.check_images and global_key in features:
        print_header("Image check (sample-based)")
        black_episodes = []
        samples_per_ep = min(3, max(1, n_episodes // 10))
        sample_episodes = np.linspace(0, n_episodes - 1, samples_per_ep,
                                      dtype=int)

        for ep_idx in sample_episodes:
            ep_data = dataset.get_episode(ep_idx)
            images = ep_data[global_key]
            frame_indices = [0, max(0, len(images) // 2), len(images) - 1]
            frame_indices = sorted(set(f for f in frame_indices if f < len(images)))

            for fi in frame_indices:
                img = images[fi]
                if hasattr(img, "numpy"):
                    img = img.numpy()
                img = np.asarray(img, dtype=np.float32)
                is_black = check_black_frame(img)
                if is_black:
                    black_episodes.append((ep_idx, fi))

        if black_episodes:
            for ep, fi in black_episodes[:10]:
                warnings.append(
                    f"Episode {ep} frame {fi}: appears black/very dark."
                )
            if len(black_episodes) > 10:
                warnings.append(
                    f"... and {len(black_episodes) - 10} more black frames."
                )
            print(f"  [WARN] {len(black_episodes)} black/dark frame(s) found.")
        else:
            print(f"  Sampled {len(sample_episodes)} episodes, "
                  f"no black frames detected.  OK")

    # ---- Per-episode summary ----
    print_header("Per-episode summary")
    print(f"  {'Ep':>4s}  {'Frames':>6s}  "
          f"{'ArmMotion':>10s}  {'GripMean':>9s}  {'GripStd':>8s}")
    print(f"  {'-' * 46}")
    for ep_idx in range(n_episodes):
        ep_len = ep_lengths[ep_idx]
        grip = all_gripper_action[ep_idx]
        arm_motion_str = ""
        if ep_idx < len(ep_motions):
            arm_motion_str = f"{ep_motions[ep_idx]:10.3f}"
        else:
            arm_motion_str = f"{'N/A':>10s}"
        print(
            f"  {ep_idx:4d}  {ep_len:6d}  "
            f"{arm_motion_str}  {float(np.mean(grip)):9.5f}  "
            f"{float(np.std(grip)):8.5f}"
        )

    # ---- Final result ----
    print_header("Validation result")
    print(f"  Errors  : {len(errors)}")
    print(f"  Warnings: {len(warnings)}")
    if errors:
        print("\n  [FAIL] Dataset has errors that must be fixed before training.")
        return 1
    if warnings:
        print("\n  [PASS with warnings] Review the warnings above before training.")
    else:
        print("\n  [PASS] Dataset looks clean and ready for training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
