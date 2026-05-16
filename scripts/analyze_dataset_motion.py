#!/usr/bin/env python3
"""Analyze LeRobot dataset motion, still segments, jumps, and alignment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def require_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plots. Install it with: pip install matplotlib"
        ) from exc


def load_info(dataset_root: Path) -> dict:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing LeRobot metadata: {info_path}")
    return json.loads(info_path.read_text())


def read_parquet_tree(root: Path) -> pd.DataFrame:
    paths = sorted(root.glob("chunk-*/file-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files found under {root}")
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def stack_vectors(series: pd.Series, key: str) -> np.ndarray:
    try:
        return np.stack([np.asarray(value, dtype=np.float32) for value in series.to_list()])
    except ValueError as exc:
        raise ValueError(f"Could not stack vector column '{key}'") from exc


def count_static_prefix(motion: np.ndarray, threshold: float) -> int:
    moving = np.flatnonzero(motion > threshold)
    return int(moving[0]) if len(moving) else int(len(motion))


def count_static_suffix(motion: np.ndarray, threshold: float) -> int:
    moving = np.flatnonzero(motion > threshold)
    return int(len(motion) - moving[-1] - 1) if len(moving) else int(len(motion))


def episode_meta_by_index(episodes_df: pd.DataFrame) -> dict[int, pd.Series]:
    if episodes_df.empty or "episode_index" not in episodes_df:
        return {}
    return {int(row["episode_index"]): row for _, row in episodes_df.iterrows()}


def video_file_path(info: dict, video_key: str, chunk_index: int, file_index: int) -> Path:
    return Path(
        info["video_path"].format(
            video_key=video_key, chunk_index=int(chunk_index), file_index=int(file_index)
        )
    )


def read_video_frames_at_timestamps(video_path: Path, timestamps: np.ndarray, max_side: int = 160):
    try:
        import cv2
    except ImportError:
        return None, "opencv-python is not installed"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, f"could not open {video_path}"

    frames = []
    try:
        for ts in timestamps:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(ts) * 1000.0)
            ok, frame = cap.read()
            if not ok:
                return None, f"could not decode frame at {ts:.3f}s"
            h, w = frame.shape[:2]
            scale = min(1.0, max_side / max(h, w))
            if scale < 1.0:
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            frames.append(frame)
    finally:
        cap.release()
    return frames, None


def image_repeat_stats(
    dataset_root: Path,
    info: dict,
    episode_meta: pd.Series | None,
    ep_df: pd.DataFrame,
    video_key: str,
    motion: np.ndarray,
    motion_threshold: float,
    pixel_threshold: float,
) -> tuple[float, float, str]:
    if episode_meta is None:
        return math.nan, math.nan, "missing_episode_metadata"
    if video_key not in info.get("features", {}):
        return math.nan, math.nan, f"missing_video_key:{video_key}"

    try:
        chunk = int(episode_meta[f"videos/{video_key}/chunk_index"])
        file_idx = int(episode_meta[f"videos/{video_key}/file_index"])
        from_ts = float(episode_meta[f"videos/{video_key}/from_timestamp"])
    except KeyError as exc:
        return math.nan, math.nan, f"missing_video_metadata:{exc}"

    video_path = dataset_root / video_file_path(info, video_key, chunk, file_idx)
    timestamps = from_ts + ep_df["timestamp"].to_numpy(dtype=np.float32)
    frames, error = read_video_frames_at_timestamps(video_path, timestamps)
    if error:
        return math.nan, math.nan, error

    diffs = []
    for prev, cur in zip(frames[:-1], frames[1:], strict=False):
        diffs.append(float(np.mean(np.abs(cur.astype(np.float32) - prev.astype(np.float32)))))
    if not diffs:
        return math.nan, math.nan, "too_short"

    diffs = np.asarray(diffs, dtype=np.float32)
    repeated = diffs <= pixel_threshold
    moving_pairs = motion[1:] > motion_threshold
    repeat_ratio = float(np.mean(repeated))
    repeated_while_moving = float(np.mean(repeated[moving_pairs])) if np.any(moving_pairs) else math.nan
    return repeat_ratio, repeated_while_moving, ""


def plot_motion(plt, out_path: Path, ep_id: int, motion: np.ndarray, threshold: float):
    x = np.arange(len(motion))
    moving = np.flatnonzero(motion > threshold)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, motion, label="max(abs(action[:6] - state[:6]))", linewidth=1.5)
    ax.axhline(threshold, color="tab:red", linestyle="--", linewidth=1, label="threshold")
    if len(moving):
        ax.axvline(int(moving[0]), color="tab:green", linestyle=":", linewidth=1, label="first motion")
        ax.axvline(int(moving[-1]), color="tab:purple", linestyle=":", linewidth=1, label="last motion")
    ax.set_title(f"Episode {ep_id:03d} motion")
    ax.set_xlabel("local frame")
    ax.set_ylabel("rad")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_joints(plt, out_path: Path, ep_id: int, states: np.ndarray, actions: np.ndarray):
    x = np.arange(len(states))
    fig, axes = plt.subplots(6, 1, figsize=(12, 11), sharex=True)
    for joint_idx, ax in enumerate(axes):
        ax.plot(x, states[:, joint_idx], label="state", linewidth=1)
        ax.plot(x, actions[:, joint_idx], label="action", linewidth=1, alpha=0.8)
        ax.set_ylabel(f"j{joint_idx + 1}")
        ax.grid(True, alpha=0.2)
    axes[0].set_title(f"Episode {ep_id:03d} joints")
    axes[-1].set_xlabel("local frame")
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_gripper(plt, out_path: Path, ep_id: int, states: np.ndarray, actions: np.ndarray):
    x = np.arange(len(states))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, states[:, 6], label="state gripper", linewidth=1.5)
    ax.plot(x, actions[:, 6], label="action gripper", linewidth=1.5)
    ax.plot(x, actions[:, 6] - states[:, 6], label="action - state", linewidth=1)
    ax.set_title(f"Episode {ep_id:03d} gripper")
    ax.set_xlabel("local frame")
    ax.set_ylabel("m")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def build_report_row(
    ep_id: int,
    ep_df: pd.DataFrame,
    episode_meta: pd.Series | None,
    states: np.ndarray,
    actions: np.ndarray,
    motion: np.ndarray,
    fps: int,
    motion_threshold: float,
    jump_threshold: float,
) -> dict:
    first_window = motion[: min(fps, len(motion))]
    last_window = motion[max(0, len(motion) - fps) :]
    start_static_frames = count_static_prefix(motion, motion_threshold)
    end_static_frames = count_static_suffix(motion, motion_threshold)
    moving = np.flatnonzero(motion > motion_threshold)

    state_step = np.zeros(len(states), dtype=np.float32)
    action_step = np.zeros(len(actions), dtype=np.float32)
    if len(states) > 1:
        state_step[1:] = np.max(np.abs(np.diff(states[:, :6], axis=0)), axis=1)
        action_step[1:] = np.max(np.abs(np.diff(actions[:, :6], axis=0)), axis=1)
    jump_mask = (state_step > jump_threshold) | (action_step > jump_threshold) | (motion > jump_threshold)

    frame_index = ep_df["frame_index"].to_numpy(dtype=np.int64) if "frame_index" in ep_df else np.arange(len(ep_df))
    timestamp = ep_df["timestamp"].to_numpy(dtype=np.float32) if "timestamp" in ep_df else np.arange(len(ep_df)) / fps
    dataset_index = ep_df["index"].to_numpy(dtype=np.int64) if "index" in ep_df else np.arange(len(ep_df))

    expected_dt = 1.0 / fps
    timestamp_diffs = np.diff(timestamp)
    timestamp_regular = bool(
        len(timestamp_diffs) == 0 or np.max(np.abs(timestamp_diffs - expected_dt)) < max(1e-3, expected_dt * 0.1)
    )

    meta_len = int(episode_meta["length"]) if episode_meta is not None and "length" in episode_meta else -1
    meta_from = (
        int(episode_meta["dataset_from_index"])
        if episode_meta is not None and "dataset_from_index" in episode_meta
        else -1
    )
    meta_to = (
        int(episode_meta["dataset_to_index"])
        if episode_meta is not None and "dataset_to_index" in episode_meta
        else -1
    )

    warnings = []
    if start_static_frames >= fps:
        warnings.append("long_static_start")
    if end_static_frames >= fps:
        warnings.append("long_static_end")
    if bool(np.any(jump_mask)):
        warnings.append("jump")
    if int(frame_index[0]) != 0:
        warnings.append("frame_index_not_zero")
    if abs(float(timestamp[0])) > max(1e-3, expected_dt * 0.1):
        warnings.append("timestamp_not_zero")
    if not bool(np.all(np.diff(frame_index) == 1)):
        warnings.append("frame_index_gap")
    if not timestamp_regular:
        warnings.append("timestamp_gap")
    if meta_from >= 0 and (int(dataset_index[0]) != meta_from or int(dataset_index[-1]) + 1 != meta_to):
        warnings.append("metadata_index_mismatch")
    if meta_len >= 0 and meta_len != len(ep_df):
        warnings.append("metadata_length_mismatch")

    return {
        "episode_id": ep_id,
        "length": len(ep_df),
        "metadata_length": meta_len,
        "first_1s_static_ratio": float(np.mean(first_window <= motion_threshold)) if len(first_window) else math.nan,
        "last_1s_static_ratio": float(np.mean(last_window <= motion_threshold)) if len(last_window) else math.nan,
        "start_static_frames": start_static_frames,
        "end_static_frames": end_static_frames,
        "long_static_start": start_static_frames >= fps,
        "long_static_end": end_static_frames >= fps,
        "first_motion": int(moving[0]) if len(moving) else -1,
        "last_motion": int(moving[-1]) if len(moving) else -1,
        "motion_ratio": float(np.mean(motion > motion_threshold)) if len(motion) else 0.0,
        "max_motion": float(np.max(motion)) if len(motion) else 0.0,
        "max_state_step": float(np.max(state_step)) if len(state_step) else 0.0,
        "max_action_step": float(np.max(action_step)) if len(action_step) else 0.0,
        "has_jump": bool(np.any(jump_mask)),
        "jump_frames": ",".join(map(str, np.flatnonzero(jump_mask)[:20].tolist())),
        "gripper_state_min": float(np.min(states[:, 6])),
        "gripper_state_max": float(np.max(states[:, 6])),
        "gripper_action_min": float(np.min(actions[:, 6])),
        "gripper_action_max": float(np.max(actions[:, 6])),
        "frame_index_start": int(frame_index[0]),
        "frame_index_end": int(frame_index[-1]),
        "frame_index_contiguous": bool(np.all(np.diff(frame_index) == 1)),
        "timestamp_start": float(timestamp[0]),
        "timestamp_end": float(timestamp[-1]),
        "timestamp_regular": timestamp_regular,
        "dataset_index_start": int(dataset_index[0]),
        "dataset_index_end": int(dataset_index[-1]),
        "metadata_dataset_from_index": meta_from,
        "metadata_dataset_to_index": meta_to,
        "metadata_index_match": bool(
            meta_from < 0 or (int(dataset_index[0]) == meta_from and int(dataset_index[-1]) + 1 == meta_to)
        ),
        "warnings": ";".join(warnings),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("data/lerobot_dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/motion"))
    parser.add_argument("--motion-threshold", type=float, default=0.005)
    parser.add_argument("--jump-threshold", type=float, default=0.20)
    parser.add_argument("--check-image-repeats", action="store_true",
                        help="Decode video frames and estimate adjacent repeated-image ratios.")
    parser.add_argument("--image-repeat-key", default="observation.images.wrist_rgb")
    parser.add_argument("--image-repeat-pixel-threshold", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    plt = require_matplotlib()
    info = load_info(dataset_root)
    fps = int(info.get("fps", 30))
    data_df = read_parquet_tree(dataset_root / "data")
    episodes_df = read_parquet_tree(dataset_root / "meta" / "episodes")
    meta_by_ep = episode_meta_by_index(episodes_df)

    video_fps = {}
    for key, feature in info.get("features", {}).items():
        if feature.get("dtype") == "video":
            video_fps[key] = feature.get("info", {}).get("video.fps")

    print(f"Dataset: {dataset_root}")
    print(f"Metadata fps: {fps}")
    if video_fps:
        print(f"Video fps: {video_fps}")
    print(f"Frames: {len(data_df)}, episodes in data: {data_df['episode_index'].nunique()}")

    rows = []
    for ep_id in sorted(data_df["episode_index"].unique()):
        ep_id = int(ep_id)
        ep_df = data_df[data_df["episode_index"] == ep_id].copy()
        ep_df = ep_df.sort_values("index" if "index" in ep_df else "frame_index")
        states = stack_vectors(ep_df["observation.state"], "observation.state")
        actions = stack_vectors(ep_df["action"], "action")
        motion = np.max(np.abs(actions[:, :6] - states[:, :6]), axis=1)
        episode_meta = meta_by_ep.get(ep_id)

        row = build_report_row(
            ep_id, ep_df, episode_meta, states, actions, motion,
            fps, args.motion_threshold, args.jump_threshold,
        )

        if args.check_image_repeats:
            repeat_ratio, repeated_while_moving, repeat_error = image_repeat_stats(
                dataset_root,
                info,
                episode_meta,
                ep_df,
                args.image_repeat_key,
                motion,
                args.motion_threshold,
                args.image_repeat_pixel_threshold,
            )
            row["image_repeat_ratio"] = repeat_ratio
            row["image_repeated_while_moving_ratio"] = repeated_while_moving
            row["image_repeat_error"] = repeat_error

        rows.append(row)

        plot_motion(plt, output_dir / f"episode_{ep_id:03d}_motion.png", ep_id, motion, args.motion_threshold)
        plot_joints(plt, output_dir / f"episode_{ep_id:03d}_joints.png", ep_id, states, actions)
        plot_gripper(plt, output_dir / f"episode_{ep_id:03d}_gripper.png", ep_id, states, actions)

        print(
            f"ep {ep_id:03d}: len={row['length']} motion_ratio={row['motion_ratio']:.3f} "
            f"start_static={row['start_static_frames']} end_static={row['end_static_frames']} "
            f"warnings={row['warnings'] or '-'}"
        )

    report_path = output_dir / "motion_report.csv"
    with report_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    summary_path = output_dir / "dataset_summary.json"
    summary = {
        "dataset_root": str(dataset_root),
        "fps": fps,
        "video_fps": video_fps,
        "num_frames": int(len(data_df)),
        "num_episodes": int(data_df["episode_index"].nunique()),
        "motion_threshold": args.motion_threshold,
        "jump_threshold": args.jump_threshold,
        "report_csv": str(report_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nSaved plots and report to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
