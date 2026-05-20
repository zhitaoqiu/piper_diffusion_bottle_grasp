#!/usr/bin/env python3
"""Rebuild a trimmed LeRobot dataset without modifying the source dataset."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FEATURES = {"timestamp", "frame_index", "episode_index", "index", "task_index"}


def set_hf_cache_defaults(cache_dir: Path) -> None:
    os.environ.setdefault("HF_HOME", str(cache_dir / "hf_home"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_dir / "datasets"))


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


def user_features_from_info(info: dict) -> dict:
    features = {}
    for key, feature in info["features"].items():
        if key in DEFAULT_FEATURES:
            continue
        copied = dict(feature)
        if "shape" in copied:
            copied["shape"] = tuple(copied["shape"])
        features[key] = copied
    return features


def tensor_or_array_to_numpy(value, dtype=None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    else:
        value = np.asarray(value)
    if dtype is not None:
        value = value.astype(dtype)
    return value


def compute_trim_bounds(motion: np.ndarray, threshold: float, preroll: int, tail: int) -> tuple[int, int, int, int]:
    moving = np.flatnonzero(motion > threshold)
    ep_len = len(motion)
    if len(moving) == 0:
        return 0, ep_len, -1, -1
    first_motion = int(moving[0])
    last_motion = int(moving[-1])
    start = max(0, first_motion - preroll)
    end = min(ep_len, last_motion + tail)
    if end <= last_motion:
        end = min(ep_len, last_motion + 1)
    if end <= start:
        end = min(ep_len, start + 1)
    return start, end, first_motion, last_motion


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="piper/bottle_pick_place_aside")
    parser.add_argument("--output-repo-id", default=None)
    parser.add_argument("--motion-threshold", type=float, default=0.005)
    parser.add_argument("--preroll-frames", type=int, default=5)
    parser.add_argument("--tail-frames", type=int, default=8)
    parser.add_argument("--episode", type=int, action="append", default=None,
                        help="Only rebuild this source episode. May be provided multiple times.")
    parser.add_argument("--report-path", type=Path, default=Path("reports/trim_report.csv"))
    parser.add_argument("--overwrite", action="store_true",
                        help="Delete output-root first if it already exists.")
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/piper_hf_cache"))
    parser.add_argument("--video-backend", default=None)
    parser.add_argument("--vcodec", default="libsvtav1")
    parser.add_argument("--encoder-threads", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_root = args.input_root
    output_root = args.output_root
    report_path = args.report_path
    output_repo_id = args.output_repo_id or f"{args.repo_id}_trimmed"

    if not input_root.exists():
        raise FileNotFoundError(f"Input dataset root does not exist: {input_root}")
    if output_root.exists():
        if not args.overwrite:
            raise SystemExit(f"Output root already exists: {output_root}. Use --overwrite to replace it.")
        shutil.rmtree(output_root)

    set_hf_cache_defaults(args.cache_dir)
    sys.path.insert(0, str(PROJECT_ROOT))

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    info = load_info(input_root)
    fps = int(info.get("fps", 30))
    features = user_features_from_info(info)
    use_videos = any(feature.get("dtype") == "video" for feature in features.values())

    data_df = read_parquet_tree(input_root / "data")
    data_df = data_df.reset_index(drop=True)
    data_df["_row_pos"] = np.arange(len(data_df), dtype=np.int64)
    episodes = sorted(int(ep) for ep in data_df["episode_index"].unique())
    if args.episode is not None:
        requested = set(args.episode)
        episodes = [ep for ep in episodes if ep in requested]
        missing = requested - set(episodes)
        if missing:
            raise ValueError(f"Requested episodes not found: {sorted(missing)}")
    if not episodes:
        raise ValueError("No episodes selected for rebuild")
    print(f"Selected source episodes: {episodes}")

    print(f"Loading source dataset: {input_root}")
    source = LeRobotDataset(
        args.repo_id,
        root=input_root,
        return_uint8=True,
        video_backend=args.video_backend,
    )

    print(f"Creating rebuilt dataset: {output_root}")
    target = LeRobotDataset.create(
        repo_id=output_repo_id,
        fps=fps,
        features=features,
        root=output_root,
        robot_type=info.get("robot_type"),
        use_videos=use_videos,
        vcodec=args.vcodec,
        encoder_threads=args.encoder_threads,
        data_files_size_in_mb=info.get("data_files_size_in_mb"),
        video_files_size_in_mb=info.get("video_files_size_in_mb"),
    )

    report_rows = []
    try:
        for ep_id in episodes:
            ep_df = data_df[data_df["episode_index"] == ep_id].copy()
            ep_df = ep_df.sort_values("index" if "index" in ep_df else "frame_index")
            states = stack_vectors(ep_df["observation.state"], "observation.state")
            actions = stack_vectors(ep_df["action"], "action")
            motion = np.max(np.abs(actions[:, :6] - states[:, :6]), axis=1)

            start, end, first_motion, last_motion = compute_trim_bounds(
                motion, args.motion_threshold, args.preroll_frames, args.tail_frames
            )
            kept_rows = ep_df.iloc[start:end]
            if kept_rows.empty:
                raise RuntimeError(f"Episode {ep_id} trim range is empty: start={start}, end={end}")

            for _, raw_row in kept_rows.iterrows():
                item = source[int(raw_row["_row_pos"])]
                frame = {"task": item.get("task", "Grasp the bottle from the table")}
                for key, feature in features.items():
                    if key not in item:
                        raise KeyError(f"Missing key '{key}' in source frame")
                    if feature["dtype"] in ("image", "video"):
                        frame[key] = tensor_or_array_to_numpy(item[key], dtype=np.uint8)
                    else:
                        frame[key] = tensor_or_array_to_numpy(item[key], dtype=np.float32)
                target.add_frame(frame)

            target.save_episode()

            old_len = len(ep_df)
            new_len = int(end - start)
            row = {
                "episode_id": ep_id,
                "old_len": old_len,
                "new_len": new_len,
                "removed_prefix": int(start),
                "removed_suffix": int(old_len - end),
                "first_motion": first_motion,
                "last_motion": last_motion,
                "motion_ratio": float(np.mean(motion > args.motion_threshold)) if len(motion) else 0.0,
            }
            report_rows.append(row)
            print(
                f"ep {ep_id:03d}: {old_len} -> {new_len}, "
                f"removed prefix={row['removed_prefix']} suffix={row['removed_suffix']}, "
                f"first_motion={first_motion}, last_motion={last_motion}"
            )
    finally:
        target.finalize()

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode_id",
                "old_len",
                "new_len",
                "removed_prefix",
                "removed_suffix",
                "first_motion",
                "last_motion",
                "motion_ratio",
            ],
        )
        writer.writeheader()
        writer.writerows(report_rows)

    print(f"\nWrote rebuilt dataset to {output_root}")
    print(f"Wrote trim report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
