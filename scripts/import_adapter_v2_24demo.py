#!/usr/bin/env python3
"""Import the ACT adapter-v2 demos into a fresh Diffusion LeRobot dataset.

The ACT workspace keeps the validated 10-demo baseline and 15 follow-up demos
as separate datasets. Episode 9 from the follow-up set is excluded by default,
which produces the 24-demo dataset used by this Diffusion path.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ACT_ROOT_DEFAULT = Path("/home/huatec/piper_act_bottle_grasp")
PRIMARY_RELATIVE_ROOT = Path("data/lerobot_dataset_piper_bottle_adapter_v2_10demo")
FOLLOWUP_RELATIVE_ROOT = Path("data/lerobot_dataset_piper_bottle_adapter_v2_new_demos")
TARGET_DEFAULT = PROJECT_ROOT / "data" / "lerobot_dataset_piper_bottle_adapter_v2_24demo"
DEFAULT_REPO_ID = "piper/adapter_v2_24demo_diffusion"
DEFAULT_TASK = "grasp the bottle and complete the adapter v2 pick trajectory"
DEFAULT_FEATURES = {"timestamp", "frame_index", "episode_index", "index", "task_index"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--act-root", type=Path, default=ACT_ROOT_DEFAULT)
    parser.add_argument("--primary-root", type=Path, default=None,
                        help="Override the ACT adapter-v2 10-demo source root.")
    parser.add_argument("--followup-root", type=Path, default=None,
                        help="Override the ACT adapter-v2 follow-up source root.")
    parser.add_argument("--target-root", type=Path, default=TARGET_DEFAULT)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--exclude-followup-episode", type=int, action="append", default=[9],
                        help="Follow-up episode to skip. Defaults to the known bad episode 9.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/piper_diffusion_hf_cache"))
    parser.add_argument("--video-backend", default=None)
    parser.add_argument("--vcodec", default="libsvtav1")
    parser.add_argument("--encoder-threads", type=int, default=None)
    return parser.parse_args()


def set_hf_cache_defaults(cache_dir: Path) -> None:
    os.environ.setdefault("HF_HOME", str(cache_dir / "hf_home"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_dir / "datasets"))


def load_info(dataset_root: Path) -> dict:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing LeRobot metadata: {info_path}")
    return json.loads(info_path.read_text(encoding="utf-8"))


def user_features(info: dict) -> dict:
    features = {}
    for key, feature in info["features"].items():
        if key in DEFAULT_FEATURES:
            continue
        copied = dict(feature)
        if "shape" in copied:
            copied["shape"] = tuple(copied["shape"])
        features[key] = copied
    return features


def feature_signature(info: dict) -> dict:
    return {
        key: {
            "dtype": feature.get("dtype"),
            "shape": tuple(feature.get("shape", ())),
            "names": tuple(feature["names"]) if feature.get("names") else None,
        }
        for key, feature in user_features(info).items()
    }


def as_numpy(value, *, dtype=None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    else:
        value = np.asarray(value)
    if dtype is not None:
        value = value.astype(dtype)
    return value


def episode_indices(dataset) -> list[int]:
    values = np.asarray(dataset.hf_dataset["episode_index"])
    return sorted(int(value) for value in np.unique(values))


def episode_positions(dataset, episode_id: int) -> np.ndarray:
    values = np.asarray(dataset.hf_dataset["episode_index"])
    return np.flatnonzero(values == episode_id)


def copy_episode(target, source, episode_id: int, features: dict) -> int:
    positions = episode_positions(source, episode_id)
    if len(positions) == 0:
        raise ValueError(f"Source episode {episode_id} is empty.")

    for position in positions:
        item = source[int(position)]
        frame = {"task": item.get("task", DEFAULT_TASK)}
        for key, feature in features.items():
            if key not in item:
                raise KeyError(f"Source frame from episode {episode_id} is missing feature {key!r}.")
            dtype = np.uint8 if feature.get("dtype") in ("image", "video") else np.float32
            frame[key] = as_numpy(item[key], dtype=dtype)
        target.add_frame(frame)
    target.save_episode()
    return int(len(positions))


def resolve_sources(args) -> tuple[Path, Path]:
    primary = args.primary_root or args.act_root / PRIMARY_RELATIVE_ROOT
    followup = args.followup_root or args.act_root / FOLLOWUP_RELATIVE_ROOT
    return primary, followup


def validate_sources(primary_root: Path, followup_root: Path) -> tuple[dict, dict]:
    primary_info = load_info(primary_root)
    followup_info = load_info(followup_root)
    if int(primary_info.get("fps", 0)) != int(followup_info.get("fps", 0)):
        raise ValueError("Adapter-v2 sources use different FPS values.")
    if feature_signature(primary_info) != feature_signature(followup_info):
        raise ValueError("Adapter-v2 sources use different LeRobot feature schemas.")
    return primary_info, followup_info


def reset_target(target_root: Path, overwrite: bool) -> None:
    if not target_root.exists():
        return
    if not overwrite:
        raise SystemExit(f"Target already exists: {target_root}. Use --overwrite to rebuild it.")
    shutil.rmtree(target_root)


def main() -> int:
    args = parse_args()
    primary_root, followup_root = resolve_sources(args)
    primary_info, _ = validate_sources(primary_root, followup_root)
    reset_target(args.target_root, args.overwrite)
    set_hf_cache_defaults(args.cache_dir)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = user_features(primary_info)
    use_videos = any(feature.get("dtype") == "video" for feature in features.values())
    excluded_followup = set(args.exclude_followup_episode)

    print("=" * 72)
    print("Import ACT adapter-v2 demos for Diffusion")
    print(f"  primary source : {primary_root}")
    print(f"  follow-up      : {followup_root}")
    print(f"  skip follow-up : {sorted(excluded_followup)}")
    print(f"  target         : {args.target_root}")
    print(f"  repo_id        : {args.repo_id}")
    print("=" * 72)

    primary = LeRobotDataset(
        repo_id="piper/adapter_v2_10demo",
        root=primary_root,
        return_uint8=True,
        video_backend=args.video_backend,
    )
    followup = LeRobotDataset(
        repo_id="piper/adapter_v2_new_demos",
        root=followup_root,
        return_uint8=True,
        video_backend=args.video_backend,
    )

    selected = [
        ("primary", primary, ep)
        for ep in episode_indices(primary)
    ]
    selected.extend(
        ("followup", followup, ep)
        for ep in episode_indices(followup)
        if ep not in excluded_followup
    )

    target = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=int(primary_info["fps"]),
        features=features,
        root=args.target_root,
        robot_type=primary_info.get("robot_type"),
        use_videos=use_videos,
        vcodec=args.vcodec,
        encoder_threads=args.encoder_threads,
        data_files_size_in_mb=primary_info.get("data_files_size_in_mb"),
        video_files_size_in_mb=primary_info.get("video_files_size_in_mb"),
    )

    manifest_episodes = []
    total_frames = 0
    try:
        for target_episode, (label, source, source_episode) in enumerate(selected):
            frame_count = copy_episode(target, source, source_episode, features)
            total_frames += frame_count
            manifest_episodes.append(
                {
                    "target_episode": target_episode,
                    "source": label,
                    "source_episode": source_episode,
                    "frames": frame_count,
                }
            )
            print(
                f"  ep {target_episode:02d}: {label} source ep {source_episode:02d} "
                f"-> {frame_count} frames"
            )
    finally:
        target.finalize()

    manifest = {
        "repo_id": args.repo_id,
        "primary_root": str(primary_root),
        "followup_root": str(followup_root),
        "excluded_followup_episodes": sorted(excluded_followup),
        "total_episodes": len(manifest_episodes),
        "total_frames": total_frames,
        "episodes": manifest_episodes,
    }
    manifest_path = args.target_root / "meta" / "adapter_v2_import_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print()
    print(f"Imported {len(manifest_episodes)} episodes and {total_frames} frames.")
    print(f"Fresh LeRobot stats were written during finalize: {args.target_root / 'meta' / 'stats.json'}")
    print(f"Import manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
