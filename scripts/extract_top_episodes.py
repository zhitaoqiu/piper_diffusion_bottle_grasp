#!/usr/bin/env python3
"""Extract selected episodes from a LeRobot dataset into a new standalone dataset.

Usage:
  python scripts/extract_top_episodes.py \
    --source-root data/lerobot_dataset \
    --source-repo piper/bottle_pick_place_aside \
    --episodes 0 1 2 3 4 9 10 11 12 13 \
    --target-repo piper/bottle_pick_place_aside_top10 \
    --target-root data/lerobot_dataset_top10
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def extract_episodes(
    source_root: Path,
    source_repo: str,
    episode_indices: list[int],
    target_root: Path,
    target_repo: str,
):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # --- Load source episodes ---
    print(f"Loading episodes {episode_indices} from {source_root} ...")
    src = LeRobotDataset(
        repo_id=source_repo,
        root=source_root,
        episodes=episode_indices,
        video_backend="pyav",
    )
    print(f"  Episodes: {src.num_episodes}")
    print(f"  Frames: {src.num_frames}")
    print(f"  FPS: {src.fps}")

    action_dim = src.features["action"]["shape"][0]
    state_dim = src.features["observation.state"]["shape"][0]
    print(f"  State dim: {state_dim}  Action dim: {action_dim}")

    # --- Build feature dict for new dataset ---
    features = {}
    for key, ft in src.features.items():
        if key in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
            continue
        fdict = {"dtype": ft["dtype"], "shape": tuple(ft["shape"])}
        if "names" in ft:
            fdict["names"] = ft["names"]
        features[key] = fdict

    image_keys = [k for k in features if k.startswith("observation.images.")]
    has_images = len(image_keys) > 0
    print(f"  Image keys: {image_keys}")

    # --- Create new dataset ---
    if target_root.exists():
        import shutil, time
        stamp = time.strftime("%Y%m%d_%H%M%S")
        backup = target_root.with_name(f"{target_root.name}_backup_{stamp}")
        shutil.move(str(target_root), str(backup))
        print(f"  Moved existing {target_root} -> {backup}")

    fps = int(src.fps)
    dst = LeRobotDataset.create(
        repo_id=target_repo,
        fps=fps,
        features=features,
        root=target_root,
        use_videos=has_images,
        image_writer_processes=0,
    )
    print(f"  Created new dataset at {target_root}")

    # --- Copy frames ---
    task_str = src[0].get("task", "")
    if isinstance(task_str, torch.Tensor):
        task_str = str(task_str.item())
    elif isinstance(task_str, np.ndarray):
        task_str = str(task_str.item())
    if not task_str:
        task_str = "pick up the bottle and place it aside"

    print(f"  Task: {task_str}")
    print(f"  Copying {src.num_frames} frames from {len(episode_indices)} episodes ...")

    # Get per-episode boundaries from source
    import pandas as pd
    meta_files = sorted((source_root / "meta" / "episodes").glob("*/*.parquet"))
    dfs = [pd.read_parquet(f) for f in meta_files]
    ep_meta = pd.concat(dfs, ignore_index=True)
    ep_meta = ep_meta[ep_meta["episode_index"].isin(episode_indices)]
    ep_meta = ep_meta.sort_values("episode_index")

    # Build local frame ranges: since source is loaded with specific episodes,
    # frames are contiguous starting at 0. Use meta to get per-episode lengths.
    local_offset = 0
    ep_ranges = []
    for _, row in ep_meta.iterrows():
        n_frames = int(row["length"])
        ep_ranges.append((int(row["episode_index"]), local_offset, local_offset + n_frames))
        local_offset += n_frames

    for src_ep, ep_start, ep_end in ep_ranges:
        n_frames = ep_end - ep_start
        print(f"  Ep{src_ep}: {n_frames} frames (local indices {ep_start}-{ep_end}) ...")
        for i in tqdm(range(ep_start, ep_end), desc=f"    Ep{src_ep}", leave=False):
            item = src[i]
            frame = {
                "observation.state": item["observation.state"].numpy().astype(np.float32),
                "action": item["action"].numpy().astype(np.float32),
                "task": task_str,
            }
            for key in image_keys:
                img = item[key]
                if isinstance(img, torch.Tensor):
                    img = img.numpy()
                if img.ndim == 3 and img.shape[0] in (1, 3):
                    img = img.astype(np.uint8)
                elif img.ndim == 3 and img.shape[-1] == 3:
                    img = np.transpose(img, (2, 0, 1)).astype(np.uint8)
                frame[key] = img
            dst.add_frame(frame)

        dst.save_episode()

    dst.finalize()
    total_frames = sum(int(row["length"]) for _, row in ep_meta.iterrows())
    print(f"  Saved {len(episode_indices)} episodes ({total_frames} frames)")

    # --- Verify ---
    print(f"\nVerifying extracted dataset ...")
    verify = LeRobotDataset(
        repo_id=target_repo,
        root=target_root,
        video_backend="pyav",
    )
    print(f"  Episodes: {verify.num_episodes}")
    print(f"  Frames: {verify.num_frames}")
    print(f"  FPS: {verify.fps}")

    # Quick data check: compare gripper actions at sample points
    print(f"\n  Spot-checking gripper values ...")
    sample_indices = sorted(set([0, 1, total_frames // 3, 2 * total_frames // 3, total_frames - 2, total_frames - 1]))
    sample_indices = [i for i in sample_indices if i < total_frames]

    all_ok = True
    for si in sample_indices:
        v = verify[si]
        grip_val = float(v["action"][-1])
        task_val = str(v.get("task", ""))
        print(f"    frame {si}: grip={grip_val:.4f}  task={task_val}")

    print(f"\nDone. New dataset: {target_root}")
    print(f"  Repo: {target_repo}")
    return verify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default="data/lerobot_dataset")
    parser.add_argument("--source-repo", default="piper/bottle_pick_place_aside")
    parser.add_argument("--episodes", type=int, nargs="+", required=True)
    parser.add_argument("--target-repo", default="piper/bottle_pick_place_aside_top10")
    parser.add_argument("--target-root", type=Path, default="data/lerobot_dataset_top10")
    args = parser.parse_args()

    extract_episodes(
        source_root=args.source_root,
        source_repo=args.source_repo,
        episode_indices=args.episodes,
        target_root=args.target_root,
        target_repo=args.target_repo,
    )


if __name__ == "__main__":
    main()
