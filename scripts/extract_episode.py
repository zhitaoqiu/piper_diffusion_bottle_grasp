#!/usr/bin/env python3
"""Extract a single episode from a LeRobot dataset into a new standalone dataset.

Usage:
  python scripts/extract_episode.py \
    --source-root data/lerobot_dataset \
    --source-repo piper/bottle_pick_place_aside \
    --episode 1 \
    --target-repo piper/bottle_pick_place_aside_overfit_ep1 \
    --target-root data/lerobot_dataset_overfit_ep1
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def extract_episode(
    source_root: Path,
    source_repo: str,
    episode_idx: int,
    target_root: Path,
    target_repo: str,
):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # --- Load source episode ---
    print(f"Loading episode {episode_idx} from {source_root} ...")
    src = LeRobotDataset(
        repo_id=source_repo,
        root=source_root,
        episodes=[episode_idx],
        video_backend="pyav",
    )
    print(f"  Episodes: {src.num_episodes}")
    print(f"  Frames: {src.num_frames}")
    print(f"  FPS: {src.fps}")
    print(f"  Features: {list(src.features.keys())}")

    action_dim = src.features["action"]["shape"][0]
    state_dim = src.features["observation.state"]["shape"][0]
    print(f"  State dim: {state_dim}  Action dim: {action_dim}")

    # Check gripper pattern
    all_grip = []
    sample_n = min(10, src.num_frames)
    for i in np.linspace(0, src.num_frames - 1, sample_n, dtype=int):
        item = src[i]
        all_grip.append(float(item["action"][-1]))
    has_open = all_grip[0] > 0.05
    has_close = min(all_grip) < 0.02
    end_open = all_grip[-1] > 0.05
    print(f"  Gripper action: start={all_grip[0]:.4f} end={all_grip[-1]:.4f} min={min(all_grip):.4f}")
    print(f"  Pattern: open_start={has_open}  has_close={has_close}  open_end={end_open}")

    # --- Build feature dict for new dataset ---
    features = {}
    for key, ft in src.features.items():
        if key in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
            continue
        fdict = {"dtype": ft["dtype"], "shape": tuple(ft["shape"])}
        if "names" in ft:
            fdict["names"] = ft["names"]
        features[key] = fdict

    # Check what image keys exist
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
    print(f"  Copying {src.num_frames} frames ...")
    task_str = src[0].get("task", "")
    if isinstance(task_str, torch.Tensor):
        task_str = str(task_str.item())
    elif isinstance(task_str, np.ndarray):
        task_str = str(task_str.item())
    if not task_str:
        task_str = "pick up the bottle and place it aside"

    for i in tqdm(range(src.num_frames), desc="  Frames"):
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
            # Ensure (C, H, W) uint8
            if img.ndim == 3 and img.shape[0] in (1, 3):
                img = img.astype(np.uint8)
            elif img.ndim == 3 and img.shape[-1] == 3:
                img = np.transpose(img, (2, 0, 1)).astype(np.uint8)
            frame[key] = img
        dst.add_frame(frame)

    dst.save_episode()
    dst.finalize()
    print(f"  Saved 1 episode ({src.num_frames} frames)")

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

    v0 = verify[0]
    vN = verify[verify.num_frames - 1]
    grip_start = float(v0["action"][-1])
    grip_end = float(vN["action"][-1])
    print(f"  Gripper: start={grip_start:.4f}  end={grip_end:.4f}")

    # Check a few frames match
    src_grip = []
    dst_grip = []
    for i in np.linspace(0, src.num_frames - 1, 5, dtype=int):
        src_grip.append(float(src[i]["action"][-1]))
        dst_grip.append(float(verify[i]["action"][-1]))
    print(f"  Source grip sample: {[f'{g:.4f}' for g in src_grip]}")
    print(f"  Verify grip sample: {[f'{g:.4f}' for g in dst_grip]}")

    if np.allclose(src_grip, dst_grip, atol=1e-6):
        print("  Data integrity: PASS")
    else:
        print("  Data integrity: FAIL - source and target differ!")

    print(f"\nDone. Dataset: {target_root}")
    print(f"  Repo: {target_repo}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default="data/lerobot_dataset")
    parser.add_argument("--source-repo", default="piper/bottle_pick_place_aside")
    parser.add_argument("--episode", type=int, default=1)
    parser.add_argument("--target-repo", default="piper/bottle_pick_place_aside_overfit_ep1")
    parser.add_argument("--target-root", type=Path, default="data/lerobot_dataset_overfit_ep1")
    args = parser.parse_args()

    extract_episode(
        source_root=args.source_root,
        source_repo=args.source_repo,
        episode_idx=args.episode,
        target_root=args.target_root,
        target_repo=args.target_repo,
    )


if __name__ == "__main__":
    main()
