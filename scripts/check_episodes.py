#!/usr/bin/env python3
"""Check episode quality for bottle pick-and-place dataset.

Usage:
  python scripts/check_episodes.py \
    --repo-id piper/bottle_pick_place_aside \
    --root data/lerobot_dataset
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def check_episodes(dataset_root: Path, repo_id: str):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(
        repo_id=repo_id,
        root=dataset_root,
        video_backend="pyav",
    )

    print(f"Dataset: {repo_id}")
    print(f"Episodes: {ds.num_episodes}  Total frames: {ds.num_frames}  FPS: {ds.fps}")
    print(f"Features: {list(ds.features.keys())}")

    image_keys = [k for k in ds.features if k.startswith("observation.images.")]
    print(f"Image keys: {image_keys}")

    # Check action shape
    a0 = ds[0]["action"].numpy()
    action_dim = len(a0)
    print(f"Action dim: {action_dim}")
    print()

    # Per-episode analysis
    ep_meta = pd.read_parquet(
        sorted((dataset_root / "meta" / "episodes").glob("*/*.parquet"))[0]
    )
    ep_meta = ep_meta.sort_values("episode_index")

    all_pass = True
    episodes_ok = []
    episodes_fail = []

    for _, row in ep_meta.iterrows():
        ep_idx = int(row["episode_index"])
        n_frames = int(row["length"])
        ds_from = int(row["dataset_from_index"])
        ds_to = int(row["dataset_to_index"])

        # Sample actions and states
        sample_every = max(1, n_frames // 50)
        actions = []
        states = []
        for i in range(ds_from, ds_to, sample_every):
            item = ds[i]
            actions.append(item["action"].numpy())
            states.append(item["observation.state"].numpy())

        actions = np.array(actions)
        states = np.array(states)

        # Gripper check
        grip_act = actions[:, 6]
        grip_state = states[:, 6]
        grip_min = float(grip_act.min())
        grip_max = float(grip_act.max())
        grip_start = float(grip_act[0])
        grip_end = float(grip_act[-1])

        # Pattern detection (gripper closes around bottle, not to 0)
        grip_wide_open = 0.07  # fully open > 7cm
        grip_drop = grip_max - grip_min  # how much it closes
        has_open_start = grip_start > grip_wide_open
        has_close = grip_drop > 0.03  # closes by > 3cm (bottle diameter)
        has_open_end = grip_end > grip_wide_open
        pattern_ok = has_open_start and has_close and has_open_end

        # Action smoothness: spike = single-frame delta far beyond local mean
        action_diffs = np.abs(np.diff(actions, axis=0))
        max_diff_per_dim = action_diffs.max(axis=0)
        mean_diff_per_dim = action_diffs.mean(axis=0)
        # A spike is a physically implausible single-step jump (> 0.8 rad ≈ 45°)
        spike_dims = [f"dim{d}" for d in range(action_dim)
                      if max_diff_per_dim[d] > 0.8]
        has_spike = len(spike_dims) > 0

        # State continuity: check for jumps
        state_diffs = np.abs(np.diff(states, axis=0))
        state_max_diff = state_diffs.max(axis=0)

        # Duration
        duration = n_frames / ds.fps

        # Result
        issues = []
        if not pattern_ok:
            if not has_open_start:
                issues.append(f"gripper NOT OPEN at start (grip={grip_start:.4f})")
            if not has_close:
                issues.append(f"gripper drop too small: {grip_drop:.4f} (need >0.03, bottle diameter)")
            if not has_open_end:
                issues.append(f"gripper NOT OPEN at end (grip={grip_end:.4f})")
        if has_spike:
            issues.append(f"action SPIKES in {', '.join(spike_dims)}")
        if float(state_max_diff.max()) > 1.0:
            issues.append(f"state JUMP max={float(state_max_diff.max()):.3f}")

        passed = len(issues) == 0

        # Print
        status = "PASS" if passed else "FAIL"
        print(f"Episode {ep_idx}: {n_frames:>4d} frames ({duration:.1f}s)  [{status}]")
        print(f"  Gripper: start={grip_start:.4f}  min={grip_min:.4f}  max={grip_max:.4f}  end={grip_end:.4f}")
        pattern_str = "open→close→open" if has_open_start and has_close and has_open_end else (
            "open→close" if has_open_start and has_close else "incomplete"
        )
        print(f"  Pattern: {pattern_str}")
        print(f"  Action max delta:  {[f'{d:.3f}' for d in max_diff_per_dim]}")
        print(f"  Action mean delta: {[f'{d:.3f}' for d in mean_diff_per_dim]}")
        if issues:
            for issue in issues:
                print(f"  ISSUE: {issue}")
        print()

        if passed:
            episodes_ok.append(ep_idx)
        else:
            episodes_fail.append(ep_idx)
            all_pass = False

    # Summary
    print("=" * 50)
    print(f"PASS: {len(episodes_ok)} episodes  {episodes_ok}")
    print(f"FAIL: {len(episodes_fail)} episodes  {episodes_fail}")
    print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAIL'}")

    if episodes_fail:
        print()
        print("Failing episodes should be discarded. Collect replacements.")
    else:
        print()
        print("All episodes meet quality standards. Ready for training.")

    return all_pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    check_episodes(dataset_root=args.root, repo_id=args.repo_id)


if __name__ == "__main__":
    main()
