#!/usr/bin/env python3
"""Rank all episodes by quality and select the best N.

Usage:
  python scripts/rank_episodes.py \
    --repo-id piper/bottle_pick_place_aside \
    --root data/lerobot_dataset \
    --top-n 10
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

JOINT_NAMES = ["J1", "J2", "J3", "J4", "J5", "J6", "Grip"]


def score_episode(ds, ep_idx, ds_from, ds_to):
    """Score a single episode. Returns dict with metrics and a total score."""
    n_frames = ds_to - ds_from
    duration = n_frames / ds.fps

    result = {
        "episode": ep_idx,
        "frames": n_frames,
        "duration_s": round(duration, 1),
        "pass": True,
        "disqualify": [],
        "warnings": [],
        "score": 0.0,
    }

    # Load all actions and states for this episode
    actions = []
    states = []
    grip_actions = []
    for i in range(ds_from, ds_to):
        item = ds[i]
        a = item["action"].numpy()
        s = item["observation.state"].numpy()
        actions.append(a)
        states.append(s)
        grip_actions.append(float(a[6]))

    actions = np.array(actions)
    states = np.array(states)
    grip_actions = np.array(grip_actions)

    # === 1. Gripper pattern (MUST PASS) ===
    grip_start = float(grip_actions[0])
    grip_end = float(grip_actions[-1])
    grip_min = float(grip_actions.min())
    grip_max = float(grip_actions.max())
    grip_drop = grip_max - grip_min

    has_open_start = grip_start > 0.07
    has_close = grip_drop > 0.03
    has_open_end = grip_end > 0.07

    result["grip_start"] = round(grip_start, 4)
    result["grip_end"] = round(grip_end, 4)
    result["grip_min"] = round(grip_min, 4)
    result["grip_max"] = round(grip_max, 4)
    result["grip_drop"] = round(grip_drop, 4)

    if not has_open_start:
        result["disqualify"].append(f"gripper NOT OPEN at start ({grip_start:.4f})")
        result["pass"] = False
    if not has_close:
        result["disqualify"].append(f"gripper never closed (drop={grip_drop:.4f}, need >0.03)")
        result["pass"] = False
    if not has_open_end:
        result["disqualify"].append(f"gripper NOT OPEN at end ({grip_end:.4f})")
        result["pass"] = False

    # === 2. Duration check ===
    if n_frames < 100:
        result["warnings"].append(f"too SHORT: {n_frames}f = {duration:.1f}s")
    elif n_frames < 150:
        result["warnings"].append(f"short: {n_frames}f = {duration:.1f}s (ideal >150)")
    elif n_frames > 300:
        result["warnings"].append(f"too LONG: {n_frames}f = {duration:.1f}s")

    # === 3. Action spikes ===
    action_diffs = np.abs(np.diff(actions, axis=0))
    max_diff_per_dim = action_diffs.max(axis=0)
    mean_diff_per_dim = action_diffs.mean(axis=0)
    spike_dims = []
    for d in range(7):
        if max_diff_per_dim[d] > 0.8:
            spike_dims.append(f"{JOINT_NAMES[d]}={max_diff_per_dim[d]:.3f}")
    if spike_dims:
        result["disqualify"].append(f"action SPIKE: {', '.join(spike_dims)}")
        result["pass"] = False

    result["max_action_diff"] = round(float(max_diff_per_dim.max()), 4)
    result["mean_action_diff"] = round(float(mean_diff_per_dim.mean()), 4)

    # === 4. State continuity ===
    state_diffs = np.abs(np.diff(states, axis=0))
    state_max_diff = state_diffs.max(axis=0)
    if float(state_max_diff.max()) > 1.0:
        result["disqualify"].append(f"state JUMP max={float(state_max_diff.max()):.3f}")
        result["pass"] = False
    result["max_state_diff"] = round(float(state_max_diff.max()), 4)

    # === 5. Action smoothness score ===
    # Lower variation in action diffs = smoother
    action_cv = float(np.std(action_diffs) / (np.mean(action_diffs) + 1e-8))
    result["action_cv"] = round(action_cv, 2)

    # === 6. Gripper smoothness ===
    grip_diffs = np.abs(np.diff(grip_actions))
    grip_jerk = float(np.diff(grip_diffs).std()) if len(grip_diffs) > 1 else 0
    result["grip_jerk"] = round(grip_jerk, 6)

    # === Compute score (continuous, not bucketed) ===
    if not result["pass"]:
        result["score"] = -1  # disqualified
    else:
        # 1. Duration: ideal 170-220 frames (17-22s pick-and-place)
        #    Linear falloff outside this range
        ideal_center = 195
        dur_score = max(0.0, 1.0 - abs(n_frames - ideal_center) / 100)

        # 2. Gripper drop: prefer 0.045-0.055 (solid grip)
        #    Linear falloff outside
        ideal_drop = 0.05
        grip_score = max(0.0, 1.0 - abs(grip_drop - ideal_drop) / 0.03)

        # 3. Smoothness: lower CV = better. Range ~2-3
        smooth_score = max(0.0, 1.0 - (action_cv - 2.0) / 2.0)

        # 4. Gripper jitter: lower = better. Linear 0 to 0.005
        jerk_score = max(0.0, 1.0 - grip_jerk / 0.005)

        result["score"] = round(
            0.25 * dur_score + 0.25 * grip_score + 0.25 * smooth_score + 0.25 * jerk_score,
            4
        )
        result["scores_detail"] = {
            "dur": round(dur_score, 4),
            "grip": round(grip_score, 4),
            "smooth": round(smooth_score, 4),
            "jerk": round(jerk_score, 4),
        }

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(
        repo_id=args.repo_id,
        root=args.root,
        video_backend="pyav",
    )

    print(f"Dataset: {args.repo_id}")
    print(f"Episodes: {ds.num_episodes}  Frames: {ds.num_frames}  FPS: {ds.fps}")
    print()

    meta_files = sorted((args.root / "meta" / "episodes").glob("*/*.parquet"))
    dfs = [pd.read_parquet(f) for f in meta_files]
    ep_meta = pd.concat(dfs, ignore_index=True).sort_values("episode_index")

    all_results = []
    for _, row in ep_meta.iterrows():
        ep_idx = int(row["episode_index"])
        ds_from = int(row["dataset_from_index"])
        ds_to = int(row["dataset_to_index"])

        r = score_episode(ds, ep_idx, ds_from, ds_to)
        all_results.append(r)

    # Sort by score (pass first, then by score descending)
    passed = [r for r in all_results if r["pass"]]
    failed = [r for r in all_results if not r["pass"]]

    passed.sort(key=lambda r: r["score"], reverse=True)
    failed.sort(key=lambda r: r["episode"])

    # Print all results
    header = f"{'Ep':>4} {'Frames':>6} {'Dur':>5} {'GripSt':>8} {'GripMin':>8} {'GripMax':>8} {'GripEnd':>8} {'Drop':>6} {'MaxAΔ':>8} {'CV':>6} {'Jerk':>8} {'Score':>6} {'Status'}"
    print(header)
    print("-" * len(header))

    for r in passed + failed:
        status = "PASS" if r["pass"] else "FAIL"
        line = (
            f"{r['episode']:>4} {r['frames']:>6} {r['duration_s']:>4.1f}s "
            f"{r['grip_start']:>8.4f} {r['grip_min']:>8.4f} {r['grip_max']:>8.4f} "
            f"{r['grip_end']:>8.4f} {r['grip_drop']:>6.4f} "
            f"{r['max_action_diff']:>8.4f} {r['action_cv']:>6.2f} "
            f"{r['grip_jerk']:>8.6f} {r['score']:>6.3f} {status}"
        )
        print(line)
        if r["disqualify"]:
            for dq in r["disqualify"]:
                print(f"       FAIL: {dq}")
        if r["warnings"]:
            for w in r["warnings"]:
                print(f"       WARN: {w}")

    # Summary
    print()
    print("=" * 60)
    print(f"PASS: {len(passed)}  FAIL: {len(failed)}")

    # Top N selection
    print()
    n_select = min(args.top_n, len(passed))
    if n_select < args.top_n:
        print(f"WARNING: Only {len(passed)} episodes PASS quality check, need {args.top_n}!")
        # Consider borderline failed episodes
        borderline = [r for r in failed if len(r["disqualify"]) == 1]
        print(f"  Borderline (1 issue): {[r['episode'] for r in borderline]}")
    else:
        selected = passed[:n_select]
        selected.sort(key=lambda r: r["episode"])
        print(f"Selected top {n_select} episodes: {[r['episode'] for r in selected]}")
        print()
        print("To extract these episodes into a new dataset, run:")
        print(f"  python scripts/extract_top_episodes.py \\")
        print(f"    --repo-id {args.repo_id} \\")
        print(f"    --root {args.root} \\")
        print(f"    --episodes {' '.join(str(r['episode']) for r in selected)} \\")
        print(f"    --target-repo {args.repo_id}_top{n_select} \\")
        print(f"    --target-root data/lerobot_dataset_top{n_select}")

    # Print rank-ordered list for manual review
    print()
    print("Rank-ordered PASS list (best first):")
    for i, r in enumerate(passed):
        d = r.get("scores_detail", {})
        print(f"  #{i+1}: Ep{r['episode']} score={r['score']:.4f} "
              f"frames={r['frames']} grip_drop={r['grip_drop']:.4f} "
              f"cv={r['action_cv']:.2f} "
              f"dur={d.get('dur',0):.3f} grip={d.get('grip',0):.3f} "
              f"smooth={d.get('smooth',0):.3f} jerk={d.get('jerk',0):.3f}")


if __name__ == "__main__":
    main()
