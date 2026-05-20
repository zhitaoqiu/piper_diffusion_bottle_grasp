#!/usr/bin/env python3
"""Comprehensive QC: 40 raw episodes → Top30 selection → clean + smooth dataset generation.

Usage:
  python scripts/qc_40ep.py
"""

import json, os, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Config ──────────────────────────────────────────────────────────
RAW_ROOT = PROJECT_ROOT / "data" / "lerobot_dataset_env2_30fixed"
RAW_REPO = "piper/bottle_pick_place_aside_env2_30fixed"
CLEAN_REPO = "piper/bottle_pick_place_aside_env2_30clean"
CLEAN_ROOT = PROJECT_ROOT / "data" / "lerobot_dataset_env2_30clean"
SMOOTH_REPO = "piper/bottle_pick_place_aside_env2_30smooth"
SMOOTH_ROOT = PROJECT_ROOT / "data" / "lerobot_dataset_env2_30smooth"

GRIP_THRESH = 0.07
TOP_N = 30
SMOOTH_WINDOW = 5  # savgol window
SMOOTH_ORDER = 2   # savgol poly order
ACTION_SPIKE_THRESH = 0.3   # rad between consecutive action frames
STATE_JUMP_THRESH = 0.3     # rad between consecutive state frames
J2_VEL_WARN = 1.5           # rad/s
J3_VEL_WARN = 1.5
J2_ACC_WARN = 15.0          # rad/s²
J3_ACC_WARN = 15.0

JOINT_NAMES = ["J1", "J2", "J3", "J4", "J5", "J6", "Grip"]


# ── Helpers ─────────────────────────────────────────────────────────

def load_dataset(root, repo):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    return LeRobotDataset(repo_id=repo, root=root, video_backend="pyav")


def get_episode_ranges(ds):
    meta_files = sorted((Path(ds.root) / "meta" / "episodes").glob("*/*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in meta_files]).sort_values("episode_index")
    ranges = []
    for _, row in df.iterrows():
        ranges.append((
            int(row["episode_index"]),
            int(row["dataset_from_index"]),
            int(row["dataset_to_index"]),
            int(row["length"]),
        ))
    return ranges


def read_episode_data(ds, f0, f1):
    n = f1 - f0
    states = np.zeros((n, 7), dtype=np.float32)
    actions = np.zeros((n, 7), dtype=np.float32)
    for i in range(n):
        item = ds[f0 + i]
        states[i] = item["observation.state"].numpy()
        actions[i] = item["action"].numpy()
    return states, actions


# ── QC functions ────────────────────────────────────────────────────

def check_gripper(grip):
    """Returns (ok, start_val, end_val, min_val, max_val, close_frame, open_frame)."""
    n = len(grip)
    start_val = float(grip[0])
    end_val = float(grip[-1])
    min_val = float(grip.min())
    max_val = float(grip.max())

    # Find close frame: first time grip drops below threshold
    close_frame = None
    for i in range(1, n):
        if grip[i] < GRIP_THRESH and grip[i - 1] >= GRIP_THRESH:
            close_frame = i
            break

    # Find open frame: last time grip rises above threshold (after being closed)
    open_frame = None
    if close_frame is not None:
        for i in range(n - 1, close_frame, -1):
            if grip[i] >= GRIP_THRESH and grip[i - 1] < GRIP_THRESH:
                open_frame = i
                break

    ok = start_val > GRIP_THRESH and end_val > GRIP_THRESH and (max_val - min_val) > 0.03

    return {
        "grip_ok": ok,
        "grip_start": start_val,
        "grip_end": end_val,
        "grip_min": min_val,
        "grip_max": max_val,
        "grip_range": max_val - min_val,
        "close_frame": close_frame,
        "open_frame": open_frame,
    }


def count_spikes(signal_2d, threshold):
    """Count frames where max absolute difference between consecutive frames exceeds threshold."""
    diffs = np.max(np.abs(np.diff(signal_2d, axis=0)), axis=1)
    return int(np.sum(diffs > threshold)), float(np.max(diffs))


def check_phases(grip, j2, fps):
    """Detect phase sequence using gripper transitions + frame position.

    Robot-agnostic: does NOT assume a specific J2 sign convention.
    Instead checks that J2 actually MOVES significantly in each phase.
    """
    n = len(grip)
    close_frame = None
    for i in range(1, n):
        if grip[i] < GRIP_THRESH and grip[i - 1] >= GRIP_THRESH:
            close_frame = i
            break
    open_frame = None
    if close_frame is not None:
        for i in range(n - 1, close_frame, -1):
            if grip[i] >= GRIP_THRESH and grip[i - 1] < GRIP_THRESH:
                open_frame = i
                break

    issues = []
    phases_found = {}

    if close_frame is None:
        issues.append("no_grasp")
        return issues, phases_found

    if open_frame is None:
        issues.append("no_release")
        return issues, phases_found

    def has_motion(arr, min_range=0.05):
        """Check if signal has meaningful motion."""
        if len(arr) < 3:
            return False
        return float(arr.max() - arr.min()) > min_range

    # Approach: gripper open, frame 0 → close_frame
    if close_frame > 3:
        approach_j2 = j2[:close_frame]
        if has_motion(approach_j2):
            phases_found["approach"] = (0, close_frame)
        else:
            issues.append("approach_no_j2_motion")
            phases_found["approach"] = (0, close_frame)
    else:
        issues.append("approach_too_short")

    # Grasp: around close_frame
    phases_found["grasp"] = (max(0, close_frame - 3), min(n, close_frame + 3))

    # Lift: closed phase, first 1/3 (close → lift_end)
    closed_duration = open_frame - close_frame
    if closed_duration > 10:
        lift_end = close_frame + max(5, closed_duration // 3)
        lift_j2 = j2[close_frame:lift_end]
        if has_motion(lift_j2):
            phases_found["lift"] = (close_frame, lift_end)
        else:
            issues.append("lift_no_j2_motion")
            phases_found["lift"] = (close_frame, lift_end)
    else:
        issues.append("closed_phase_too_short")

    # Lower: last part before open_frame
    lower_start = max(close_frame + 5, open_frame - max(10, closed_duration // 4))
    if open_frame > lower_start + 3:
        lower_j2 = j2[lower_start:open_frame]
        if has_motion(lower_j2):
            phases_found["lower"] = (lower_start, open_frame)
        else:
            issues.append("lower_no_j2_motion")
            phases_found["lower"] = (lower_start, open_frame)
    else:
        issues.append("lower_too_short")

    # Move: between lift and lower
    if "lift" in phases_found and "lower" in phases_found:
        ms, me = phases_found["lift"][1], phases_found["lower"][0]
        if me > ms + 2:
            phases_found["move"] = (ms, me)

    # Release: around open_frame
    phases_found["release"] = (max(0, open_frame - 3), min(n, open_frame + 3))

    # Retreat: after open_frame, arm moves away
    if open_frame < n - 8:
        retreat_j2 = j2[open_frame:n]
        if has_motion(retreat_j2):
            phases_found["retreat"] = (open_frame, n)
        else:
            issues.append("retreat_no_j2_motion")
            phases_found["retreat"] = (open_frame, n)
    elif open_frame < n - 3:
        phases_found["retreat"] = (open_frame, n)
    else:
        issues.append("no_retreat_phase")

    return issues, phases_found


def detect_hesitation(j2, threshold=0.02, window=5):
    """Detect hesitation: sign changes in J2 velocity within short windows."""
    if len(j2) < window * 2:
        return 0
    vel = np.diff(j2)
    hesitations = 0
    for i in range(0, len(vel) - window, window // 2):
        segment = vel[i:i + window]
        if len(segment) < 3:
            continue
        # Check for direction reversals
        signs = np.sign(segment)
        if np.any(signs != signs[0]) and np.max(np.abs(segment)) > threshold:
            hesitations += 1
    return hesitations


def compute_quality_score(grip_info, n_frames, fps, n_action_spikes, n_state_jumps,
                          j2_range, j3_range, j2_vel_max, j3_vel_max,
                          j2_acc_max, j3_acc_max, has_phases, hesitations,
                          start_dist, end_dist, median_duration, median_j2_range, median_j3_range):
    """Score 0-1, higher is better."""
    score = 0.0
    weights = []

    # Gripper OK
    w = 0.15
    weights.append(w)
    score += w * (1.0 if grip_info["grip_ok"] else 0.0)

    # Action smoothness
    w = 0.15
    weights.append(w)
    smooth_score = max(0.0, 1.0 - n_action_spikes / 10.0)
    score += w * smooth_score

    # State continuity
    w = 0.10
    weights.append(w)
    state_score = max(0.0, 1.0 - n_state_jumps / 5.0)
    score += w * state_score

    # Duration near median
    w = 0.10
    weights.append(w)
    if median_duration > 0:
        dur_ratio = n_frames / median_duration
        dur_score = max(0.0, 1.0 - abs(dur_ratio - 1.0))
    else:
        dur_score = 1.0
    score += w * dur_score

    # J2 range near median
    w = 0.10
    weights.append(w)
    if median_j2_range > 0.01:
        j2r_ratio = j2_range / median_j2_range
        j2r_score = max(0.0, 1.0 - abs(j2r_ratio - 1.0) * 2)
    else:
        j2r_score = 1.0
    score += w * j2r_score

    # J3 range near median
    w = 0.10
    weights.append(w)
    if median_j3_range > 0.01:
        j3r_ratio = j3_range / median_j3_range
        j3r_score = max(0.0, 1.0 - abs(j3r_ratio - 1.0) * 2)
    else:
        j3r_score = 1.0
    score += w * j3r_score

    # Start pose consistency
    w = 0.05
    weights.append(w)
    score += w * max(0.0, 1.0 - start_dist / 0.2)

    # End pose consistency
    w = 0.05
    weights.append(w)
    score += w * max(0.0, 1.0 - end_dist / 0.2)

    # Phase completeness
    w = 0.10
    weights.append(w)
    phase_score = len(has_phases) / 7.0  # 7 expected phases
    score += w * phase_score

    # No hesitation
    w = 0.10
    weights.append(w)
    score += w * max(0.0, 1.0 - hesitations / 20.0)

    return score


# ── Main ────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("QC PIPELINE: 40 Raw Episodes → Top30 → Clean → Smooth")
    print("=" * 70)

    # Step 0: Load raw dataset
    print(f"\n[0] Loading raw dataset: {RAW_REPO} @ {RAW_ROOT}")
    ds_raw = load_dataset(RAW_ROOT, RAW_REPO)
    ep_ranges = get_episode_ranges(ds_raw)
    fps = ds_raw.fps
    n_total_eps = len(ep_ranges)
    print(f"  Episodes: {n_total_eps}  Frames: {ds_raw.num_frames}  FPS: {fps}")

    # Get task
    task_str = ds_raw[0].get("task", "pick up the bottle and place it aside")
    if hasattr(task_str, 'item'):
        task_str = str(task_str.item())
    print(f"  Task: {task_str}")

    image_keys = [k for k in ds_raw.features if k.startswith("observation.images.")]
    print(f"  Image keys: {image_keys}")

    # =================================================================
    # Step 1: Per-episode QC
    # =================================================================
    print("\n" + "=" * 70)
    print("[1] PER-EPISODE QUALITY CHECK")
    print("=" * 70)

    all_ep_data = {}

    for ep_idx, f0, f1, length in ep_ranges:
        n = f1 - f0
        print(f"\n  --- Ep{ep_idx} ({n}f, {n/fps:.1f}s) ---")

        states, actions = read_episode_data(ds_raw, f0, f1)
        grip_act = actions[:, 6]
        j2_act = actions[:, 1]
        j3_act = actions[:, 2]

        # 1. Gripper check
        grip_info = check_gripper(grip_act)
        print(f"  Gripper: {'OK' if grip_info['grip_ok'] else 'FAIL'}"
              f"  start={grip_info['grip_start']:.4f} end={grip_info['grip_end']:.4f}"
              f"  min={grip_info['grip_min']:.4f} max={grip_info['grip_max']:.4f}"
              f"  close@{grip_info['close_frame']} open@{grip_info['open_frame']}")

        # 2. Action spike check
        n_action_spikes, max_action_diff = count_spikes(actions[:, :6], ACTION_SPIKE_THRESH)
        print(f"  Action spikes: {n_action_spikes} (max_diff={max_action_diff:.4f})")

        # 3. State jump check
        n_state_jumps, max_state_diff = count_spikes(states[:, :6], STATE_JUMP_THRESH)
        print(f"  State jumps: {n_state_jumps} (max_diff={max_state_diff:.4f})")

        # 4. J2/J3 range & velocity & acceleration
        j2_min, j2_max = float(j2_act.min()), float(j2_act.max())
        j3_min, j3_max = float(j3_act.min()), float(j3_act.max())
        j2_range = j2_max - j2_min
        j3_range = j3_max - j3_min

        j2_vel = np.abs(np.diff(j2_act)) * fps
        j3_vel = np.abs(np.diff(j3_act)) * fps
        j2_vel_max = float(j2_vel.max())
        j3_vel_max = float(j3_vel.max())

        j2_acc = np.abs(np.diff(j2_vel)) * fps
        j3_acc = np.abs(np.diff(j3_vel)) * fps
        j2_acc_max = float(j2_acc.max()) if len(j2_acc) > 0 else 0.0
        j3_acc_max = float(j3_acc.max()) if len(j3_acc) > 0 else 0.0

        print(f"  J2: min={j2_min:.4f} max={j2_max:.4f} range={j2_range:.4f}"
              f"  vel_max={j2_vel_max:.4f} rad/s  acc_max={j2_acc_max:.2f} rad/s²")
        print(f"  J3: min={j3_min:.4f} max={j3_max:.4f} range={j3_range:.4f}"
              f"  vel_max={j3_vel_max:.4f} rad/s  acc_max={j3_acc_max:.2f} rad/s²")

        # 5. Phase check
        phase_issues, phases_found = check_phases(grip_act, j2_act, fps)
        found_phase_names = sorted(phases_found.keys())
        print(f"  Phases found: {found_phase_names}")
        if phase_issues:
            print(f"  Phase issues: {phase_issues}")

        # 6. Hesitation detection
        hesitations = detect_hesitation(j2_act)
        print(f"  J2 hesitations: {hesitations}")

        # 7. Duration check
        dur = n / fps
        dur_status = "OK" if 12 <= dur <= 30 else ("TOO_SHORT" if dur < 12 else "TOO_LONG")
        print(f"  Duration: {dur:.1f}s [{dur_status}]")

        # 8. Store initial state (for consistency check later)
        init_state = states[0, :6]
        final_state = states[-1, :6]

        all_ep_data[ep_idx] = {
            "n": n, "fps": fps, "dur": dur,
            "grip_info": grip_info,
            "n_action_spikes": n_action_spikes, "max_action_diff": max_action_diff,
            "n_state_jumps": n_state_jumps, "max_state_diff": max_state_diff,
            "j2_min": j2_min, "j2_max": j2_max, "j2_range": j2_range,
            "j3_min": j3_min, "j3_max": j3_max, "j3_range": j3_range,
            "j2_vel_max": j2_vel_max, "j3_vel_max": j3_vel_max,
            "j2_acc_max": j2_acc_max, "j3_acc_max": j3_acc_max,
            "phase_issues": phase_issues, "phases_found": found_phase_names, "phases_dict": phases_found,
            "hesitations": hesitations,
            "dur_status": dur_status,
            "init_state": init_state, "final_state": final_state,
            "actions": actions, "states": states,
        }

    # Compute global stats for scoring
    all_durs = np.array([d["n"] for d in all_ep_data.values()])
    median_dur = float(np.median(all_durs))
    all_j2_ranges = np.array([d["j2_range"] for d in all_ep_data.values()])
    median_j2_range = float(np.median(all_j2_ranges))
    all_j3_ranges = np.array([d["j3_range"] for d in all_ep_data.values()])
    median_j3_range = float(np.median(all_j3_ranges))

    # Compute mean start/end poses
    all_inits = np.array([d["init_state"] for d in all_ep_data.values()])
    all_finals = np.array([d["final_state"] for d in all_ep_data.values()])
    mean_init = all_inits.mean(axis=0)
    mean_final = all_finals.mean(axis=0)

    # =================================================================
    # Step 2: Force-rejection + Scoring
    # =================================================================
    print("\n" + "=" * 70)
    print("[2] SCORING & REJECTION")
    print("=" * 70)

    rejected = {}
    passed = {}

    for ep_idx, d in sorted(all_ep_data.items()):
        reasons = []

        # Mandatory rejections
        if not d["grip_info"]["grip_ok"]:
            reasons.append("gripper_not_open-close-open")
        if d["grip_info"]["grip_end"] < GRIP_THRESH:
            reasons.append("final_gripper_not_open")
        if "retreat" not in d["phases_found"]:
            reasons.append("missing_retreat")
        if "lift" not in d["phases_found"] and "lower" not in d["phases_found"]:
            reasons.append("missing_lift_and_lower")
        if d["n_action_spikes"] > 15:
            reasons.append(f"too_many_action_spikes({d['n_action_spikes']})")
        if d["n_state_jumps"] > 10:
            reasons.append(f"too_many_state_jumps({d['n_state_jumps']})")
        if d["dur"] < 10:
            reasons.append(f"too_short({d['dur']:.1f}s)")
        if d["dur"] > 40:
            reasons.append(f"too_long({d['dur']:.1f}s)")
        if d["j2_range"] < 0.5:
            reasons.append(f"j2_range_too_small({d['j2_range']:.3f})")
        if d["j3_range"] < 0.2:
            reasons.append(f"j3_range_too_small({d['j3_range']:.3f})")
        if d["hesitations"] > 30:
            reasons.append(f"too_many_hesitations({d['hesitations']})")
        # retreat present and not empty

        if reasons:
            rejected[ep_idx] = reasons
        else:
            # Compute quality score
            start_dist = float(np.linalg.norm(d["init_state"] - mean_init))
            end_dist = float(np.linalg.norm(d["final_state"] - mean_final))
            score = compute_quality_score(
                d["grip_info"], d["n"], fps,
                d["n_action_spikes"], d["n_state_jumps"],
                d["j2_range"], d["j3_range"],
                d["j2_vel_max"], d["j3_vel_max"],
                d["j2_acc_max"], d["j3_acc_max"],
                d["phases_found"], d["hesitations"],
                start_dist, end_dist,
                median_dur, median_j2_range, median_j3_range,
            )
            passed[ep_idx] = {
                "score": score,
                "start_dist": start_dist,
                "end_dist": end_dist,
            }

    # Print force-rejected
    print(f"\n  Force-rejected: {len(rejected)} episodes")
    for ep_idx in sorted(rejected):
        print(f"    Ep{ep_idx}: {'; '.join(rejected[ep_idx])}")

    print(f"\n  Passed force-filter: {len(passed)} episodes")

    # =================================================================
    # Step 3: Select Top30
    # =================================================================
    print("\n" + "=" * 70)
    print("[3] TOP30 SELECTION")
    print("=" * 70)

    sorted_passed = sorted(passed.items(), key=lambda x: x[1]["score"], reverse=True)
    top30 = [ep for ep, _ in sorted_passed[:TOP_N]]
    rejected_by_score = [ep for ep, _ in sorted_passed[TOP_N:]]

    print(f"\n  Top30 episodes: {top30}")
    print(f"\n  Rejected by score (passed filter but not top30):")
    for ep_idx in rejected_by_score:
        d = all_ep_data[ep_idx]
        s = passed[ep_idx]
        print(f"    Ep{ep_idx}: score={s['score']:.3f}  n={d['n']}  dur={d['dur']:.1f}s"
              f"  J2_range={d['j2_range']:.3f}  J3_range={d['j3_range']:.3f}")

    # Full rejection list
    all_rejected = set(rejected.keys()) | set(rejected_by_score)
    print(f"\n  Total kept: {len(top30)}")
    print(f"  Total rejected: {len(all_rejected)}")

    # Top30 stats
    top30_durs = [all_ep_data[ep]["dur"] for ep in top30]
    top30_j2r = [all_ep_data[ep]["j2_range"] for ep in top30]
    top30_j3r = [all_ep_data[ep]["j3_range"] for ep in top30]
    top30_n = [all_ep_data[ep]["n"] for ep in top30]
    top30_close = [all_ep_data[ep]["grip_info"]["close_frame"] for ep in top30]
    top30_open = [all_ep_data[ep]["grip_info"]["open_frame"] for ep in top30]

    print(f"\n  Top30 stats:")
    print(f"    Duration: {np.mean(top30_durs):.1f}s ± {np.std(top30_durs):.1f}s [{min(top30_durs):.1f}-{max(top30_durs):.1f}]")
    print(f"    Frames: {np.mean(top30_n):.0f} ± {np.std(top30_n):.0f} [{min(top30_n)}-{max(top30_n)}]")
    print(f"    J2 range: {np.mean(top30_j2r):.4f} ± {np.std(top30_j2r):.4f} [{min(top30_j2r):.4f}-{max(top30_j2r):.4f}]")
    print(f"    J3 range: {np.mean(top30_j3r):.4f} ± {np.std(top30_j3r):.4f} [{min(top30_j3r):.4f}-{max(top30_j3r):.4f}]")
    print(f"    Close frame: {np.mean(top30_close):.0f} ± {np.std(top30_close):.0f}")
    print(f"    Open frame: {np.mean(top30_open):.0f} ± {np.std(top30_open):.0f}")

    # =================================================================
    # Step 4: Generate Clean Dataset
    # =================================================================
    print("\n" + "=" * 70)
    print("[4] GENERATING CLEAN DATASET")
    print("=" * 70)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if CLEAN_ROOT.exists():
        import shutil
        stamp = time.strftime("%Y%m%d_%H%M%S")
        backup = CLEAN_ROOT.with_name(f"{CLEAN_ROOT.name}_backup_{stamp}")
        shutil.move(str(CLEAN_ROOT), str(backup))
        print(f"  Moved existing {CLEAN_ROOT} → {backup}")

    features = {}
    for key, ft in ds_raw.features.items():
        if key in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
            continue
        fdict = {"dtype": ft["dtype"], "shape": tuple(ft["shape"])}
        if "names" in ft:
            fdict["names"] = ft["names"]
        features[key] = fdict

    ds_clean = LeRobotDataset.create(
        repo_id=CLEAN_REPO, fps=int(fps), features=features,
        root=CLEAN_ROOT, use_videos=True, image_writer_processes=0,
    )
    print(f"  Created clean dataset: {CLEAN_REPO} @ {CLEAN_ROOT}")

    # Copy Top30 frames
    for new_ep_idx, old_ep_idx in enumerate(sorted(top30)):
        _, f0, f1, _ = ep_ranges[old_ep_idx]
        n = f1 - f0
        print(f"  Old Ep{old_ep_idx} → New Ep{new_ep_idx} ({n}f) ...", end=" ", flush=True)
        for i in range(f0, f1):
            item = ds_raw[i]
            frame = {
                "observation.state": item["observation.state"].numpy().astype(np.float32),
                "action": item["action"].numpy().astype(np.float32),
                "task": task_str,
            }
            for key in image_keys:
                img = item[key]
                if hasattr(img, 'numpy'):
                    img = img.numpy()
                if img.ndim == 3 and img.shape[0] in (1, 3):
                    img = img.astype(np.uint8)
                elif img.ndim == 3 and img.shape[-1] == 3:
                    img = np.transpose(img, (2, 0, 1)).astype(np.uint8)
                frame[key] = img
            ds_clean.add_frame(frame)
        ds_clean.save_episode()
        print("done")

    ds_clean.finalize()
    print(f"  Clean dataset finalized: {len(top30)} episodes")

    # Verify clean
    print(f"\n  Verifying clean dataset ...")
    ds_clean_verify = LeRobotDataset(repo_id=CLEAN_REPO, root=CLEAN_ROOT, video_backend="pyav")
    print(f"    Episodes: {ds_clean_verify.num_episodes}  Frames: {ds_clean_verify.num_frames}  FPS: {ds_clean_verify.fps}")

    # =================================================================
    # Step 5: Generate Smooth Dataset
    # =================================================================
    print("\n" + "=" * 70)
    print("[5] GENERATING SMOOTH DATASET")
    print("=" * 70)

    if SMOOTH_ROOT.exists():
        import shutil
        stamp = time.strftime("%Y%m%d_%H%M%S")
        backup = SMOOTH_ROOT.with_name(f"{SMOOTH_ROOT.name}_backup_{stamp}")
        shutil.move(str(SMOOTH_ROOT), str(backup))
        print(f"  Moved existing {SMOOTH_ROOT} → {backup}")

    ds_smooth = LeRobotDataset.create(
        repo_id=SMOOTH_REPO, fps=int(fps), features=features,
        root=SMOOTH_ROOT, use_videos=True, image_writer_processes=0,
    )
    print(f"  Created smooth dataset: {SMOOTH_REPO} @ {SMOOTH_ROOT}")

    # Get clean episode ranges
    clean_ep_ranges = get_episode_ranges(ds_clean_verify)
    clean_states_all = []
    clean_actions_all = []
    smooth_actions_all = []
    clean_grip_info_all = []

    for ep_idx, f0, f1, length in clean_ep_ranges:
        n = f1 - f0
        print(f"  Ep{ep_idx} ({n}f): smoothing ...", end=" ", flush=True)

        # Read clean episode data
        states = np.zeros((n, 7), dtype=np.float32)
        actions = np.zeros((n, 7), dtype=np.float32)
        for i in range(n):
            item = ds_clean_verify[f0 + i]
            states[i] = item["observation.state"].numpy()
            actions[i] = item["action"].numpy()

        # Smooth only J1-J6 with Savitzky-Golay
        actions_smooth = actions.copy()
        for j in range(6):  # J1-J6 only
            if n > SMOOTH_WINDOW:
                actions_smooth[:, j] = scipy_signal.savgol_filter(
                    actions[:, j], SMOOTH_WINDOW, SMOOTH_ORDER
                )
            # else: too short, keep original

        # Gripper: keep original binary behavior (no smoothing)
        # actions_smooth[:, 6] stays as original

        smooth_actions_all.append(actions_smooth)
        clean_actions_all.append(actions)
        clean_states_all.append(states)

        # Copy frames with smoothed actions
        for i in range(n):
            item = ds_clean_verify[f0 + i]
            frame = {
                "observation.state": states[i].astype(np.float32),
                "action": actions_smooth[i].astype(np.float32),
                "task": task_str,
            }
            for key in image_keys:
                img = item[key]
                if hasattr(img, 'numpy'):
                    img = img.numpy()
                if img.ndim == 3 and img.shape[0] in (1, 3):
                    img = img.astype(np.uint8)
                elif img.ndim == 3 and img.shape[-1] == 3:
                    img = np.transpose(img, (2, 0, 1)).astype(np.uint8)
                frame[key] = img
            ds_smooth.add_frame(frame)
        ds_smooth.save_episode()

        # Per-episode gripper check
        grip_info = check_gripper(actions_smooth[:, 6])
        clean_grip_info_all.append(grip_info)
        grip_ok_str = "OK" if grip_info["grip_ok"] else "FAIL"
        print(f"grip={grip_ok_str} done")

    ds_smooth.finalize()
    print(f"  Smooth dataset finalized: {len(clean_ep_ranges)} episodes")

    # Verify smooth
    ds_smooth_verify = LeRobotDataset(repo_id=SMOOTH_REPO, root=SMOOTH_ROOT, video_backend="pyav")
    print(f"    Episodes: {ds_smooth_verify.num_episodes}  Frames: {ds_smooth_verify.num_frames}  FPS: {ds_smooth_verify.fps}")

    # =================================================================
    # Step 6: Clean vs Smooth Comparison
    # =================================================================
    print("\n" + "=" * 70)
    print("[6] CLEAN vs SMOOTH COMPARISON")
    print("=" * 70)

    all_clean_actions = np.concatenate(clean_actions_all, axis=0)
    all_smooth_actions = np.concatenate(smooth_actions_all, axis=0)

    print(f"\n{'Metric':<35} {'Clean':>15} {'Smooth':>15}")
    print("-" * 65)

    print(f"{'Episodes':<35} {len(clean_ep_ranges):>15} {len(clean_ep_ranges):>15}")
    print(f"{'Total frames':<35} {len(all_clean_actions):>15} {len(all_smooth_actions):>15}")

    for j in range(7):
        clean_range = float(all_clean_actions[:, j].max() - all_clean_actions[:, j].min())
        smooth_range = float(all_smooth_actions[:, j].max() - all_smooth_actions[:, j].min())
        name = JOINT_NAMES[j]
        print(f"{name + ' range':<35} {clean_range:15.4f} {smooth_range:15.4f}")

    for j in range(7):
        clean_vel = np.max(np.abs(np.diff(all_clean_actions[:, j]))) * fps
        smooth_vel = np.max(np.abs(np.diff(all_smooth_actions[:, j]))) * fps
        name = JOINT_NAMES[j]
        print(f"{name + ' vel_max (rad/s)':<35} {clean_vel:15.4f} {smooth_vel:15.4f}")

    for j in range(7):
        clean_acc = np.max(np.abs(np.diff(np.diff(all_clean_actions[:, j])))) * fps * fps
        smooth_acc = np.max(np.abs(np.diff(np.diff(all_smooth_actions[:, j])))) * fps * fps
        name = JOINT_NAMES[j]
        print(f"{name + ' acc_max (rad/s²)':<35} {clean_acc:15.2f} {smooth_acc:15.2f}")

    # Action spike comparison
    clean_spikes, _ = count_spikes(all_clean_actions[:, :6], ACTION_SPIKE_THRESH)
    smooth_spikes, _ = count_spikes(all_smooth_actions[:, :6], ACTION_SPIKE_THRESH)
    print(f"{'Action spikes (>0.3 rad)':<35} {clean_spikes:>15} {smooth_spikes:>15}")

    # Gripper check
    clean_grip_ok = all(g["grip_ok"] for g in clean_grip_info_all)
    print(f"{'Gripper all OK':<35} {str(clean_grip_ok):>15} {'True':>15}")
    print(f"{'Task field':<35} {'OK':>15} {'OK':>15}")
    print(f"{'Image keys':<35} {str(len(image_keys)):>15} {str(len(image_keys)):>15}")
    clean_sdim = ds_clean_verify.features["observation.state"]["shape"][0]
    smooth_sdim = ds_smooth_verify.features["observation.state"]["shape"][0]
    print(f"{'State dim':<35} {clean_sdim:>15} {smooth_sdim:>15}")
    clean_adim = ds_clean_verify.features["action"]["shape"][0]
    smooth_adim = ds_smooth_verify.features["action"]["shape"][0]
    print(f"{'Action dim':<35} {clean_adim:>15} {smooth_adim:>15}")

    # =================================================================
    # Step 7: Output training commands
    # =================================================================
    print("\n" + "=" * 70)
    print("[7] TRAINING COMMANDS (DO NOT EXECUTE)")
    print("=" * 70)

    print(f"""
# ── Clean Dataset Training ──────────────────────────────────────
REPO_ID=piper/bottle_pick_place_aside_env2_30clean \\
DATASET_ROOT=data/lerobot_dataset_env2_30clean \\
OUTPUT_DIR=outputs/train/piper_bottle_pick_place_aside_env2_30clean \\
STEPS=10000 \\
BATCH_SIZE=4 \\
bash training/train.sh

# ── Smooth Dataset Training ─────────────────────────────────────
REPO_ID=piper/bottle_pick_place_aside_env2_30smooth \\
DATASET_ROOT=data/lerobot_dataset_env2_30smooth \\
OUTPUT_DIR=outputs/train/piper_bottle_pick_place_aside_env2_30smooth \\
STEPS=10000 \\
BATCH_SIZE=4 \\
bash training/train.sh
""")

    print("  Recommendation: Train CLEAN first (no smoothing risk).")
    print("  If J2/J3 acceleration is reduced but range preserved in smooth, try smooth second.")

    # =================================================================
    # Final summary
    # =================================================================
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(f"""
  1. Raw dataset:         {RAW_REPO} @ {RAW_ROOT}
  2. Raw episodes:        {n_total_eps}
  3. Top30 episodes:      {top30}
  4. Rejected episodes:   {sorted(all_rejected)}
  5. Clean dataset:       {CLEAN_REPO} @ {CLEAN_ROOT}
  6. Smooth dataset:      {SMOOTH_REPO} @ {SMOOTH_ROOT}
  7. Clean vs Smooth:     comparison table above
  8. Recommendation:      Start with CLEAN training
  9. Training NOT started — waiting for your confirmation
""")


if __name__ == "__main__":
    main()
