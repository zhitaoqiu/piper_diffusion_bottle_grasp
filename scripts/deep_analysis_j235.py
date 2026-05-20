#!/usr/bin/env python3
"""Deep analysis of J2/J3/J5 direction mismatches from 5-episode evaluation.

Checks multiple direction metrics beyond mean-based comparison:
  - start-to-end net displacement
  - phase-based (gripper-driven segmentation) direction
  - derivative sign agreement

Generates per-episode GT vs Pred plots saved to outputs/eval/5ep_gt_pred_plots/
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import safetensors.torch
import torch
from tqdm import tqdm

JOINT_NAMES = ["J1", "J2", "J3", "J4", "J5", "J6", "Grip"]
TARGET_JOINTS = ["J2", "J3", "J5"]
TARGET_IDX = {"J2": 1, "J3": 2, "J5": 4}
GRIP_IDX = 6


def segment_phases(gripper_actions, fps=10):
    """Segment episode into pick-and-place phases based on gripper action.

    Uses gripper action values and their derivative to detect:
      0: approach     – gripper fully open, moving toward bottle
      1: close/hold   – gripper closing around bottle + holding
      2: retreat      – gripper opening + arm moving back

    Returns list of (phase_name, start_frame, end_frame).
    """
    n = len(gripper_actions)
    g = np.array(gripper_actions)

    from scipy.ndimage import uniform_filter1d
    g_smooth = uniform_filter1d(g, size=5)

    grip_open_thresh = 0.07
    grip_closed_thresh = 0.06

    is_closed = g_smooth < grip_closed_thresh
    closed_frames = np.where(is_closed)[0]

    if len(closed_frames) == 0:
        return [("full_traj", 0, n - 1)]

    close_start = closed_frames[0]
    close_end = closed_frames[-1]

    # Backtrack to find when closing started
    for i in range(close_start, max(0, close_start - 30), -1):
        if g_smooth[i] > 0.085:
            close_start = i
            break

    # Forward track to find when opening ended
    for i in range(close_end, min(n - 1, close_end + 30)):
        if g_smooth[i] > 0.085:
            close_end = i
            break

    phases = []
    approach_end = max(0, close_start - 5)
    if approach_end > 5:
        phases.append(("approach", 0, approach_end))
    phases.append(("close_hold", approach_end, close_end))
    retreat_start = min(n - 1, close_end + 5)
    if retreat_start < n - 5:
        phases.append(("retreat", retreat_start, n - 1))

    return phases


def compute_direction_metrics(gt, pred):
    """Compute multiple direction agreement metrics."""
    gt = np.asarray(gt, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)

    net_gt = gt[-1] - gt[0]
    net_pred = pred[-1] - pred[0]

    gt_diff = np.diff(gt)
    pred_diff = np.diff(pred)

    eps = 1e-6
    mask = (np.abs(gt_diff) > eps) & (np.abs(pred_diff) > eps)
    if mask.sum() > 0:
        deriv_agree = (np.sign(gt_diff[mask]) == np.sign(pred_diff[mask])).mean()
    else:
        deriv_agree = 1.0

    return {
        "net_disp_gt": float(net_gt),
        "net_disp_pred": float(net_pred),
        "net_disp_same_sign": (net_gt * net_pred) >= 0,
        "net_disp_diff": float(abs(net_gt - net_pred)),
        "deriv_sign_agree": float(deriv_agree),
        "abs_net_disp_gt": float(abs(net_gt)),
    }


def load_policy(checkpoint_dir, device="cuda"):
    """Load a DiffusionPolicy from a checkpoint directory."""
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.configs.policies import PreTrainedConfig

    checkpoint_dir = Path(checkpoint_dir)

    # Import DiffusionConfig to register it in PreTrainedConfig's choice registry
    cfg = PreTrainedConfig.from_pretrained(str(checkpoint_dir))

    policy = DiffusionPolicy(cfg)
    state_dict = safetensors.torch.load_file(
        str(checkpoint_dir / "model.safetensors"), device=device
    )
    policy.load_state_dict(state_dict)
    policy.to(device)
    policy.eval()
    return policy, cfg


def load_dataset(root, repo_id, episodes=None):
    """Load LeRobot dataset."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    return LeRobotDataset(
        repo_id=repo_id,
        root=root,
        episodes=episodes,
        video_backend="pyav",
    )


def run_inference(policy, config, dataset, device="cuda"):
    """Run inference on all frames using policy.diffusion.generate_actions()."""
    from lerobot.policies.diffusion.modeling_diffusion import OBS_IMAGES, OBS_STATE

    n_obs_steps = config.n_obs_steps
    image_keys = list(config.image_features.keys())

    ep_meta = sorted(
        Path(dataset.root).glob("meta/episodes/*/*.parquet")
    )[0]
    import pandas as pd
    meta = pd.read_parquet(ep_meta).sort_values("episode_index")

    all_gt = []
    all_pred = []
    all_grip = []

    n_episodes = len(meta)

    for ep_i in range(n_episodes):
        row = meta.iloc[ep_i]
        ds_from = int(row["dataset_from_index"])
        ds_to = int(row["dataset_to_index"])

        ep_gt = []
        ep_pred = []
        ep_grip = []

        # Pre-load all frames for this episode
        states_list = []
        images_list = []
        actions_list = []

        for i in range(ds_from, ds_to):
            item = dataset[i]
            state = item["observation.state"].numpy().astype(np.float32)
            action = item["action"].numpy().astype(np.float32)

            # Get image (first image key)
            img_key = image_keys[0]
            img = item[img_key]
            if isinstance(img, torch.Tensor):
                img = img.numpy()
            if img.ndim == 3 and img.shape[0] in (1, 3):
                img = img.astype(np.float32) / 255.0
            elif img.ndim == 3 and img.shape[-1] == 3:
                img = np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.0

            states_list.append(state)
            images_list.append(img)
            actions_list.append(action)

        n_frames = len(states_list)

        # Run inference frame by frame with n_obs_steps history
        for t in range(n_frames):
            action_gt = actions_list[t]
            ep_gt.append(action_gt)
            ep_grip.append(float(action_gt[GRIP_IDX]))

            # Build observation window
            obs_start = max(0, t - n_obs_steps + 1)
            obs_count = t - obs_start + 1

            # Pad by repeating the first observation
            obs_states = np.zeros((n_obs_steps, 7), dtype=np.float32)
            obs_imgs = np.zeros((n_obs_steps, 3, 480, 640), dtype=np.float32)

            for j in range(n_obs_steps):
                src_idx = max(0, t - (n_obs_steps - 1 - j))
                obs_states[j] = states_list[src_idx]
                obs_imgs[j] = images_list[src_idx]

            # Build batch: (B=1, n_obs_steps, ...)
            # Build tensors with correct shapes:
            # OBS_STATE: (B=1, n_obs_steps, state_dim=7)
            # OBS_IMAGES: (B=1, n_obs_steps, num_cameras=1, C=3, H=480, W=640)
            batch = {
                OBS_STATE: torch.from_numpy(obs_states).unsqueeze(0).to(device),
                OBS_IMAGES: torch.from_numpy(obs_imgs).unsqueeze(0).unsqueeze(2).to(device),
            }

            with torch.no_grad():
                actions = policy.diffusion.generate_actions(batch)

            # actions shape: (1, n_action_steps, 7)
            pred_action = actions[0, 0].cpu().numpy().astype(np.float32)
            ep_pred.append(pred_action)

        all_gt.append(np.array(ep_gt, dtype=np.float32))
        all_pred.append(np.array(ep_pred, dtype=np.float32))
        all_grip.append(np.array(ep_grip, dtype=np.float32))

        print(f"  Ep{ep_i}: {n_frames} frames, GT range [{ds_from}, {ds_to})")

    return all_gt, all_pred, all_grip


def analyze_and_plot(all_gt, all_pred, all_grip, output_dir, fps=10):
    """Run full analysis and generate plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_eps = len(all_gt)
    results = {}
    lines = []

    lines.append("=" * 80)
    lines.append("J2/J3/J5 DEEP DIRECTION ANALYSIS")
    lines.append("=" * 80)

    for ep_idx in range(n_eps):
        gt = all_gt[ep_idx]
        pred = all_pred[ep_idx]
        grip = all_grip[ep_idx]
        n_frames = len(gt)

        lines.append(f"\n{'='*60}")
        lines.append(f"EPISODE {ep_idx}  ({n_frames} frames, {n_frames/fps:.1f}s)")
        lines.append(f"{'='*60}")

        # Segment phases
        phases = segment_phases(grip, fps)
        lines.append(f"\nPhases detected: {[(p[0], p[1], p[2]) for p in phases]}")

        ep_result = {"phases": {}, "overall": {}}

        # ---- Overall per-joint analysis ----
        lines.append(f"\n{'Joint':<6} {'GT start':>10} {'GT end':>10} {'Net GT':>10} "
                     f"{'Pred start':>10} {'Pred end':>10} {'Net Pred':>10} "
                     f"{'SameSign':>8} {'DerivAgree':>10} {'AbsNetGT':>10}")
        lines.append("-" * 96)

        for jname in JOINT_NAMES:
            jidx = JOINT_NAMES.index(jname)
            j_gt = gt[:, jidx]
            j_pred = pred[:, jidx]
            dm = compute_direction_metrics(j_gt, j_pred)
            ep_result["overall"][jname] = dm

            lines.append(
                f"{jname:<6} {j_gt[0]:10.4f} {j_gt[-1]:10.4f} {dm['net_disp_gt']:+10.4f} "
                f"{j_pred[0]:10.4f} {j_pred[-1]:10.4f} {dm['net_disp_pred']:+10.4f} "
                f"{'OK' if dm['net_disp_same_sign'] else 'WRONG':>8} "
                f"{dm['deriv_sign_agree']:10.3f} {dm['abs_net_disp_gt']:10.4f}"
            )

        # ---- Per-phase analysis for J2, J3, J5 ----
        lines.append(f"\n--- Per-Phase Analysis ---")

        for phase_name, p_start, p_end in phases:
            p_gt = gt[p_start:p_end+1]
            p_pred = pred[p_start:p_end+1]

            if len(p_gt) < 3:
                continue

            lines.append(f"\n  Phase '{phase_name}' (frames {p_start}-{p_end}, {len(p_gt)} frames):")

            phase_data = {}
            for jname in TARGET_JOINTS:
                jidx = JOINT_NAMES.index(jname)
                j_gt = p_gt[:, jidx]
                j_pred = p_pred[:, jidx]
                mse = np.mean((j_gt - j_pred) ** 2)
                r = np.corrcoef(j_gt, j_pred)[0, 1] if len(j_gt) > 2 else 0
                dm = compute_direction_metrics(j_gt, j_pred)

                phase_data[jname] = {"mse": mse, "r": r, **dm}

                lines.append(
                    f"    {jname}: MSE={mse:.6f}  r={r:+.4f}  "
                    f"net_gt={dm['net_disp_gt']:+.4f}  net_pred={dm['net_disp_pred']:+.4f}  "
                    f"same_sign={'OK' if dm['net_disp_same_sign'] else 'WRONG'}  "
                    f"deriv_agree={dm['deriv_sign_agree']:.3f}"
                )

            ep_result["phases"][phase_name] = phase_data

        results[f"ep{ep_idx}"] = ep_result

        # ---- Generate plot ----
        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
        colors = ['#e8f5e9', '#fff3e0', '#e3f2fd', '#fce4ec', '#f3e5f5']

        for ax_i, jname in enumerate(["J2", "J3", "J5"]):
            ax = axes[ax_i]
            jidx = JOINT_NAMES.index(jname)
            frames = np.arange(n_frames)

            ax.plot(frames, gt[:, jidx], 'b-', linewidth=1.5, alpha=0.8, label='GT')
            ax.plot(frames, pred[:, jidx], 'r--', linewidth=1.5, alpha=0.8, label='Pred')
            ax.set_ylabel(f'{jname} (rad)')
            ax.legend(loc='upper right', fontsize=7)
            ax.grid(True, alpha=0.3)

            for pi, (pname, ps, pe) in enumerate(phases):
                ax.axvspan(ps, pe, alpha=0.15, color=colors[pi % len(colors)])
                mid = (ps + pe) / 2
                ymax = ax.get_ylim()[1]
                ax.text(mid, ymax * 0.95, pname[:8],
                        ha='center', va='top', fontsize=6,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

        # Gripper subplot
        ax = axes[3]
        frames = np.arange(n_frames)
        ax.plot(frames, grip, 'g-', linewidth=1.5, label='Gripper GT')
        ax.axhline(y=0.07, color='gray', linestyle=':', alpha=0.5, label='open thresh')
        ax.axhline(y=0.06, color='gray', linestyle='--', alpha=0.5, label='closed thresh')
        ax.set_ylabel('Gripper (m)')
        ax.set_xlabel('Frame')
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3)
        for pi, (pname, ps, pe) in enumerate(phases):
            ax.axvspan(ps, pe, alpha=0.15, color=colors[pi % len(colors)])

        fig.suptitle(f'Episode {ep_idx} — J2/J3/J5 GT vs Pred with Gripper Phases',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        fig.savefig(output_dir / f'ep{ep_idx}_j235_analysis.png', dpi=150)
        plt.close(fig)

        # All 7 joints plot
        fig, axes = plt.subplots(7, 1, figsize=(14, 14), sharex=True)
        for ax_i, jname in enumerate(JOINT_NAMES):
            ax = axes[ax_i]
            jidx = JOINT_NAMES.index(jname)
            frames = np.arange(n_frames)
            ax.plot(frames, gt[:, jidx], 'b-', linewidth=1.2, alpha=0.8, label='GT')
            ax.plot(frames, pred[:, jidx], 'r--', linewidth=1.2, alpha=0.8, label='Pred')
            ax.set_ylabel(jname)
            ax.legend(loc='upper right', fontsize=6)
            ax.grid(True, alpha=0.3)
            for pi, (pname, ps, pe) in enumerate(phases):
                ax.axvspan(ps, pe, alpha=0.1, color=colors[pi % len(colors)])
        axes[-1].set_xlabel('Frame')
        fig.suptitle(f'Episode {ep_idx} — All Joints GT vs Pred', fontsize=13, fontweight='bold')
        plt.tight_layout()
        fig.savefig(output_dir / f'ep{ep_idx}_all_joints.png', dpi=150)
        plt.close(fig)

    # ---- Cross-episode summary ----
    lines.append(f"\n{'='*60}")
    lines.append("CROSS-EPISODE SUMMARY for J2, J3, J5")
    lines.append(f"{'='*60}")

    for jname in TARGET_JOINTS:
        lines.append(f"\n--- {jname} across all episodes ---")
        lines.append(f"{'Ep':<6} {'NetGT':>10} {'NetPred':>10} {'SameSign':>8} "
                     f"{'DerivAgree':>10} {'AbsNetGT':>10} {'Issue':>20}")
        lines.append("-" * 74)

        all_same = True
        all_deriv_good = True
        all_small_net = True

        for ep_idx in range(n_eps):
            dm = results[f"ep{ep_idx}"]["overall"][jname]
            issue = ""
            if not dm["net_disp_same_sign"]:
                issue = "WRONG_SIGN"
                all_same = False
            if dm["deriv_sign_agree"] < 0.5:
                issue += " LOW_DERIV"
                all_deriv_good = False
            if dm["abs_net_disp_gt"] > 0.05:
                all_small_net = False
                if not dm["net_disp_same_sign"]:
                    issue += " SIG_DISP_WRONG"

            lines.append(
                f"Ep{ep_idx:<3} {dm['net_disp_gt']:+10.4f} {dm['net_disp_pred']:+10.4f} "
                f"{'OK' if dm['net_disp_same_sign'] else 'WRONG':>8} "
                f"{dm['deriv_sign_agree']:10.3f} {dm['abs_net_disp_gt']:10.4f} "
                f"{issue:>20}"
            )

        lines.append(f"\n  Summary for {jname}:")
        lines.append(f"    All same sign: {all_same}")
        lines.append(f"    All deriv agree > 0.5: {all_deriv_good}")
        lines.append(f"    All net displacement < 0.05 rad: {all_small_net}")
        if all_small_net:
            lines.append(f"    VERDICT: Net displacement too small (< 0.05 rad) — direction metric UNRELIABLE")
        if all_deriv_good:
            lines.append(f"    VERDICT: Derivative sign agreement good — local motion direction CORRECT")
        if all_same and not all_small_net:
            lines.append(f"    VERDICT: Start-to-end direction matches — trajectory endpoint CORRECT")

    # ---- Final judgment ----
    lines.append(f"\n{'='*60}")
    lines.append("FINAL JUDGMENT")
    lines.append(f"{'='*60}")

    j2_data = [results[f"ep{ep_idx}"]["overall"]["J2"] for ep_idx in range(n_eps)]
    j3_data = [results[f"ep{ep_idx}"]["overall"]["J3"] for ep_idx in range(n_eps)]
    j5_data = [results[f"ep{ep_idx}"]["overall"]["J5"] for ep_idx in range(n_eps)]

    j2_small = all(d["abs_net_disp_gt"] < 0.05 for d in j2_data)
    j3_small = all(d["abs_net_disp_gt"] < 0.05 for d in j3_data)
    j5_small = all(d["abs_net_disp_gt"] < 0.05 for d in j5_data)

    j2_deriv = [d["deriv_sign_agree"] for d in j2_data]
    j3_deriv = [d["deriv_sign_agree"] for d in j3_data]
    j5_deriv = [d["deriv_sign_agree"] for d in j5_data]

    j2_signs = [d["net_disp_same_sign"] for d in j2_data]
    j3_signs = [d["net_disp_same_sign"] for d in j3_data]
    j5_signs = [d["net_disp_same_sign"] for d in j5_data]

    lines.append(f"\n  J2: small_net={j2_small}, signs={j2_signs}, deriv={[f'{v:.3f}' for v in j2_deriv]}")
    lines.append(f"  J3: small_net={j3_small}, signs={j3_signs}, deriv={[f'{v:.3f}' for v in j3_deriv]}")
    lines.append(f"  J5: small_net={j5_small}, signs={j5_signs}, deriv={[f'{v:.3f}' for v in j5_deriv]}")

    lines.append(f"\n  Phase-level check (close_hold phase = core manipulation):")
    for ep_idx in range(n_eps):
        phases = results[f"ep{ep_idx}"]["phases"]
        for pname, pdata in phases.items():
            if "close" in pname or "hold" in pname:
                lines.append(f"    Ep{ep_idx} {pname}:")
                for jname in TARGET_JOINTS:
                    pd_ = pdata[jname]
                    lines.append(
                        f"      {jname}: MSE={pd_['mse']:.6f}  r={pd_['r']:+.4f}  "
                        f"net_gt={pd_['net_disp_gt']:+.4f}  same_sign={'OK' if pd_['net_disp_same_sign'] else 'WRONG'}"
                    )

    # Overall verdict
    all_deriv_ok = all(np.mean(dlist) > 0.5 for dlist in [j2_deriv, j3_deriv, j5_deriv])
    j2_j3_close_ok = True
    for ep_idx in range(n_eps):
        phases = results[f"ep{ep_idx}"]["phases"]
        for pname, pdata in phases.items():
            if "close" in pname or "hold" in pname:
                for jname in ["J2", "J3"]:
                    if not pdata[jname]["net_disp_same_sign"]:
                        net_gt = pdata[jname]["net_disp_gt"]
                        if abs(net_gt) > 0.02:  # significant motion in wrong direction
                            j2_j3_close_ok = False

    if j2_small and j3_small and j5_small and all_deriv_ok:
        verdict = "A"
        lines.append(f"\n  >>> VERDICT A: Direction metrics are FALSE POSITIVES caused by small net displacement.")
        lines.append(f"      J2/J3/J5 net displacements are all < 0.05 rad (≈ 3°), too small for")
        lines.append(f"      mean-based direction comparison to be reliable.")
        lines.append(f"      Derivative sign agreement confirms local motion direction is correct.")
        lines.append(f"      5-episode offline fitting is PASSED.")
    elif not j2_j3_close_ok:
        verdict = "B"
        lines.append(f"\n  >>> VERDICT B: J2/J3 direction is WRONG in close_hold phase (core manipulation).")
        lines.append(f"      This is NOT a false positive — the model genuinely predicts wrong motion")
        lines.append(f"      during lift/lower. Check action/state time alignment, delta_timestamps,")
        lines.append(f"      normalization, and training config.")
    elif not all_deriv_ok:
        verdict = "B"
        lines.append(f"\n  >>> VERDICT B: Derivative sign agreement is poor — check training config.")
    else:
        verdict = "C"
        lines.append(f"\n  >>> VERDICT C: Mixed results — some joints OK, others marginal.")
        lines.append(f"      Check specific joints/phases above for details.")

    lines.append(f"\n  Plots saved to: {output_dir}")

    return "\n".join(lines), results, verdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path,
                        default="outputs/train/piper_bottle_pick_place_aside_5ep/checkpoints/last/pretrained_model")
    parser.add_argument("--dataset-root", type=Path, default="data/lerobot_dataset")
    parser.add_argument("--repo-id", default="piper/bottle_pick_place_aside")
    parser.add_argument("--output-dir", type=Path, default="outputs/eval/5ep_gt_pred_plots")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-inference", action="store_true",
                        help="Skip model inference, load cached predictions")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    dataset = load_dataset(args.dataset_root, args.repo_id)
    print(f"  Episodes: {dataset.num_episodes}, Frames: {dataset.num_frames}, FPS: {dataset.fps}")

    cache_dir = output_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    gt_cache = cache_dir / "all_gt.npy"
    pred_cache = cache_dir / "all_pred.npy"
    grip_cache = cache_dir / "all_grip.npy"

    if args.skip_inference and gt_cache.exists():
        print("Loading cached predictions...")
        all_gt = list(np.load(gt_cache, allow_pickle=True))
        all_pred = list(np.load(pred_cache, allow_pickle=True))
        all_grip = list(np.load(grip_cache, allow_pickle=True))
    else:
        print("Loading policy...")
        policy, config = load_policy(args.checkpoint, args.device)
        print(f"  Config: n_obs_steps={config.n_obs_steps}, horizon={config.horizon}, "
              f"n_action_steps={config.n_action_steps}")

        print("Running inference on all frames...")
        all_gt, all_pred, all_grip = run_inference(
            policy, config, dataset, args.device
        )

        np.save(gt_cache, np.array(all_gt, dtype=object), allow_pickle=True)
        np.save(pred_cache, np.array(all_pred, dtype=object), allow_pickle=True)
        np.save(grip_cache, np.array(all_grip, dtype=object), allow_pickle=True)

    print(f"Loaded predictions for {len(all_gt)} episodes")

    print("\nRunning analysis...")
    report, results, verdict = analyze_and_plot(all_gt, all_pred, all_grip, output_dir, fps=int(dataset.fps))

    print(report)

    report_path = output_dir / "deep_analysis_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to {report_path}")
    print(f"Final Verdict: {verdict}")


if __name__ == "__main__":
    main()
