#!/usr/bin/env python3
"""Deep analysis: J2/J3 amplitude damping, gripper curves, per-phase breakdown.

Usage:
  python scripts/deep_analysis_top10.py \
    --checkpoint outputs/train/piper_bottle_pick_place_aside_top10/checkpoints/last/pretrained_model \
    --dataset-root data/lerobot_dataset_top10 \
    --repo-id piper/bottle_pick_place_aside_top10 \
    --outdir outputs/eval/top10_deep_analysis
"""

import argparse, json, os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import safetensors.torch
import torch

JOINT_NAMES = ["J1", "J2", "J3", "J4", "J5", "J6", "Grip"]

# Phase labels for the standard pick-and-place template
PHASES = ["approach", "grasp", "lift", "move", "lower", "release", "retreat"]


def segment_by_gripper_and_j2(grip_gt, j2_gt, grip_thresh=0.07):
    """Segment episode into phases using gripper + J2 heuristics.

    Returns list of (start_idx, end_idx, phase_name) tuples.
    """
    n = len(grip_gt)
    # Find key transitions
    open_mask = grip_gt > grip_thresh  # gripper open

    # Find grasp point: where gripper first closes (open -> closed)
    grasp_idx = None
    for i in range(1, n):
        if open_mask[i - 1] and not open_mask[i]:
            grasp_idx = i
            break

    # Find release point: where gripper opens again (closed -> open)
    release_idx = None
    for i in range(n - 1, 0, -1):
        if not open_mask[i - 1] and open_mask[i]:
            release_idx = i
            break

    if grasp_idx is None or release_idx is None:
        return [(0, n, "unknown")]

    # J2-based segmentation: J2 going down = lowering, going up = lifting
    j2_smooth = np.convolve(j2_gt, np.ones(5) / 5, mode='same')

    segments = []

    # 1. Approach: start to grasp (gripper open, J2 going down)
    approach_end = grasp_idx
    segments.append((0, approach_end, "approach"))

    # 2. Grasp: around grasp_idx +/- small window (gripper closing)
    grasp_window = min(10, max(3, (release_idx - grasp_idx) // 8))
    grasp_start = max(0, grasp_idx - grasp_window // 2)
    grasp_end = min(n, grasp_idx + grasp_window // 2)
    segments.append((grasp_start, grasp_end, "grasp"))

    # 3. Lift: gripper closed, J2 rising
    lift_end = grasp_end
    for i in range(grasp_end, release_idx):
        if j2_smooth[i] > j2_smooth[max(0, i - 10)]:
            lift_end = i
        else:
            break
    lift_end = min(lift_end + 5, release_idx)
    segments.append((grasp_end, lift_end, "lift"))

    # 4. Move: gripper closed, J2 stable (high plateau)
    move_start = lift_end
    # Find where J2 starts dropping significantly
    move_end = release_idx
    j2_peak = np.argmax(j2_smooth[move_start:release_idx])
    j2_peak += move_start
    # After peak, find where J2 drops to 80% of peak
    for i in range(j2_peak, release_idx):
        if j2_gt[i] < 0.8 * j2_gt[j2_peak]:
            move_end = i
            break
    if move_end <= move_start:
        move_end = move_start + (release_idx - move_start) // 2
    segments.append((move_start, move_end, "move"))

    # 5. Lower: gripper closed, J2 dropping
    lower_end = release_idx
    segments.append((move_end, lower_end, "lower"))

    # 6. Release: around release_idx
    release_window = min(10, max(3, (n - release_idx) // 8))
    rel_start = max(lower_end, release_idx - release_window // 2)
    rel_end = min(n, release_idx + release_window // 2)
    segments.append((rel_start, rel_end, "release"))

    # 7. Retreat: after release to end
    if rel_end < n:
        segments.append((rel_end, n, "retreat"))

    # Validate and fix overlapping segments
    result = []
    for s, e, name in segments:
        s = max(0, int(s))
        e = min(n, int(e))
        if e > s:
            if result and s < result[-1][1]:
                s = result[-1][1]
            if e > s:
                result.append((s, e, name))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path,
                   default="outputs/train/piper_bottle_pick_place_aside_top10/checkpoints/last/pretrained_model")
    p.add_argument("--dataset-root", type=Path, default="data/lerobot_dataset_top10")
    p.add_argument("--repo-id", default="piper/bottle_pick_place_aside_top10")
    p.add_argument("--device", default="cuda")
    p.add_argument("--outdir", type=Path, default="outputs/eval/top10_deep_analysis")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy, OBS_IMAGES, OBS_STATE
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    print("Loading dataset...", flush=True)
    ds = LeRobotDataset(repo_id=args.repo_id, root=args.dataset_root, video_backend="pyav")
    print(f"  Episodes: {ds.num_episodes}  Frames: {ds.num_frames}", flush=True)

    print("Loading policy...", flush=True)
    cfg = PreTrainedConfig.from_pretrained(str(args.checkpoint))
    policy = DiffusionPolicy(cfg)
    sd = safetensors.torch.load_file(str(args.checkpoint / "model.safetensors"), device=args.device)
    policy.load_state_dict(sd)
    policy.to(args.device)
    policy.eval()

    # Load normalization stats
    norm_file = args.checkpoint / "policy_preprocessor_step_3_normalizer_processor.safetensors"
    norm_stats = safetensors.torch.load_file(str(norm_file))
    action_min = norm_stats["action.min"].cpu().numpy().astype(np.float32)
    action_max = norm_stats["action.max"].cpu().numpy().astype(np.float32)

    def unnormalize(pred_norm):
        return (pred_norm + 1.0) * (action_max - action_min) / 2.0 + action_min

    # Get episode boundaries
    meta_files = sorted((args.dataset_root / "meta" / "episodes").glob("*/*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in meta_files]).sort_values("episode_index")

    n_obs = cfg.n_obs_steps
    img_keys = list(cfg.image_features.keys())

    # Per-episode detailed results
    ep_data = {}

    for _, row in df.iterrows():
        ep = int(row["episode_index"])
        f0 = int(row["dataset_from_index"])
        f1 = int(row["dataset_to_index"])
        n = f1 - f0

        print(f"\nEp{ep} ({n}f): loading...", end=" ", flush=True)

        ep_states = np.zeros((n, 7), dtype=np.float32)
        ep_actions = np.zeros((n, 7), dtype=np.float32)
        ep_imgs = np.zeros((n, 3, 480, 640), dtype=np.float32)
        for i in range(n):
            item = ds[f0 + i]
            ep_states[i] = item["observation.state"].numpy()
            ep_actions[i] = item["action"].numpy()
            img = item[img_keys[0]]
            if isinstance(img, torch.Tensor):
                img = img.numpy()
            if img.ndim == 3 and img.shape[0] in (1, 3):
                ep_imgs[i] = img.astype(np.float32) / 255.0
            elif img.ndim == 3 and img.shape[-1] == 3:
                ep_imgs[i] = np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.0

        print("inferring...", end=" ", flush=True)

        gt = np.zeros((n, 7), dtype=np.float32)
        pred = np.zeros((n, 7), dtype=np.float32)

        for t in range(n):
            gt[t] = ep_actions[t]
            obs_s = np.zeros((n_obs, 7), dtype=np.float32)
            obs_i = np.zeros((n_obs, 3, 480, 640), dtype=np.float32)
            for j in range(n_obs):
                src = max(0, t - (n_obs - 1 - j))
                obs_s[j] = ep_states[src]
                obs_i[j] = ep_imgs[src]
            batch = {
                OBS_STATE: torch.from_numpy(obs_s).unsqueeze(0).to(args.device),
                OBS_IMAGES: torch.from_numpy(obs_i).unsqueeze(0).unsqueeze(2).to(args.device),
            }
            with torch.no_grad():
                acts = policy.diffusion.generate_actions(batch)
            pred[t] = unnormalize(acts[0, 0].cpu().numpy())

        print("done", flush=True)

        # Segment by gripper + J2
        grip_gt = gt[:, 6]
        grip_pred = pred[:, 6]
        j2_gt = gt[:, 1]
        j2_pred = pred[:, 1]
        segments = segment_by_gripper_and_j2(grip_gt, j2_gt)

        ep_data[ep] = {
            "gt": gt, "pred": pred,
            "grip_gt": grip_gt, "grip_pred": grip_pred,
            "j2_gt": j2_gt, "j2_pred": j2_pred,
            "segments": segments,
            "n": n,
        }

    # ===================================================================
    # Analysis outputs
    # ===================================================================
    out_lines = []
    out_lines.append("=" * 90)
    out_lines.append("DEEP ANALYSIS: J2/J3 Amplitude Damping & Per-Phase Breakdown")
    out_lines.append("=" * 90)

    # ---- Per-episode J2/J3 metrics ----
    out_lines.append("\n" + "=" * 90)
    out_lines.append("1. PER-EPISODE J2/J3 AMPLITUDE DAMPING METRICS")
    out_lines.append("=" * 90)

    j2_all = {k: [] for k in ["gt_range", "pred_range", "range_ratio", "mean_bias", "max_abs_err",
                                "mse", "pearson_r"]}
    j3_all = {k: [] for k in ["gt_range", "pred_range", "range_ratio", "mean_bias", "max_abs_err",
                                "mse", "pearson_r"]}

    for ep in sorted(ep_data.keys()):
        d = ep_data[ep]
        n = d["n"]

        out_lines.append(f"\n--- Ep{ep} ({n} frames) ---")
        out_lines.append(f"{'Joint':<6} {'GT min':>10} {'GT max':>10} {'GT range':>10} "
                         f"{'Pred min':>10} {'Pred max':>10} {'Pred range':>10} "
                         f"{'Range ratio':>11} {'Mean bias':>10} {'Max |err|':>10} "
                         f"{'MSE':>10} {'Pearson r':>10}")

        for j_name, j_idx in [("J2", 1), ("J3", 2)]:
            gt_j = d["gt"][:, j_idx]
            pred_j = d["pred"][:, j_idx]
            gt_range = float(gt_j.max() - gt_j.min())
            pred_range = float(pred_j.max() - pred_j.min())
            range_ratio = pred_range / gt_range if gt_range > 1e-6 else float("nan")
            mean_bias = float(pred_j.mean() - gt_j.mean())
            max_abs_err = float(np.max(np.abs(pred_j - gt_j)))
            mse_j = float(np.mean((pred_j - gt_j) ** 2))
            r_j = float(np.corrcoef(gt_j, pred_j)[0, 1])

            out_lines.append(
                f"{j_name:<6} {gt_j.min():10.4f} {gt_j.max():10.4f} {gt_range:10.4f} "
                f"{pred_j.min():10.4f} {pred_j.max():10.4f} {pred_range:10.4f} "
                f"{range_ratio:11.3f} {mean_bias:10.4f} {max_abs_err:10.4f} "
                f"{mse_j:10.4f} {r_j:10.4f}"
            )

            store = j2_all if j_name == "J2" else j3_all
            store["gt_range"].append(gt_range)
            store["pred_range"].append(pred_range)
            store["range_ratio"].append(range_ratio)
            store["mean_bias"].append(mean_bias)
            store["max_abs_err"].append(max_abs_err)
            store["mse"].append(mse_j)
            store["pearson_r"].append(r_j)

    # Summary
    out_lines.append(f"\n{'Joint':<6} {'Avg RangeRatio':>15} {'Avg MeanBias':>15} "
                     f"{'Avg Max|err|':>15} {'Avg MSE':>12} {'Avg Pearson r':>15}")
    for j_name, store in [("J2", j2_all), ("J3", j3_all)]:
        out_lines.append(
            f"{j_name:<6} {np.mean(store['range_ratio']):15.3f} {np.mean(store['mean_bias']):15.4f} "
            f"{np.mean(store['max_abs_err']):15.4f} {np.mean(store['mse']):12.4f} "
            f"{np.mean(store['pearson_r']):15.4f}"
        )

    # ---- Per-phase breakdown ----
    out_lines.append("\n\n" + "=" * 90)
    out_lines.append("2. PER-PHASE J2/J3 ERROR BREAKDOWN")
    out_lines.append("=" * 90)

    phase_stats = {ph: {"J2_mse": [], "J3_mse": [], "J2_range_ratio": [], "J3_range_ratio": [],
                         "n_frames": []} for ph in PHASES}

    for ep in sorted(ep_data.keys()):
        d = ep_data[ep]
        out_lines.append(f"\nEp{ep} phases:")
        for s, e, phase_name in d["segments"]:
            if e - s < 2:
                continue
            j2_mse = float(np.mean((d["j2_gt"][s:e] - d["j2_pred"][s:e]) ** 2))
            j3_mse = float(np.mean((d["gt"][s:e, 2] - d["pred"][s:e, 2]) ** 2))
            j2_gt_range = float(d["j2_gt"][s:e].max() - d["j2_gt"][s:e].min())
            j2_pred_range = float(d["j2_pred"][s:e].max() - d["j2_pred"][s:e].min())
            j2_range_ratio = j2_pred_range / j2_gt_range if j2_gt_range > 1e-6 else float("nan")
            j3_gt_range = float(d["gt"][s:e, 2].max() - d["gt"][s:e, 2].min())
            j3_pred_range = float(d["pred"][s:e, 2].max() - d["pred"][s:e, 2].min())
            j3_range_ratio = j3_pred_range / j3_gt_range if j3_gt_range > 1e-6 else float("nan")
            out_lines.append(
                f"  {phase_name:<10} frames={e-s:>4}  "
                f"J2 MSE={j2_mse:.4f}  J2 range_ratio={j2_range_ratio:.3f}  "
                f"J3 MSE={j3_mse:.4f}  J3 range_ratio={j3_range_ratio:.3f}"
            )
            phase_stats[phase_name]["J2_mse"].append(j2_mse)
            phase_stats[phase_name]["J3_mse"].append(j3_mse)
            phase_stats[phase_name]["J2_range_ratio"].append(j2_range_ratio)
            phase_stats[phase_name]["J3_range_ratio"].append(j3_range_ratio)
            phase_stats[phase_name]["n_frames"].append(e - s)

    out_lines.append(f"\n{'Phase':<10} {'Avg J2 MSE':>12} {'Avg J2 RangeRatio':>18} "
                     f"{'Avg J3 MSE':>12} {'Avg J3 RangeRatio':>18} {'Total frames':>13}")
    for ph in PHASES:
        if phase_stats[ph]["n_frames"]:
            avg_j2_mse = np.mean(phase_stats[ph]["J2_mse"])
            avg_j3_mse = np.mean(phase_stats[ph]["J3_mse"])
            avg_j2_rr = np.nanmean(phase_stats[ph]["J2_range_ratio"])
            avg_j3_rr = np.nanmean(phase_stats[ph]["J3_range_ratio"])
            total_f = sum(phase_stats[ph]["n_frames"])
            out_lines.append(
                f"{ph:<10} {avg_j2_mse:12.4f} {avg_j2_rr:18.3f} "
                f"{avg_j3_mse:12.4f} {avg_j3_rr:18.3f} {total_f:13}"
            )

    # ---- Gripper curves ----
    out_lines.append("\n\n" + "=" * 90)
    out_lines.append("3. GRIPPER CURVES (saved to output directory)")
    out_lines.append("=" * 90)

    n_eps = len(ep_data)
    n_cols = min(5, n_eps)
    n_rows = (n_eps + n_cols - 1) // n_cols

    # Per-episode individual plots
    for ep in sorted(ep_data.keys()):
        d = ep_data[ep]
        t = np.arange(d["n"])

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Gripper
        ax = axes[0]
        ax.plot(t, d["grip_gt"], 'b-', label='GT', linewidth=1.5, alpha=0.8)
        ax.plot(t, d["grip_pred"], 'r--', label='Pred', linewidth=1.5, alpha=0.8)
        ax.axhline(y=0.07, color='gray', linestyle=':', alpha=0.5, label='threshold')
        # Shade phases
        colors = ['#e8f5e9', '#fff3e0', '#e3f2fd', '#fce4ec', '#f3e5f5', '#e0f2f1', '#fff8e1']
        for (s, e, phase_name), c in zip(d["segments"], colors):
            ax.axvspan(s, e, alpha=0.15, color=c)
            ax.text((s + e) / 2, ax.get_ylim()[0] + 0.002, phase_name[:4],
                    ha='center', fontsize=7, color='gray')
        ax.set_title(f'Ep{ep} Gripper')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Gripper position')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # J2
        ax = axes[1]
        ax.plot(t, d["j2_gt"], 'b-', label='GT', linewidth=1.5, alpha=0.8)
        ax.plot(t, d["j2_pred"], 'r--', label='Pred', linewidth=1.5, alpha=0.8)
        for (s, e, phase_name), c in zip(d["segments"], colors):
            ax.axvspan(s, e, alpha=0.15, color=c)
        ax.set_title(f'Ep{ep} J2')
        ax.set_xlabel('Frame')
        ax.set_ylabel('J2 (rad)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(args.outdir / f"ep{ep}_gripper_j2.png", dpi=100)
        plt.close(fig)
        out_lines.append(f"  Ep{ep}: saved {args.outdir / f'ep{ep}_gripper_j2.png'}")

    # ---- Summary grid of all episodes gripper ----
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = np.atleast_1d(axes).flatten()
    for idx, ep in enumerate(sorted(ep_data.keys())):
        d = ep_data[ep]
        t = np.arange(d["n"])
        ax = axes[idx]
        ax.plot(t, d["grip_gt"], 'b-', label='GT', linewidth=1, alpha=0.7)
        ax.plot(t, d["grip_pred"], 'r--', label='Pred', linewidth=1, alpha=0.7)
        ax.axhline(y=0.07, color='gray', linestyle=':', alpha=0.4)
        ax.set_title(f'Ep{ep} Gripper')
        ax.set_ylabel('Grip')
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.2)
    for idx in range(n_eps, len(axes)):
        axes[idx].set_visible(False)
    plt.tight_layout()
    fig.savefig(args.outdir / "all_episodes_gripper.png", dpi=120)
    plt.close(fig)

    # ---- J2 summary grid ----
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = np.atleast_1d(axes).flatten()
    for idx, ep in enumerate(sorted(ep_data.keys())):
        d = ep_data[ep]
        t = np.arange(d["n"])
        ax = axes[idx]
        ax.plot(t, d["j2_gt"], 'b-', label='GT', linewidth=1, alpha=0.7)
        ax.plot(t, d["j2_pred"], 'r--', label='Pred', linewidth=1, alpha=0.7)
        ax.set_title(f'Ep{ep} J2')
        ax.set_ylabel('J2 (rad)')
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.2)
    for idx in range(n_eps, len(axes)):
        axes[idx].set_visible(False)
    plt.tight_layout()
    fig.savefig(args.outdir / "all_episodes_j2.png", dpi=120)
    plt.close(fig)

    # ---- J2 Pred vs GT scatter (all episodes) ----
    all_j2_gt = np.concatenate([ep_data[ep]["j2_gt"] for ep in sorted(ep_data.keys())])
    all_j2_pred = np.concatenate([ep_data[ep]["j2_pred"] for ep in sorted(ep_data.keys())])
    all_j3_gt = np.concatenate([ep_data[ep]["gt"][:, 2] for ep in sorted(ep_data.keys())])
    all_j3_pred = np.concatenate([ep_data[ep]["pred"][:, 2] for ep in sorted(ep_data.keys())])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, gt, pred, name in [(axes[0], all_j2_gt, all_j2_pred, "J2"),
                                 (axes[1], all_j3_gt, all_j3_pred, "J3")]:
        ax.scatter(gt, pred, s=2, alpha=0.3)
        lim_min = min(gt.min(), pred.min())
        lim_max = max(gt.max(), pred.max())
        ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', linewidth=0.5, label='ideal')
        ax.set_xlabel('GT')
        ax.set_ylabel('Pred')
        ax.set_title(f'{name} Pred vs GT')
        r = np.corrcoef(gt, pred)[0, 1]
        mse = np.mean((gt - pred) ** 2)
        ax.text(0.05, 0.95, f'r={r:.3f}\nMSE={mse:.4f}', transform=ax.transAxes,
                verticalalignment='top', fontsize=9, family='monospace')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(args.outdir / "j2_j3_scatter.png", dpi=120)
    plt.close(fig)

    # ---- Save all predictions as .npz for offline use ----
    out_lines.append(f"\nSaving predictions to {args.outdir / 'all_pred_data.npz'}")
    pred_data = {}
    for ep in sorted(ep_data.keys()):
        d = ep_data[ep]
        pred_data[f"ep{ep}_gt"] = d["gt"]
        pred_data[f"ep{ep}_pred"] = d["pred"]
        pred_data[f"ep{ep}_segments"] = np.array([(s, e) for s, e, _ in d["segments"]], dtype=np.int32)
    np.savez_compressed(args.outdir / "all_pred_data.npz", **pred_data)

    # ---- Print all results ----
    report = "\n".join(out_lines)
    print(report)
    with open(args.outdir / "analysis_report.txt", "w") as f:
        f.write(report)
    print(f"\nReport saved to {args.outdir / 'analysis_report.txt'}")
    print(f"Plots saved to {args.outdir}/")


if __name__ == "__main__":
    main()
