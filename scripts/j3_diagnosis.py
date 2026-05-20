#!/usr/bin/env python3
"""J3 amplitude damping diagnosis: phase-by-phase, J2/J3 coupling, data distribution.

Usage:
  python scripts/j3_diagnosis.py \
    --checkpoint outputs/train/piper_bottle_pick_place_aside_env2_30clean/checkpoints/last/pretrained_model \
    --dataset-root data/lerobot_dataset_env2_30clean \
    --repo-id piper/bottle_pick_place_aside_env2_30clean \
    --outdir outputs/eval/clean30_j3_diagnosis
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
PHASES = ["approach", "grasp", "lift", "move", "lower", "release", "retreat"]


def segment_by_gripper_and_j2(grip_gt, j2_gt, grip_thresh=0.07):
    """Segment episode into phases using gripper + J2 heuristics."""
    n = len(grip_gt)
    open_mask = grip_gt > grip_thresh

    grasp_idx = None
    for i in range(1, n):
        if open_mask[i - 1] and not open_mask[i]:
            grasp_idx = i
            break
    release_idx = None
    for i in range(n - 1, 0, -1):
        if not open_mask[i - 1] and open_mask[i]:
            release_idx = i
            break
    if grasp_idx is None or release_idx is None:
        return [(0, n, "unknown")]

    j2_smooth = np.convolve(j2_gt, np.ones(5) / 5, mode='same')
    segments = []

    segments.append((0, grasp_idx, "approach"))
    gw = min(10, max(3, (release_idx - grasp_idx) // 8))
    gs, ge = max(0, grasp_idx - gw // 2), min(n, grasp_idx + gw // 2)
    segments.append((gs, ge, "grasp"))

    lift_end = ge
    for i in range(ge, release_idx):
        if j2_smooth[i] > j2_smooth[max(0, i - 10)]:
            lift_end = i
        else:
            break
    lift_end = min(lift_end + 5, release_idx)
    segments.append((ge, lift_end, "lift"))

    move_start, move_end = lift_end, release_idx
    j2_peak = np.argmax(j2_smooth[move_start:release_idx])
    if j2_peak > 0:
        j2_peak += move_start
        for i in range(j2_peak, release_idx):
            if j2_gt[i] < 0.8 * j2_gt[j2_peak]:
                move_end = i
                break
    if move_end <= move_start:
        move_end = move_start + max(1, (release_idx - move_start) // 2)
    segments.append((move_start, move_end, "move"))
    segments.append((move_end, release_idx, "lower"))

    rw = min(10, max(3, (n - release_idx) // 8))
    rs, re = max(move_end, release_idx - rw // 2), min(n, release_idx + rw // 2)
    segments.append((rs, re, "release"))
    if re < n:
        segments.append((re, n, "retreat"))

    result = []
    for s, e, name in segments:
        s, e = int(s), int(e)
        if e > s:
            if result and s < result[-1][1]:
                s = result[-1][1]
            if e > s:
                result.append((s, e, name))
    return result


def derivative_sign_agreement(gt, pred):
    """Fraction of frames where GT and Pred derivative signs match."""
    d_gt = np.diff(gt)
    d_pred = np.diff(pred)
    if len(d_gt) < 2:
        return 1.0
    match = (d_gt >= 0) == (d_pred >= 0)
    return float(np.mean(match))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--repo-id", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-batch-size", type=int, default=8)
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy, OBS_IMAGES, OBS_STATE
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    print("Loading dataset...", flush=True)
    ds = LeRobotDataset(repo_id=args.repo_id, root=args.dataset_root, video_backend="pyav")
    print(f"  Episodes: {ds.num_episodes}  Frames: {ds.num_frames}", flush=True)

    print("Loading policy...", flush=True)
    cfg = PreTrainedConfig.from_pretrained(str(args.checkpoint))
    policy = DiffusionPolicy(cfg)
    sd = safetensors.torch.load_file(str(args.checkpoint / "model.safetensors"), device=str(args.device))
    policy.load_state_dict(sd)
    policy.to(args.device)
    policy.eval()

    norm_file = args.checkpoint / "policy_preprocessor_step_3_normalizer_processor.safetensors"
    norm_stats = safetensors.torch.load_file(str(norm_file))
    action_min = norm_stats["action.min"].cpu().numpy().astype(np.float32)
    action_max = norm_stats["action.max"].cpu().numpy().astype(np.float32)

    def unnormalize(pred_norm):
        return (pred_norm + 1.0) * (action_max - action_min) / 2.0 + action_min

    meta_files = sorted((args.dataset_root / "meta" / "episodes").glob("*/*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in meta_files]).sort_values("episode_index")

    n_obs = cfg.n_obs_steps
    img_keys = list(cfg.image_features.keys())
    img_key = img_keys[0]
    horizon = cfg.horizon
    micro_batch = args.eval_batch_size

    rng = torch.Generator(device=args.device)
    rng.manual_seed(args.seed)

    # ================================================================
    #  Inference on all episodes
    # ================================================================
    ep_data = {}

    for _, row in df.iterrows():
        ep = int(row["episode_index"])
        f0 = int(row["dataset_from_index"])
        f1 = int(row["dataset_to_index"])
        n = f1 - f0

        print(f"\nEp{ep} ({n}f): ", end="", flush=True)

        states = np.zeros((n, 7), dtype=np.float32)
        actions = np.zeros((n, 7), dtype=np.float32)
        imgs = np.zeros((n, 3, 480, 640), dtype=np.float32)
        for i in range(n):
            item = ds[f0 + i]
            states[i] = item["observation.state"].numpy()
            actions[i] = item["action"].numpy()
            img = item[img_key]
            if isinstance(img, torch.Tensor):
                img = img.numpy()
            if img.ndim == 3 and img.shape[0] in (1, 3):
                imgs[i] = img.astype(np.float32) / 255.0
            elif img.ndim == 3 and img.shape[-1] == 3:
                imgs[i] = np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.0

        # Build obs windows + noise
        all_obs_s = np.zeros((n, n_obs, 7), dtype=np.float32)
        all_obs_i = np.zeros((n, n_obs, 3, 480, 640), dtype=np.float32)
        all_noise = torch.zeros(n, 1, horizon, 7, device=args.device, dtype=torch.float32)
        for t in range(n):
            for j in range(n_obs):
                src = max(0, t - (n_obs - 1 - j))
                all_obs_s[t, j] = states[src]
                all_obs_i[t, j] = imgs[src]
            all_noise[t] = torch.randn(1, horizon, 7, generator=rng, device=args.device,
                                       dtype=torch.float32)

        # Micro-batched inference
        pred = np.zeros((n, 7), dtype=np.float32)
        for start in range(0, n, micro_batch):
            end = min(start + micro_batch, n)
            batch = {
                OBS_STATE: torch.from_numpy(all_obs_s[start:end]).to(args.device),
                OBS_IMAGES: torch.from_numpy(all_obs_i[start:end]).unsqueeze(2).to(args.device),
            }
            noise = all_noise[start:end].squeeze(1)
            with torch.no_grad():
                acts = policy.diffusion.generate_actions(batch, noise=noise)
            pred[start:end] = unnormalize(acts[:, 0].cpu().numpy())

        gt = actions
        segments = segment_by_gripper_and_j2(gt[:, 6], gt[:, 1])
        ep_data[ep] = {"gt": gt, "pred": pred, "segments": segments, "n": n}
        print("done", flush=True)

    # ================================================================
    #  1. Phase-by-phase J3 statistics
    # ================================================================
    print("\n" + "=" * 80)
    print("1. PHASE-BY-PHASE J3 STATISTICS")
    print("=" * 80)

    phase_stats = {ph: {"gt_range": [], "pred_range": [], "range_ratio": [], "mse": [],
                        "pearson_r": [], "mean_bias": [], "deriv_sign_agree": []}
                   for ph in PHASES}

    for ep in sorted(ep_data.keys()):
        d = ep_data[ep]
        for s, e, ph in d["segments"]:
            if e - s < 2:
                continue
            gt_j3 = d["gt"][s:e, 2]
            pred_j3 = d["pred"][s:e, 2]
            gt_rng = float(gt_j3.max() - gt_j3.min())
            pr_rng = float(pred_j3.max() - pred_j3.min())
            phase_stats[ph]["gt_range"].append(gt_rng)
            phase_stats[ph]["pred_range"].append(pr_rng)
            phase_stats[ph]["range_ratio"].append(pr_rng / gt_rng if gt_rng > 1e-6 else float("nan"))
            phase_stats[ph]["mse"].append(float(np.mean((gt_j3 - pred_j3) ** 2)))
            phase_stats[ph]["pearson_r"].append(float(np.corrcoef(gt_j3, pred_j3)[0, 1]) if len(gt_j3) > 2 else float("nan"))
            phase_stats[ph]["mean_bias"].append(float(pred_j3.mean() - gt_j3.mean()))
            phase_stats[ph]["deriv_sign_agree"].append(derivative_sign_agreement(gt_j3, pred_j3))

    print(f"\n{'Phase':<10} {'GT_range':>10} {'Pred_range':>10} {'RangeRatio':>10} "
          f"{'MSE':>10} {'PearsonR':>10} {'MeanBias':>10} {'DirAgree':>10}")
    print("-" * 82)
    for ph in PHASES:
        ps = phase_stats[ph]
        if not ps["mse"]:
            continue
        print(f"{ph:<10} {np.mean(ps['gt_range']):10.4f} {np.mean(ps['pred_range']):10.4f} "
              f"{np.nanmean(ps['range_ratio']):10.3f} {np.mean(ps['mse']):10.6f} "
              f"{np.nanmean(ps['pearson_r']):10.4f} {np.mean(ps['mean_bias']):10.4f} "
              f"{np.mean(ps['deriv_sign_agree']):10.3f}")

    # ================================================================
    #  2. J2/J3 coupling analysis
    # ================================================================
    print("\n" + "=" * 80)
    print("2. J2/J3 COUPLING ANALYSIS")
    print("=" * 80)

    print(f"\n{'Ep':<6} {'J2_GT_rng':>10} {'J2_Pred_rng':>10} {'J2_RR':>8} "
          f"{'J3_GT_rng':>10} {'J3_Pred_rng':>10} {'J3_RR':>8} {'J2_MSE':>10} {'J3_MSE':>10}")
    print("-" * 84)

    j2_rr_list, j3_rr_list, j2_mse_list, j3_mse_list, j3_gt_range_list = [], [], [], [], []
    ep_labels = sorted(ep_data.keys())

    for ep in ep_labels:
        d = ep_data[ep]
        for tag, idx, rr_list, mse_list, gt_rng_list in [
            ("J2", 1, j2_rr_list, j2_mse_list, None),
            ("J3", 2, j3_rr_list, j3_mse_list, j3_gt_range_list),
        ]:
            gt_j = d["gt"][:, idx]
            pred_j = d["pred"][:, idx]
            gt_rng = float(gt_j.max() - gt_j.min())
            pr_rng = float(pred_j.max() - pred_j.min())
            rr_list.append(pr_rng / gt_rng if gt_rng > 1e-6 else float("nan"))
            mse_list.append(float(np.mean((gt_j - pred_j) ** 2)))
            if gt_rng_list is not None:
                gt_rng_list.append(gt_rng)

        print(f"Ep{ep:<4} {float(d['gt'][:, 1].max() - d['gt'][:, 1].min()):10.4f} "
              f"{float(d['pred'][:, 1].max() - d['pred'][:, 1].min()):10.4f} "
              f"{j2_rr_list[-1]:8.3f} "
              f"{float(d['gt'][:, 2].max() - d['gt'][:, 2].min()):10.4f} "
              f"{float(d['pred'][:, 2].max() - d['pred'][:, 2].min()):10.4f} "
              f"{j3_rr_list[-1]:8.3f} {j2_mse_list[-1]:10.6f} {j3_mse_list[-1]:10.6f}")

    j2_rr = np.array(j2_rr_list)
    j3_rr = np.array(j3_rr_list)
    j3_mse_arr = np.array(j3_mse_list)
    j3_gt_rng_arr = np.array(j3_gt_range_list)

    # Correlation between J2 and J3 range_ratios
    j2j3_rr_corr = float(np.corrcoef(j2_rr, j3_rr)[0, 1])
    print(f"\nJ2/J3 range_ratio correlation: {j2j3_rr_corr:+.4f}")
    if j2j3_rr_corr < -0.3:
        print("  ⚠ Negative correlation detected → J2↑ leads to J3↓ (shoulder-elbow compensation)")
    elif j2j3_rr_corr > 0.3:
        print("  Positive correlation → both improve/deteriorate together")
    else:
        print("  No strong coupling → J2 and J3 errors are independent")

    # Correlation between J2 MSE and J3 MSE
    mse_corr = float(np.corrcoef(np.array(j2_mse_list), j3_mse_arr)[0, 1])
    print(f"J2/J3 MSE correlation: {mse_corr:+.4f}")

    print(f"\nJ2 range_ratio: mean={j2_rr.mean():.4f}  std={j2_rr.std():.4f}")
    print(f"J3 range_ratio: mean={j3_rr.mean():.4f}  std={j3_rr.std():.4f}")

    # ================================================================
    #  3. J3 data distribution analysis
    # ================================================================
    print("\n" + "=" * 80)
    print("3. J3 DATA DISTRIBUTION ANALYSIS")
    print("=" * 80)

    print(f"\nJ3 GT range across episodes: mean={j3_gt_rng_arr.mean():.4f}  "
          f"std={j3_gt_rng_arr.std():.4f}  "
          f"min={j3_gt_rng_arr.min():.4f} (Ep{ep_labels[np.argmin(j3_gt_rng_arr)]})  "
          f"max={j3_gt_rng_arr.max():.4f} (Ep{ep_labels[np.argmax(j3_gt_rng_arr)]})")

    # Correlation: J3 GT range vs J3 MSE → if positive, larger J3 motion = harder to predict
    gt_rng_vs_mse = float(np.corrcoef(j3_gt_rng_arr, j3_mse_arr)[0, 1])
    print(f"J3 GT_range vs J3 MSE correlation: {gt_rng_vs_mse:+.4f}")
    if gt_rng_vs_mse > 0.3:
        print("  Larger J3 motion → higher MSE (diffusion shrinks to mean)")
    elif gt_rng_vs_mse < -0.3:
        print("  Smaller J3 motion → higher MSE (unexpected)")
    else:
        print("  No clear relationship between J3 range and prediction error")

    # Top10 comparison reference
    print("\nReference — Top10 J3 range_ratio avg: 0.625")
    print(f"Reference — Top10 J3 GT range (est): {0.825:.4f}")
    print(f"Clean30  J3 range_ratio avg: {j3_rr.mean():.4f}")
    print(f"Clean30  J3 GT range avg:    {j3_gt_rng_arr.mean():.4f}")

    # Is the Clean30 J3 distribution more spread out?
    j3_pred_rng = np.array([float(ep_data[ep]["pred"][:, 2].max() - ep_data[ep]["pred"][:, 2].min())
                            for ep in ep_labels])
    print(f"\nJ3 Pred range: mean={j3_pred_rng.mean():.4f}  std={j3_pred_rng.std():.4f}")
    print(f"J3 GT → Pred range shrinkage: {j3_pred_rng.mean() / j3_gt_rng_arr.mean():.4f}")

    # ================================================================
    #  4. Generate per-episode curves
    # ================================================================
    print("\n" + "=" * 80)
    print("4. GENERATING PER-EPISODE CURVES")
    print("=" * 80)

    n_cols = 5
    n_rows = (len(ep_labels) + n_cols - 1) // n_cols

    # ---- Individual plots ----
    for ep in ep_labels:
        d = ep_data[ep]
        t = np.arange(d["n"])

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        colors = ['#e8f5e9', '#fff3e0', '#e3f2fd', '#fce4ec', '#f3e5f5', '#e0f2f1', '#fff8e1']
        for ax in axes:
            for (s, e, ph), c in zip(d["segments"], colors):
                ax.axvspan(s, e, alpha=0.12, color=c)

        # Gripper
        ax = axes[0]
        ax.plot(t, d["gt"][:, 6], 'b-', label='GT', linewidth=1.2)
        ax.plot(t, d["pred"][:, 6], 'r--', label='Pred', linewidth=1.2)
        ax.axhline(y=0.07, color='gray', linestyle=':', alpha=0.5)
        ax.set_ylabel("Grip (m)")
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        # Phase labels
        for s, e, ph in d["segments"]:
            ax.text((s + e) / 2, 0.095, ph[:4], ha='center', fontsize=6, color='gray')
        ax.set_ylim(0.0, 0.11)

        # J2
        ax = axes[1]
        ax.plot(t, d["gt"][:, 1], 'b-', label='GT', linewidth=1.2)
        ax.plot(t, d["pred"][:, 1], 'r--', label='Pred', linewidth=1.2)
        j2_rr_ep = j2_rr_list[ep] if ep < len(j2_rr_list) else float("nan")
        ax.set_ylabel("J2 (rad)")
        ax.set_title(f"Ep{ep} — J2 range_ratio={j2_rr_ep:.3f}")
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)

        # J3
        ax = axes[2]
        ax.plot(t, d["gt"][:, 2], 'b-', label='GT', linewidth=1.2)
        ax.plot(t, d["pred"][:, 2], 'r--', label='Pred', linewidth=1.2)
        j3_rr_ep = j3_rr_list[ep] if ep < len(j3_rr_list) else float("nan")
        ax.set_ylabel("J3 (rad)")
        ax.set_xlabel("Frame")
        ax.set_title(f"Ep{ep} — J3 range_ratio={j3_rr_ep:.3f}  GT_range={j3_gt_rng_arr[ep]:.4f}")
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(args.outdir / f"ep{ep}_j23_grip.png", dpi=100)
        plt.close(fig)

    print(f"  Saved {len(ep_labels)} individual plots to {args.outdir}/")

    # ---- J3 summary grid ----
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2.5 * n_rows))
    axes = np.atleast_1d(axes).flatten()
    for idx, ep in enumerate(ep_labels):
        d = ep_data[ep]
        t = np.arange(d["n"])
        ax = axes[idx]
        ax.plot(t, d["gt"][:, 2], 'b-', label='GT', linewidth=0.8, alpha=0.8)
        ax.plot(t, d["pred"][:, 2], 'r--', label='Pred', linewidth=0.8, alpha=0.8)
        ax.set_title(f"Ep{ep} J3 RR={j3_rr_list[ep]:.2f}", fontsize=8)
        ax.set_ylabel("J3", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.2)
    for idx in range(len(ep_labels), len(axes)):
        axes[idx].set_visible(False)
    plt.tight_layout()
    fig.savefig(args.outdir / "all_j3_summary.png", dpi=120)
    plt.close(fig)
    print(f"  Saved J3 summary grid to {args.outdir}/all_j3_summary.png")

    # ---- J2 vs J3 range_ratio scatter ----
    fig, ax = plt.subplots(figsize=(8, 6))
    for ep in ep_labels:
        ax.annotate(str(ep), (j2_rr_list[ep], j3_rr_list[ep]), fontsize=7,
                    ha='center', va='center')
    ax.scatter(j2_rr_list, j3_rr_list, alpha=0.5)
    ax.axhline(y=0.625, color='green', linestyle=':', alpha=0.5, label='Top10 J3 baseline')
    ax.axvline(x=0.653, color='blue', linestyle=':', alpha=0.5, label='Top10 J2 baseline')
    ax.axhline(y=0.70, color='orange', linestyle='--', alpha=0.5, label='J3 target 0.70')
    ax.set_xlabel("J2 range_ratio")
    ax.set_ylabel("J3 range_ratio")
    ax.set_title(f"J2 vs J3 range_ratio per episode (r={j2j3_rr_corr:+.3f})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(args.outdir / "j2j3_range_ratio_scatter.png", dpi=120)
    plt.close(fig)
    print(f"  Saved J2/J3 scatter to {args.outdir}/j2j3_range_ratio_scatter.png")

    # ---- J3 GT range vs J3 MSE scatter ----
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(j3_gt_rng_arr, j3_mse_arr, alpha=0.6)
    for ep in ep_labels:
        ax.annotate(str(ep), (j3_gt_rng_arr[ep], j3_mse_arr[ep]), fontsize=7,
                    ha='center', va='center')
    ax.set_xlabel("J3 GT range (rad)")
    ax.set_ylabel("J3 MSE")
    ax.set_title(f"J3 GT range vs MSE (r={gt_rng_vs_mse:+.3f})")
    ax.grid(True, alpha=0.3)
    fig.savefig(args.outdir / "j3_gtrange_vs_mse.png", dpi=120)
    plt.close(fig)
    print(f"  Saved J3 range vs MSE scatter to {args.outdir}/j3_gtrange_vs_mse.png")

    # ================================================================
    #  Save numerical results
    # ================================================================
    results = {
        "phase_stats": {ph: {k: (np.nanmean(v) if v else float("nan")) for k, v in ps.items()}
                        for ph, ps in phase_stats.items()},
        "j2j3_rr_correlation": float(j2j3_rr_corr),
        "j2j3_mse_correlation": float(mse_corr),
        "j3_gt_range_vs_mse_correlation": float(gt_rng_vs_mse),
        "j2_range_ratio_mean": float(j2_rr.mean()),
        "j2_range_ratio_std": float(j2_rr.std()),
        "j3_range_ratio_mean": float(j3_rr.mean()),
        "j3_range_ratio_std": float(j3_rr.std()),
        "j3_gt_range_mean": float(j3_gt_rng_arr.mean()),
        "j3_gt_range_std": float(j3_gt_rng_arr.std()),
        "per_episode": [
            {"ep": int(ep), "j2_rr": float(j2_rr_list[ep]), "j3_rr": float(j3_rr_list[ep]),
             "j3_gt_range": float(j3_gt_rng_arr[ep]), "j3_mse": float(j3_mse_arr[ep]),
             "j2_mse": float(j2_mse_list[ep])}
            for ep in ep_labels
        ],
    }

    with open(args.outdir / "j3_diagnosis.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.outdir}/j3_diagnosis.json")

    # Print final judgment
    print("\n" + "=" * 80)
    print("DIAGNOSIS SUMMARY")
    print("=" * 80)
    print(f"J3 range_ratio: {j3_rr.mean():.4f} (Top10: 0.625)")
    print(f"J2 range_ratio: {j2_rr.mean():.4f} (Top10: 0.653)")
    print(f"J2/J3 range_ratio coupling: {j2j3_rr_corr:+.3f}")
    print(f"J3 GT range correlation with MSE: {gt_rng_vs_mse:+.3f}")
    print(f"J3 phase with worst range_ratio: "
          f"{max([(ph, np.nanmean(phase_stats[ph]['range_ratio'])) for ph in PHASES if phase_stats[ph]['range_ratio']], key=lambda x: abs(x[1]))}")


if __name__ == "__main__":
    main()
