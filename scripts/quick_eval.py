#!/usr/bin/env python3
"""Fast offline evaluation — streaming inference, no image preloading."""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import safetensors.torch
import torch

JOINT_NAMES = ["J1", "J2", "J3", "J4", "J5", "J6", "Grip"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path,
                   default="outputs/train/piper_bottle_pick_place_aside_top10/checkpoints/last/pretrained_model")
    p.add_argument("--dataset-root", type=Path, default="data/lerobot_dataset_top10")
    p.add_argument("--repo-id", default="piper/bottle_pick_place_aside_top10")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    # Imports
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy, OBS_IMAGES, OBS_STATE
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    print("Loading dataset...", flush=True)
    ds = LeRobotDataset(repo_id=args.repo_id, root=args.dataset_root, video_backend="pyav")
    print(f"  Episodes: {ds.num_episodes}  Frames: {ds.num_frames}  FPS: {ds.fps}", flush=True)

    print("Loading policy...", flush=True)
    cfg = PreTrainedConfig.from_pretrained(str(args.checkpoint))
    policy = DiffusionPolicy(cfg)
    sd = safetensors.torch.load_file(str(args.checkpoint / "model.safetensors"), device=args.device)
    policy.load_state_dict(sd)
    policy.to(args.device)
    policy.eval()
    print(f"  n_obs={cfg.n_obs_steps}  horizon={cfg.horizon}  n_act={cfg.n_action_steps}", flush=True)

    # Load normalization stats for MIN_MAX unnormalization
    norm_file = args.checkpoint / "policy_preprocessor_step_3_normalizer_processor.safetensors"
    norm_stats = safetensors.torch.load_file(str(norm_file))
    action_min = norm_stats["action.min"].cpu().numpy().astype(np.float32)
    action_max = norm_stats["action.max"].cpu().numpy().astype(np.float32)
    print(f"  Action min: {[f'{v:.4f}' for v in action_min]}", flush=True)
    print(f"  Action max: {[f'{v:.4f}' for v in action_max]}", flush=True)

    def unnormalize(pred_norm):
        """MIN_MAX unnormalization: [-1,1] -> [min,max]"""
        return (pred_norm + 1.0) * (action_max - action_min) / 2.0 + action_min

    # Get episode boundaries
    meta_files = sorted((args.dataset_root / "meta" / "episodes").glob("*/*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in meta_files]).sort_values("episode_index")

    n_obs = cfg.n_obs_steps
    img_keys = list(cfg.image_features.keys())

    # Results storage
    ep_results = []
    all_gt_flat = []
    all_pred_flat = []

    for _, row in df.iterrows():
        ep = int(row["episode_index"])
        f0 = int(row["dataset_from_index"])
        f1 = int(row["dataset_to_index"])
        n = f1 - f0

        print(f"  Ep{ep} ({n}f): ", end="", flush=True)

        gt = np.zeros((n, 7), dtype=np.float32)
        pred = np.zeros((n, 7), dtype=np.float32)
        grip_gt = np.zeros(n, dtype=np.float32)

        # Pre-load states, actions, images for this episode only
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
        print(".", end="", flush=True)

        # Run inference with progress
        for t in range(n):
            gt[t] = ep_actions[t]
            grip_gt[t] = gt[t, 6]

            # Build obs window
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

            if t % 100 == 0:
                print(".", end="", flush=True)

        print(" done", flush=True)

        # Per-episode metrics
        mse = float(np.mean((gt - pred) ** 2))
        r_vals = [float(np.corrcoef(gt[:, j], pred[:, j])[0, 1]) for j in range(7)]

        grip_start = float(grip_gt[0])
        grip_end = float(grip_gt[-1])
        grip_min = float(grip_gt.min())
        grip_max = float(grip_gt.max())
        grip_ok = grip_start > 0.07 and (grip_max - grip_min) > 0.03 and grip_end > 0.07

        ep_results.append({
            "ep": ep, "frames": n, "mse": mse,
            "r": r_vals,
            "grip_start": grip_start, "grip_end": grip_end,
            "grip_min": grip_min, "grip_max": grip_max,
            "grip_ok": grip_ok,
        })
        all_gt_flat.append(gt)
        all_pred_flat.append(pred)

    # Summary
    all_gt = np.concatenate(all_gt_flat, axis=0)
    all_pred = np.concatenate(all_pred_flat, axis=0)

    print("\n" + "=" * 70)
    print("TOP 10 OFFLINE EVALUATION RESULTS")
    print("=" * 70)

    overall_mse = float(np.mean((all_gt - all_pred) ** 2))
    print(f"\nOverall MSE: {overall_mse:.6f}")

    print(f"\n{'Joint':<6} {'MSE':>10} {'Pearson r':>10} {'dir'}")
    print("-" * 34)
    dir_ok = 0
    for j in range(7):
        mse_j = float(np.mean((all_gt[:, j] - all_pred[:, j]) ** 2))
        r = float(np.corrcoef(all_gt[:, j], all_pred[:, j])[0, 1])
        gt_net = all_gt[-1, j] - all_gt[0, j]
        pred_net = all_pred[-1, j] - all_pred[0, j]
        d_str = "OK" if (abs(gt_net) < 0.01 or gt_net * pred_net >= 0) else "WRONG"
        if d_str == "OK":
            dir_ok += 1
        print(f"{JOINT_NAMES[j]:<6} {mse_j:10.6f} {r:+10.4f} {d_str}")

    print(f"\n{'Episode':<10} {'Frames':>7} {'MSE':>10} {'Gripper':>8}")
    print("-" * 38)
    all_grip_ok = True
    for r in ep_results:
        gs = "OK" if r["grip_ok"] else "FAIL"
        if not r["grip_ok"]:
            all_grip_ok = False
        print(f"Ep{r['ep']:<7} {r['frames']:>7} {r['mse']:10.6f} {gs:>8}")
        if not r["grip_ok"]:
            print(f"           start={r['grip_start']:.4f} end={r['grip_end']:.4f} min={r['grip_min']:.4f} max={r['grip_max']:.4f}")

    print(f"\nDirection: {dir_ok}/7 OK")
    print(f"Gripper: {'ALL OK' if all_grip_ok else 'SOME FAIL'}")
    print(f"\nOverall MSE: {overall_mse:.6f}")


if __name__ == "__main__":
    main()
