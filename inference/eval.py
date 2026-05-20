#!/usr/bin/env python3
"""
Offline evaluation of trained Diffusion Policy on a LeRobot dataset.

Unlike deploy.py (which uses the stateful select_action API for online rollouts),
this script calls generate_actions directly with pre-built observation windows
from the dataset, so every frame's prediction is independent and reproducible.

Usage:
  conda activate piper_act
  python3 inference/eval.py \
    --checkpt outputs/train/piper_bottle_pick_place_aside/checkpoints/last/pretrained_model \
    --dataset-root data/lerobot_dataset
"""

import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import safetensors.torch
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

JOINT_NAMES = ["J1", "J2", "J3", "J4", "J5", "J6", "Grip"]


def parse_episodes(value: str):
    """Parse --episodes: 'all' -> None (all episodes), int -> number."""
    if value.lower() == "all":
        return None
    try:
        n = int(value)
        if n < 0:
            raise argparse.ArgumentTypeError("--episodes must be >= 0 or 'all'")
        return n
    except ValueError:
        raise argparse.ArgumentTypeError("--episodes must be an integer or 'all'")


def resolve_device(device_arg: str, allow_cpu: bool) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if allow_cpu:
            return torch.device("cpu")
        raise RuntimeError(
            "CUDA is not available, and CPU evaluation is disabled to avoid freezing. "
            "Fix the NVIDIA driver/CUDA runtime, or pass --allow-cpu --device cpu for a slow debug run."
        )
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device_arg}, but torch.cuda.is_available() is false.")
    if device.type == "cpu" and not allow_cpu:
        raise RuntimeError(
            "CPU evaluation is disabled by default. Pass --allow-cpu --device cpu for debugging only."
        )
    return device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpt", type=str, required=True)
    parser.add_argument("--dataset-root", type=str, default="data/lerobot_dataset")
    parser.add_argument("--dataset-repo-id", type=str, default="piper/bottle_pick_place_aside")
    parser.add_argument("--episodes", type=parse_episodes, default=None,
                        help="Number of episodes to evaluate ('all' or integer, default: all)")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for deterministic diffusion noise")
    parser.add_argument("--num-inference-steps", type=int, default=None,
                        help="Override diffusion sampling steps")
    parser.add_argument("--eval-batch-size", type=int, default=8,
                        help="Micro-batch size for GPU efficiency (1 = framewise)")
    args = parser.parse_args()

    if args.num_inference_steps is not None and args.num_inference_steps <= 0:
        parser.error("--num-inference-steps must be > 0.")

    try:
        device = resolve_device(args.device, args.allow_cpu)
    except RuntimeError as exc:
        parser.exit(2, f"error: {exc}\n")
    print(f"Device: {device}")

    # --- Load policy ---
    print(f"Loading Diffusion Policy from {args.checkpt} ...")
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy, OBS_IMAGES, OBS_STATE
    from lerobot.configs.policies import PreTrainedConfig

    cfg = PreTrainedConfig.from_pretrained(str(args.checkpt))
    policy = DiffusionPolicy(cfg)
    device_str = str(device)
    sd = safetensors.torch.load_file(str(Path(args.checkpt) / "model.safetensors"), device=device_str)
    policy.load_state_dict(sd)
    policy.to(device)
    policy.eval()

    if args.num_inference_steps is not None:
        policy.diffusion.num_inference_steps = args.num_inference_steps
        policy.config.num_inference_steps = args.num_inference_steps
    print(f"  n_obs={cfg.n_obs_steps}  horizon={cfg.horizon}  "
          f"inference_steps={policy.diffusion.num_inference_steps}")

    # --- Load normalizer stats for unnormalization ---
    norm_file = Path(args.checkpt) / "policy_preprocessor_step_3_normalizer_processor.safetensors"
    norm_stats = safetensors.torch.load_file(str(norm_file))
    action_min = norm_stats["action.min"].cpu().numpy().astype(np.float32)
    action_max = norm_stats["action.max"].cpu().numpy().astype(np.float32)

    def unnormalize(pred_norm):
        return (pred_norm + 1.0) * (action_max - action_min) / 2.0 + action_min

    # --- Load dataset ---
    print("Loading dataset ...")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dataset = LeRobotDataset(args.dataset_repo_id, root=args.dataset_root)
    print(f"  {dataset.num_episodes} episodes, {len(dataset)} frames")

    total_eps = dataset.num_episodes
    n_eval = args.episodes if args.episodes is not None else total_eps
    episode_indices = list(range(n_eval))
    print(f"Evaluating episodes: {episode_indices}")

    n_obs = cfg.n_obs_steps
    img_keys = list(cfg.image_features.keys())
    img_key = img_keys[0]
    horizon = cfg.horizon
    micro_batch = args.eval_batch_size

    rng = torch.Generator(device=device)
    rng.manual_seed(args.seed)

    all_gt_list, all_pred_list = [], []
    ep_results = []

    t_start = time.perf_counter()

    for ep_idx in tqdm(episode_indices, desc="Evaluating"):
        ep_mask = (np.array(dataset.hf_dataset["episode_index"]) == ep_idx)
        ep_frames = np.where(ep_mask)[0]
        f0, f1 = ep_frames[0], ep_frames[-1] + 1
        n = f1 - f0

        # Pre-load episode data
        ep_states = np.zeros((n, 7), dtype=np.float32)
        ep_actions = np.zeros((n, 7), dtype=np.float32)
        ep_imgs = np.zeros((n, 3, 480, 640), dtype=np.float32)
        for i in range(n):
            item = dataset[f0 + i]
            ep_states[i] = item["observation.state"].numpy()
            ep_actions[i] = item["action"].numpy()
            img = item[img_key]
            if isinstance(img, torch.Tensor):
                img = img.numpy()
            if img.ndim == 3 and img.shape[0] in (1, 3):
                ep_imgs[i] = img.astype(np.float32) / 255.0
            elif img.ndim == 3 and img.shape[-1] == 3:
                ep_imgs[i] = np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.0

        # Build observation windows and pre-generate noise per frame
        all_obs_s = np.zeros((n, n_obs, 7), dtype=np.float32)
        all_obs_i = np.zeros((n, n_obs, 3, 480, 640), dtype=np.float32)
        all_noise = torch.zeros(n, 1, horizon, 7, device=device, dtype=torch.float32)
        for t in range(n):
            for j in range(n_obs):
                src = max(0, t - (n_obs - 1 - j))
                all_obs_s[t, j] = ep_states[src]
                all_obs_i[t, j] = ep_imgs[src]
            all_noise[t] = torch.randn(1, horizon, 7, generator=rng, device=device,
                                       dtype=torch.float32)

        # Micro-batched inference
        pred = np.zeros((n, 7), dtype=np.float32)
        for start in range(0, n, micro_batch):
            end = min(start + micro_batch, n)
            batch = {
                OBS_STATE: torch.from_numpy(all_obs_s[start:end]).to(device),
                OBS_IMAGES: torch.from_numpy(all_obs_i[start:end]).unsqueeze(2).to(device),
            }
            noise = all_noise[start:end].squeeze(1)
            with torch.no_grad():
                acts = policy.diffusion.generate_actions(batch, noise=noise)
            pred[start:end] = unnormalize(acts[:, 0].cpu().numpy())

        gt = ep_actions
        mse = float(np.mean((gt - pred) ** 2))
        all_gt_list.append(gt)
        all_pred_list.append(pred)
        ep_results.append({"ep": ep_idx, "frames": n, "mse": mse})

    t_total = time.perf_counter() - t_start

    # --- Aggregate ---
    all_gt = np.concatenate(all_gt_list, axis=0)
    all_pred = np.concatenate(all_pred_list, axis=0)
    # Aliases for per-episode range_ratio computation
    ep_gt_list = all_gt_list
    ep_pred_list = all_pred_list
    overall_mse = float(np.mean((all_gt - all_pred) ** 2))

    print(f"\n{'=' * 60}")
    print(f"  EVALUATION RESULTS ({len(episode_indices)} episodes, {len(all_gt)} frames)")
    print(f"  Total time: {t_total:.1f}s")
    print(f"{'=' * 60}")
    print(f"\n  Overall MSE: {overall_mse:.6f}")

    print(f"\n  {'Joint':<6} {'MSE':>10} {'Pearson r':>10} {'GT_range':>10} {'Pred_range':>10} {'RangeRatio':>10}")
    print(f"  {'-' * 58}")
    for j in range(7):
        gt_j, pred_j = all_gt[:, j], all_pred[:, j]
        mse_j = float(np.mean((gt_j - pred_j) ** 2))
        r_j = float(np.corrcoef(gt_j, pred_j)[0, 1])
        gt_range = float(gt_j.max() - gt_j.min())
        pred_range = float(pred_j.max() - pred_j.min())
        range_ratio = pred_range / gt_range if gt_range > 1e-6 else float("nan")
        print(f"  {JOINT_NAMES[j]:<6} {mse_j:10.6f} {r_j:+10.4f} {gt_range:10.4f} "
              f"{pred_range:10.4f} {range_ratio:10.3f}")

    print(f"\n  {'Episode':<10} {'Frames':>7} {'MSE':>10}")
    print(f"  {'-' * 29}")
    for r in ep_results:
        print(f"  Ep{r['ep']:<7} {r['frames']:>7} {r['mse']:10.6f}")

    print(f"\n  J2 avg range_ratio (per-ep): {np.mean([float((p[:, 1].max() - p[:, 1].min()) / (g[:, 1].max() - g[:, 1].min())) for g, p in zip(ep_gt_list, ep_pred_list) if g[:, 1].max() - g[:, 1].min() > 1e-6]):.4f}")
    print(f"  J3 avg range_ratio (per-ep): {np.mean([float((p[:, 2].max() - p[:, 2].min()) / (g[:, 2].max() - g[:, 2].min())) for g, p in zip(ep_gt_list, ep_pred_list) if g[:, 2].max() - g[:, 2].min() > 1e-6]):.4f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
