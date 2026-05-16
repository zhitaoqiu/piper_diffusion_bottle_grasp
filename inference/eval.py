#!/usr/bin/env python3
"""
Evaluate trained Diffusion Policy on dataset episodes.

Computes MSE between predicted and ground-truth actions per episode.

Usage:
  conda activate piper_act
  python3 inference/eval.py \
    --checkpt outputs/train/piper_bottle_grasp/checkpoints/last/pretrained_model \
    --dataset-root data/lerobot_dataset
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_policy_processors(policy, checkpt: str, device: torch.device):
    from lerobot.policies.factory import make_pre_post_processors

    return make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=checkpt,
        preprocessor_overrides={
            "device_processor": {"device": device.type},
            "normalizer_processor": {"device": device.type},
        },
        postprocessor_overrides={
            "unnormalizer_processor": {"device": device.type},
            "device_processor": {"device": "cpu"},
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpt", type=str, required=True)
    parser.add_argument("--dataset-root", type=str, default="data/lerobot_dataset")
    parser.add_argument("--dataset-repo-id", type=str, default="piper/bottle_grasp")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of episodes to evaluate (0 = all)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load policy
    print(f"Loading Diffusion Policy from {args.checkpt} ...")
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    policy = DiffusionPolicy.from_pretrained(args.checkpt)
    policy.to(device)
    policy.eval()

    # Load dataset
    print("Loading dataset ...")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dataset = LeRobotDataset(args.dataset_repo_id, root=args.dataset_root)
    print(f"  {dataset.num_episodes} episodes, {len(dataset)} frames")

    # Load processors
    preprocessor, postprocessor = load_policy_processors(policy, args.checkpt, device)
    preprocessor.reset()
    postprocessor.reset()

    # Pick episodes (use last N as validation)
    total_eps = dataset.num_episodes
    n_eval = max(1, args.episodes) if args.episodes > 0 else total_eps
    n_eval = min(n_eval, total_eps)
    episode_indices = list(range(max(0, total_eps - n_eval), total_eps))
    print(f"Evaluating episodes: {episode_indices}")

    all_mse = []
    all_joint_errors = []

    for ep_idx in tqdm(episode_indices, desc="Evaluating"):
        ep_mask = (np.array(dataset.hf_dataset["episode_index"]) == ep_idx)
        ep_frames = np.where(ep_mask)[0]
        ep_start, ep_end = ep_frames[0], ep_frames[-1] + 1

        predicted = []
        ground_truth = []

        # Reset policy at episode start (clears observation history)
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

        for frame_idx in range(ep_start, ep_end):
            item = dataset[frame_idx]

            batch = {}
            for key in item:
                if key.startswith("observation.state") or key.startswith("observation.images."):
                    batch[key] = item[key].unsqueeze(0).to(device)

            with torch.inference_mode():
                norm_batch = preprocessor(batch)
                action = policy.select_action(norm_batch)
                action = postprocessor(action)

            pred = action.squeeze(0).cpu().numpy()
            gt = item["action"].cpu().numpy()

            predicted.append(pred)
            ground_truth.append(gt)

        predicted = np.array(predicted)
        ground_truth = np.array(ground_truth)

        mse = np.mean((predicted - ground_truth) ** 2)
        joint_mse = np.mean((predicted - ground_truth) ** 2, axis=0)
        all_mse.append(mse)
        all_joint_errors.append(joint_mse)

        print(f"  Ep {ep_idx}: MSE={mse:.6f}  frames={ep_end - ep_start}")

    # Summary
    mean_mse = np.mean(all_mse)
    mean_joint_mse = np.mean(all_joint_errors, axis=0)
    joint_names = ["j1", "j2", "j3", "j4", "j5", "j6", "gripper"]
    print(f"\n{'='*50}")
    print(f"  Mean MSE across {len(episode_indices)} episodes: {mean_mse:.6f}")
    print(f"  Per-joint MSE:")
    for name, err in zip(joint_names, mean_joint_mse):
        print(f"    {name}: {err:.6f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
