#!/usr/bin/env python3
"""Compare the actual hard Top-K masks selected for two prompts."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def hard_topk_mask(scores: torch.Tensor, k: int) -> torch.Tensor:
    if scores.ndim != 2:
        raise ValueError(f"Expected [layers, channels] scores, got {tuple(scores.shape)}")
    if not 0 < k <= scores.shape[-1]:
        raise ValueError(f"target_dim={k} must be in [1, {scores.shape[-1]}]")
    indices = torch.topk(scores.float(), k, dim=-1).indices
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, indices, True)


def layer_overlap(mask_a: torch.Tensor, mask_b: torch.Tensor, k: int) -> np.ndarray:
    """Paper-style overlap: shared selected channels divided by K."""
    return ((mask_a & mask_b).sum(dim=-1).float() / k).cpu().numpy()


def short_prompt(prompt: str) -> str:
    return prompt if len(prompt) <= 60 else prompt[:60] + "..."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file1", help="First routing score .pt file")
    parser.add_argument("file2", help="Second routing score .pt file")
    parser.add_argument("--target-dim", type=int, default=4096)
    parser.add_argument("--output", default="figs/routing_comparison.png")
    args = parser.parse_args()

    data_a = torch.load(args.file1, map_location="cpu", weights_only=True)
    data_b = torch.load(args.file2, map_location="cpu", weights_only=True)
    scores_a = data_a["scores"].float()
    scores_b = data_b["scores"].float()
    if scores_a.shape != scores_b.shape:
        raise ValueError(f"Score shapes differ: {tuple(scores_a.shape)} vs {tuple(scores_b.shape)}")

    mask_a = hard_topk_mask(scores_a, args.target_dim)
    mask_b = hard_topk_mask(scores_b, args.target_dim)
    difference = mask_a.logical_xor(mask_b)
    overlaps = layer_overlap(mask_a, mask_b, args.target_dim)
    num_layers = scores_a.shape[0]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.direction": "in",
            "ytick.direction": "in",
        }
    )
    figure = plt.figure(figsize=(14, 10), dpi=300)
    grid = figure.add_gridspec(3, 2, width_ratios=[3, 1])
    panels = [
        (mask_a, f"Prompt A hard Top-K mask\n'{short_prompt(data_a['prompt'])}'", "Greens"),
        (mask_b, f"Prompt B hard Top-K mask\n'{short_prompt(data_b['prompt'])}'", "Greens"),
        (difference, "Mask XOR difference", "magma"),
    ]
    for row, (values, title, color_map) in enumerate(panels):
        axis = figure.add_subplot(grid[row, 0])
        axis.imshow(
            values.numpy(),
            aspect="auto",
            cmap=color_map,
            interpolation="none",
            vmin=0,
            vmax=1,
        )
        axis.set_title(title)
        axis.set_ylabel("Layer Depth")
        if row == 2:
            axis.set_xlabel("FFN Channel Index")

    overlap_axis = figure.add_subplot(grid[:, 1])
    overlap_axis.plot(overlaps, range(num_layers), marker="o", markersize=4, color="#C00000")
    overlap_axis.set_ylim(num_layers - 0.5, -0.5)
    overlap_axis.set_xlim(0.0, 1.0)
    overlap_axis.set_title(f"Top-{args.target_dim} overlap")
    overlap_axis.set_xlabel("Shared channels / K")
    overlap_axis.grid(True, linestyle="--", alpha=0.6)

    figure.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight")
    print(f"Visualization saved to {args.output}")
    print(
        f"Layer overlap: mean={overlaps.mean():.4f}, "
        f"min={overlaps.min():.4f}, max={overlaps.max():.4f}"
    )
    if np.all(overlaps > 0.999):
        print("WARNING: every layer selects the same mask for both prompts")


if __name__ == "__main__":
    main()
