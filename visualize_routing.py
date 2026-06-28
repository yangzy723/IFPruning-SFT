#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IFPruning FFN Routing Visualizer
Compares the activation sparsity patterns of two different prompts.
"""

import sys
import torch
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

TARGET_DIM = 4096

def set_academic_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })

def compute_routing_iou(score1: torch.Tensor, score2: torch.Tensor, k: int) -> np.ndarray:
    """Computes the Intersection over Union (IoU) of top-K channels per layer."""
    num_layers = score1.shape[0]
    ious = np.zeros(num_layers)
    
    for i in range(num_layers):
        _, topk_idx1 = torch.topk(score1[i], k, dim=-1)
        _, topk_idx2 = torch.topk(score2[i], k, dim=-1)
        
        set1 = set(topk_idx1.tolist())
        set2 = set(topk_idx2.tolist())
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        ious[i] = intersection / union if union > 0 else 0
        
    return ious

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file1", type=str, help="Path to first prompt score (.pt)")
    parser.add_argument("file2", type=str, help="Path to second prompt score (.pt)")
    args = parser.parse_args()

    # 1. Load Data
    data1 = torch.load(args.file1, map_location="cpu")
    data2 = torch.load(args.file2, map_location="cpu")
    
    prompt1 = data1["prompt"][:60] + "..." if len(data1["prompt"]) > 60 else data1["prompt"]
    prompt2 = data2["prompt"][:60] + "..." if len(data2["prompt"]) > 60 else data2["prompt"]
    
    # Shape: [num_layers, ffn_dim]
    s1 = data1["scores"]
    s2 = data2["scores"]
    
    if s1.shape != s2.shape:
        raise ValueError("Score matrices shapes do not match!")
        
    num_layers, ffn_dim = s1.shape
    
    # Apply Sigmoid to normalize scores to [0, 1] for better visualization
    s1_norm = torch.sigmoid(s1).numpy()
    s2_norm = torch.sigmoid(s2).numpy()
    diff = np.abs(s1_norm - s2_norm)
    
    # Compute Layer-wise IoU
    ious = compute_routing_iou(s1, s2, TARGET_DIM)
    
    # 2. Plotting
    set_academic_style()
    fig = plt.figure(figsize=(14, 10), dpi=300)
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], width_ratios=[3, 1])
    
    # Heatmap 1
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(s1_norm, aspect='auto', cmap='viridis', interpolation='none')
    ax1.set_title(f"Prompt A Routing Scores\n'{prompt1}'", pad=10)
    ax1.set_ylabel("Layer Depth")
    
    # Heatmap 2
    ax2 = fig.add_subplot(gs[1, 0])
    im2 = ax2.imshow(s2_norm, aspect='auto', cmap='viridis', interpolation='none')
    ax2.set_title(f"Prompt B Routing Scores\n'{prompt2}'", pad=10)
    ax2.set_ylabel("Layer Depth")
    
    # Heatmap Diff
    ax3 = fig.add_subplot(gs[2, 0])
    im3 = ax3.imshow(diff, aspect='auto', cmap='magma', interpolation='none')
    ax3.set_title("Absolute Difference |A - B|", pad=10)
    ax3.set_ylabel("Layer Depth")
    ax3.set_xlabel("FFN Channel Index")
    
    # Add colorbars for heatmaps
    fig.colorbar(im1, ax=ax1, fraction=0.02, pad=0.02)
    fig.colorbar(im2, ax=ax2, fraction=0.02, pad=0.02)
    fig.colorbar(im3, ax=ax3, fraction=0.02, pad=0.02)
    
    # Line Chart: IoU
    ax_iou = fig.add_subplot(gs[:, 1])
    ax_iou.plot(ious, range(num_layers), marker='o', markersize=5, color="#C00000", linewidth=2)
    ax_iou.set_ylim(num_layers - 0.5, -0.5) # Reverse Y axis to match heatmap layers
    ax_iou.set_xlim(0.0, 1.0)
    ax_iou.set_title(f"Routing Similarity (IoU)\nTop-{TARGET_DIM} Channels", pad=10)
    ax_iou.set_xlabel("Intersection over Union")
    ax_iou.grid(True, linestyle="--", alpha=0.6)
    
    # Highlight area of highest difference (lowest IoU)
    min_iou_layer = np.argmin(ious)
    ax_iou.axhline(y=min_iou_layer, color='blue', linestyle=':', alpha=0.5)
    ax_iou.text(0.05, min_iou_layer - 0.5, f"Lowest Similarity\n(Layer {min_iou_layer})", color='blue', fontsize=10)
    
    plt.tight_layout()
    output_filename = "routing_comparison.png"
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"Visualization saved to: {output_filename}")

if __name__ == "__main__":
    main()