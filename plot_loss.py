#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IFPruning SFT Training Dynamics Visualizer
------------------------------------------
Zero-configuration script to parse training logs and generate 
professional, publication-ready dual-axis training curves.
"""

import re
from pathlib import Path
import matplotlib.pyplot as plt

# =============================================================================
# 1. Configuration
# =============================================================================
LOG_PATH = Path("./gemma-12b-ifpruning-output/logs/rank_0.log")
OUTPUT_PATH = Path("./loss_curve.png")

# 曲线平滑系数
SMOOTHING_WEIGHT = 0.85

# =============================================================================
# 2. Data Parsing
# =============================================================================
def parse_training_log(log_path: Path):
    """提取包含 Step, Loss, LR, Alpha 的日志行"""
    if not log_path.exists():
        raise FileNotFoundError(f"Critical Error: 日志文件未找到 -> {log_path}")

    pattern = re.compile(r"Step\s+(\d+)\s+\|\s+Loss=([\d.eE+-]+)\s+\|\s+LR=([\d.eE+-]+)\s+\|\s+Alpha=([\d.eE+-]+)")
    steps, losses, lrs, alphas = [], [], [], []

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                steps.append(int(match.group(1)))
                losses.append(float(match.group(2)))
                lrs.append(float(match.group(3)))
                alphas.append(float(match.group(4)))

    if not steps:
        raise ValueError("Critical Error: 未在日志中匹配到任何有效的训练指标记录。")

    return steps, losses, lrs, alphas

def compute_ema(values, weight=0.85):
    """计算指数移动平均 (EMA) 以平滑 Loss 曲线"""
    smoothed = []
    for val in values:
        if not smoothed:
            smoothed.append(val)
        else:
            smoothed.append(smoothed[-1] * weight + val * (1 - weight))
    return smoothed

# =============================================================================
# 3. Clean & Professional Visualization
# =============================================================================
def generate_professional_plot(steps, losses, lrs, alphas):
    """渲染并导出训练动态分析图"""
    
    plt.style.use("default")
    
    fig, (ax1, ax2) = plt.subplots(
        2, 1, 
        figsize=(12, 8), 
        dpi=300, 
        sharex=True, 
        gridspec_kw={'height_ratios': [2, 1]}
    )
    
    # 统一设置网格线
    for ax in (ax1, ax2):
        ax.set_facecolor('white')
        ax.grid(color='#e0e0e0', linestyle='--', linewidth=0.8, alpha=0.7)
        for spine in ax.spines.values():
            spine.set_color('#cccccc')

    # ---------------------------------------------------------
    # Top Subplot: Loss Dynamics
    # ---------------------------------------------------------
    smooth_losses = compute_ema(losses, weight=SMOOTHING_WEIGHT)
    
    ax1.plot(steps, losses, color="#3498db", linewidth=1.5, alpha=0.5, label="Raw Batch Loss")
    ax1.plot(steps, smooth_losses, color="#e74c3c", linewidth=2.5, label="EMA Smoothed Loss")
    
    ax1.set_ylabel("Cross Entropy Loss", fontsize=12, fontweight="bold", color="#333333")
    ax1.tick_params(axis="y", colors="#333333")
    ax1.legend(loc="upper right", frameon=True, edgecolor='#cccccc')

    summary_text = (
        f"Final Step: {steps[-1]}\n"
        f"Final Loss: {losses[-1]:.4f}\n"
        f"Final LR: {lrs[-1]:.3e}\n"
        f"Final Alpha: {alphas[-1]:.4f}"
    )
    ax1.text(
        0.02, 0.05, summary_text, 
        transform=ax1.transAxes, fontsize=10, family='monospace',
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc", "boxstyle": "round,pad=0.5"},
        verticalalignment="bottom"
    )

    # ---------------------------------------------------------
    # Bottom Subplot: LR & Alpha Schedules
    # ---------------------------------------------------------
    ax2.plot(steps, lrs, color="#27ae60", linewidth=2.0, label="Learning Rate")
    ax2.set_xlabel("Global Training Step", fontsize=12, fontweight="bold", color="#333333")
    ax2.set_ylabel("Learning Rate", fontsize=12, fontweight="bold", color="#27ae60")
    ax2.tick_params(axis="y", colors="#27ae60")

    ax3 = ax2.twinx()
    ax3.plot(steps, alphas, color="#8e44ad", linewidth=2.0, linestyle="--", label="Mask Alpha")
    ax3.set_ylabel("Pruning Alpha Proportion", fontsize=12, fontweight="bold", color="#8e44ad")
    ax3.tick_params(axis="y", colors="#8e44ad")
    ax3.set_ylim(-0.05, 1.05) 

    lines_2, labels_2 = ax2.get_legend_handles_labels()
    lines_3, labels_3 = ax3.get_legend_handles_labels()
    ax2.legend(lines_2 + lines_3, labels_2 + labels_3, loc="center right", frameon=True, edgecolor='#cccccc')

    # ---------------------------------------------------------
    # Layout & Export
    # ---------------------------------------------------------
    fig.suptitle("AFM-3 IFPruning SFT: Training Dynamics", fontsize=16, fontweight="bold", color="#111111")
    ax1.set_title("Target: gemma-12b | Strategy: Dynamic Activation Sparsity", fontsize=11, color="#666666", pad=12)

    fig.tight_layout()
    fig.subplots_adjust(top=0.92) 

    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"\n[{'='*40}]")
    print(f"Training dynamics plot generated successfully!")
    print(f"Output Path: {OUTPUT_PATH}")
    print(f"Trajectory: {len(steps)} steps processed.")
    print(f"[{'='*40}]\n")

if __name__ == "__main__":
    try:
        data_steps, data_losses, data_lrs, data_alphas = parse_training_log(LOG_PATH)
        generate_professional_plot(data_steps, data_losses, data_lrs, data_alphas)
    except Exception as e:
        print(f"\nExecution Failed: {e}\n")