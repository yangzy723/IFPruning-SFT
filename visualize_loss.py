#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IFPruning SFT Training Dynamics Visualizer (Conference Optimized)
"""

import re
from pathlib import Path
import matplotlib.pyplot as plt

# 1. 配置区
LOG_PATH = Path("./gemma-12B-ifpruning-output/logs/rank_0.log")
OUTPUT_PATH = Path("./loss_curve.png")

SMOOTHING_WEIGHT = 0.75

# 请根据实际使用的模型修改 ORIGINAL_FFN_DIM (例如 Gemma-4-12B 为 14336)
ORIGINAL_FFN_DIM = 14336 
TARGET_FFN_DIM = 4096

# 2. 数据解析与转换
def parse_training_log(log_path: Path):
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

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
        raise ValueError("No valid training metrics found in the log.")
        
    max_pruning_ratio = (ORIGINAL_FFN_DIM - TARGET_FFN_DIM) / ORIGINAL_FFN_DIM
    pruning_ratios = [alpha * max_pruning_ratio for alpha in alphas]

    return steps, losses, lrs, pruning_ratios, max_pruning_ratio

def compute_ema(values, weight=0.85):
    smoothed = []
    for val in values:
        if not smoothed:
            smoothed.append(val)
        else:
            smoothed.append(smoothed[-1] * weight + val * (1 - weight))
    return smoothed

# 3. 绘图风格
def set_academic_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "legend.fontsize": 12,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "axes.linewidth": 1.5,
        "lines.linewidth": 2.0,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "grid.alpha": 0.4,
        "grid.linestyle": "--"
    })

# 4. 渲染图表
def generate_conference_plot(steps, losses, lrs, pruning_ratios, target_max_ratio):
    set_academic_style()
    
    fig, (ax1, ax2) = plt.subplots(
        2, 1, 
        figsize=(10, 8), 
        dpi=300, 
        sharex=True, 
        gridspec_kw={'height_ratios': [2.2, 1]}
    )
    
    # ------------------ 顶部子图: Loss ------------------
    smooth_losses = compute_ema(losses, weight=SMOOTHING_WEIGHT)

    line1 = ax1.plot(steps, losses, color="#004C99", linewidth=1.5, linestyle=":", alpha=0.6, label="Batch Loss")
    line2 = ax1.plot(steps, smooth_losses, color="#C00000", linewidth=2, linestyle="-", label="Smoothed Loss")
    
    ax1.set_ylabel("Cross Entropy Loss")
    ax1.grid(True)
    for spine in ax1.spines.values():
        spine.set_color('black')

    # ------------------ 底部子图: LR & Sparsity ------------------
    line3 = ax2.plot(steps, lrs, color="#548235", linewidth=2.0, linestyle="-", label="Learning Rate")
    ax2.set_xlabel("Training Steps")
    ax2.set_ylabel("Learning Rate", color="#548235") 
    ax2.tick_params(axis='y', labelcolor="#548235")
    ax2.grid(True)
    
    # 获取 LR 的最大值以设置合理的 Y 轴上限
    max_lr = max(lrs)
    ax2.set_ylim(0.0, max_lr * 1.1)

    ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
    ax2.yaxis.get_offset_text().set_color("#548235")

    # 次坐标轴 (Effective Sparsity)
    ax3 = ax2.twinx()
    line4 = ax3.plot(steps, pruning_ratios, color="#7030A0", linewidth=2.0, linestyle="--", label="Effective Sparsity")
    ax3.set_ylabel("Effective Pruning Ratio", color="#7030A0", rotation=-90, va="bottom", labelpad=15)
    ax3.tick_params(axis='y', labelcolor="#7030A0")
    
    ax3.set_ylim(0.0, 1.0) 
    ax3.tick_params(direction="in")

    for spine in ax2.spines.values():
        spine.set_color('black')

    # ------------------ 合并图例 ------------------
    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    
    fig.subplots_adjust(top=0.88, hspace=0.1) 
    
    fig.legend(
        lines, 
        labels, 
        loc="upper center", 
        bbox_to_anchor=(0.5, 0.96),
        ncol=4, 
        frameon=True, 
        edgecolor="black", 
        fancybox=False, 
        framealpha=1.0
    )

    fig.savefig(OUTPUT_PATH, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

    print(f"Academic plot generated: {OUTPUT_PATH}")
    print(f"Total steps processed: {len(steps)}")
    print(f"Target max pruning ratio: {target_max_ratio * 100:.2f}%")

if __name__ == "__main__":
    try:
        data_steps, data_losses, data_lrs, data_pruning, max_pruning = parse_training_log(LOG_PATH)
        generate_conference_plot(data_steps, data_losses, data_lrs, data_pruning, max_pruning)
    except Exception as e:
        print(f"Execution Failed: {e}")