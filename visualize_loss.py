#!/usr/bin/env python3
"""Plot IFPruning loss, learning rate, and dense-to-sparse transition."""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


def parse_training_log(log_path: Path):
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")
    pattern = re.compile(
        r"Step\s+(\d+)\s+\|\s+Loss=([\d.eE+-]+)\s+\|\s+"
        r"LR=([\d.eE+-]+)\s+\|\s+Alpha=([\d.eE+-]+)"
    )
    steps, losses, learning_rates, mask_alphas = [], [], [], []
    with log_path.open("r", encoding="utf-8", errors="ignore") as log_file:
        for line in log_file:
            match = pattern.search(line)
            if match:
                steps.append(int(match.group(1)))
                losses.append(float(match.group(2)))
                learning_rates.append(float(match.group(3)))
                mask_alphas.append(float(match.group(4)))
    if not steps:
        raise ValueError("No valid training metrics found in the log")
    return steps, losses, learning_rates, mask_alphas


def compute_ema(values, weight: float):
    smoothed = []
    for value in values:
        smoothed.append(value if not smoothed else smoothed[-1] * weight + value * (1 - weight))
    return smoothed


def set_academic_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 14,
            "axes.labelsize": 16,
            "legend.fontsize": 12,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "axes.linewidth": 1.5,
            "lines.linewidth": 2.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "grid.alpha": 0.4,
            "grid.linestyle": "--",
        }
    )


def generate_plot(steps, losses, learning_rates, mask_alphas, output_path: Path, smoothing: float):
    set_academic_style()
    figure, (loss_axis, schedule_axis) = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        dpi=300,
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1]},
    )
    smooth_losses = compute_ema(losses, weight=smoothing)
    raw_line = loss_axis.plot(
        steps, losses, color="#004C99", linestyle=":", alpha=0.6, label="Batch Loss"
    )
    smooth_line = loss_axis.plot(steps, smooth_losses, color="#C00000", label="Smoothed Loss")
    loss_axis.set_ylabel("Cross Entropy Loss")
    loss_axis.grid(True)

    lr_line = schedule_axis.plot(steps, learning_rates, color="#548235", label="Learning Rate")
    schedule_axis.set_xlabel("Training Steps")
    schedule_axis.set_ylabel("Learning Rate", color="#548235")
    schedule_axis.tick_params(axis="y", labelcolor="#548235")
    schedule_axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    schedule_axis.grid(True)
    max_lr = max(learning_rates)
    schedule_axis.set_ylim(0.0, max_lr * 1.1 if max_lr > 0 else 1.0)

    alpha_axis = schedule_axis.twinx()
    alpha_line = alpha_axis.plot(
        steps,
        mask_alphas,
        color="#7030A0",
        linestyle="--",
        label="Mask Alpha",
    )
    alpha_axis.set_ylabel(
        "Dense-to-Sparse Mix Alpha",
        color="#7030A0",
        rotation=-90,
        va="bottom",
        labelpad=15,
    )
    alpha_axis.tick_params(axis="y", labelcolor="#7030A0")
    alpha_axis.set_ylim(0.0, 1.0)

    lines = raw_line + smooth_line + lr_line + alpha_line
    figure.legend(
        lines,
        [line.get_label() for line in lines],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
        ncol=4,
        frameon=True,
        edgecolor="black",
    )
    figure.subplots_adjust(top=0.88, hspace=0.1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight", pad_inches=0.1)
    plt.close(figure)
    print(f"Plot generated: {output_path} ({len(steps)} logged steps)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("gemma-12B-ifpruning-output/logs/rank_0.log"),
    )
    parser.add_argument("--output", type=Path, default=Path("figs/loss_curve.png"))
    parser.add_argument("--smoothing", type=float, default=0.75)
    args = parser.parse_args()
    if not 0.0 <= args.smoothing < 1.0:
        parser.error("--smoothing must be in [0, 1)")
    generate_plot(
        *parse_training_log(args.log),
        output_path=args.output,
        smoothing=args.smoothing,
    )


if __name__ == "__main__":
    main()
