from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))

import matplotlib
import pandas as pd


matplotlib.use("Agg")

import matplotlib.pyplot as plt


def save_trajectory_plot(frame: pd.DataFrame, output_path: Path, title: str) -> None:
    figure, axis = plt.subplots(figsize=(10, 7))
    axis.plot(frame["x"], frame["y"], label="True", linewidth=2.2)
    axis.plot(frame["noisy_x"], frame["noisy_y"], label="Noisy", alpha=0.55, linewidth=1.2)
    axis.plot(frame["kf_x"], frame["kf_y"], label="Kalman", linewidth=1.8)
    axis.plot(frame["reg_x"], frame["reg_y"], label="Regression-corrected", linewidth=1.8)
    axis.set_title(title)
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_error_curves(metrics_frame: pd.DataFrame, output_dir: Path) -> None:
    plot_metric_curves(
        metrics_frame.groupby("noise_std", as_index=False)[["rmse_noisy", "rmse_kalman", "rmse_regression"]].mean(),
        x_column="noise_std",
        title="RMSE vs Noise Level",
        x_label="Noise standard deviation (m)",
        output_path=output_dir / "rmse_vs_noise.png",
    )
    plot_metric_curves(
        metrics_frame.groupby("missing_ratio", as_index=False)[["rmse_noisy", "rmse_kalman", "rmse_regression"]].mean(),
        x_column="missing_ratio",
        title="RMSE vs Missing Data Ratio",
        x_label="Missing data ratio",
        output_path=output_dir / "rmse_vs_missing_ratio.png",
    )


def plot_metric_curves(
    frame: pd.DataFrame,
    x_column: str,
    title: str,
    x_label: str,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(frame[x_column], frame["rmse_noisy"], marker="o", label="Noisy")
    axis.plot(frame[x_column], frame["rmse_kalman"], marker="o", label="Kalman")
    axis.plot(frame[x_column], frame["rmse_regression"], marker="o", label="Regression-corrected")
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel("RMSE (m)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
