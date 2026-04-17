from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class ScenarioResult:
    scenario_name: str
    noise_std: float
    missing_ratio: float
    rmse_noisy: float
    rmse_kalman: float
    rmse_regression: float


def temporal_train_test_split(frame: pd.DataFrame, train_fraction: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_index = max(10, int(len(frame) * train_fraction))
    return frame.iloc[:split_index].copy(), frame.iloc[split_index:].copy()


def compute_rmse(estimate_xy: np.ndarray, target_xy: np.ndarray) -> float:
    squared_error = np.sum((estimate_xy - target_xy) ** 2, axis=1)
    return float(np.sqrt(np.mean(squared_error)))


def summarize_metrics(frame: pd.DataFrame, scenario_name: str, noise_std: float, missing_ratio: float) -> ScenarioResult:
    observed_xy = frame[["noisy_x", "noisy_y"]].copy()
    observed_xy["noisy_x"] = observed_xy["noisy_x"].interpolate(limit_direction="both")
    observed_xy["noisy_y"] = observed_xy["noisy_y"].interpolate(limit_direction="both")

    target_xy = frame[["x", "y"]].to_numpy()
    return ScenarioResult(
        scenario_name=scenario_name,
        noise_std=noise_std,
        missing_ratio=missing_ratio,
        rmse_noisy=compute_rmse(observed_xy.to_numpy(), target_xy),
        rmse_kalman=compute_rmse(frame[["kf_x", "kf_y"]].to_numpy(), target_xy),
        rmse_regression=compute_rmse(frame[["reg_x", "reg_y"]].to_numpy(), target_xy),
    )


def save_metrics(results: list[ScenarioResult], output_path: Path) -> pd.DataFrame:
    metrics_frame = pd.DataFrame([result.__dict__ for result in results])
    metrics_frame.to_csv(output_path, index=False)
    return metrics_frame

