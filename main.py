from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_utils import load_robotcar_trajectory
from src.download_data import download_robotcar_subset
from src.evaluation import save_metrics, summarize_metrics, temporal_train_test_split
from src.kalman import run_constant_velocity_kalman_filter
from src.models import train_regression_correction_model
from src.plotting import save_error_curves, save_trajectory_plot
from src.simulation import add_measurement_corruption


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
NOISE_LEVELS = [1.5, 3.0, 6.0]
MISSING_RATIOS = [0.0, 0.1, 0.25]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    download_robotcar_subset(DATA_DIR)
    base_trajectory = load_robotcar_trajectory(DATA_DIR)

    training_scenario = add_measurement_corruption(base_trajectory, noise_std=3.0, missing_ratio=0.1, seed=7)
    training_scenario = run_constant_velocity_kalman_filter(training_scenario)
    train_frame, _ = temporal_train_test_split(training_scenario)
    correction_model = train_regression_correction_model(train_frame)

    scenario_results = []
    scenario_frames: dict[str, pd.DataFrame] = {}

    for noise_std in NOISE_LEVELS:
        for missing_ratio in MISSING_RATIOS:
            scenario_name = f"noise_{noise_std:.1f}_missing_{missing_ratio:.2f}"
            scenario_frame = add_measurement_corruption(
                base_trajectory,
                noise_std=noise_std,
                missing_ratio=missing_ratio,
                seed=int(noise_std * 100 + missing_ratio * 1000),
            )
            scenario_frame = run_constant_velocity_kalman_filter(scenario_frame)
            predictions = correction_model.predict(scenario_frame)
            scenario_frame = pd.concat([scenario_frame, predictions], axis=1)

            _, test_frame = temporal_train_test_split(scenario_frame)
            scenario_results.append(
                summarize_metrics(
                    test_frame.reset_index(drop=True),
                    scenario_name=scenario_name,
                    noise_std=noise_std,
                    missing_ratio=missing_ratio,
                )
            )
            scenario_frames[scenario_name] = scenario_frame

    metrics_frame = save_metrics(scenario_results, OUTPUT_DIR / "metrics.csv")
    save_error_curves(metrics_frame, OUTPUT_DIR)

    representative_scenarios = [
        "noise_1.5_missing_0.00",
        "noise_3.0_missing_0.10",
        "noise_6.0_missing_0.25",
    ]
    for scenario_name in representative_scenarios:
        scenario_frames[scenario_name].to_csv(OUTPUT_DIR / f"{scenario_name}_trajectory.csv", index=False)
        save_trajectory_plot(
            scenario_frames[scenario_name],
            OUTPUT_DIR / f"{scenario_name}_trajectory.png",
            title=scenario_name.replace("_", " "),
        )

    base_trajectory.to_csv(OUTPUT_DIR / "clean_trajectory.csv", index=False)


if __name__ == "__main__":
    main()
