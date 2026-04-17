from __future__ import annotations

import numpy as np
import pandas as pd


def run_constant_velocity_kalman_filter(frame: pd.DataFrame) -> pd.DataFrame:
    times = frame["time_seconds"].to_numpy()
    observations = frame[["noisy_x", "noisy_y"]].to_numpy()
    observed = frame["observed"].to_numpy(dtype=bool)

    initial_velocity = frame[["velocity_x", "velocity_y"]].iloc[0].to_numpy(dtype=float)
    state = np.array([frame["x"].iloc[0], frame["y"].iloc[0], initial_velocity[0], initial_velocity[1]], dtype=float)
    covariance = np.eye(4) * 10.0
    measurement_matrix = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    measurement_noise = np.eye(2) * estimate_measurement_variance(frame)

    filtered_states = []
    for index in range(len(frame)):
        if index == 0:
            delta_time = max(times[1] - times[0], 1e-2) if len(frame) > 1 else 1e-1
        else:
            delta_time = max(times[index] - times[index - 1], 1e-2)

        transition_matrix = np.array(
            [
                [1.0, 0.0, delta_time, 0.0],
                [0.0, 1.0, 0.0, delta_time],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        process_noise = build_process_noise(delta_time, acceleration_variance=1.0)

        state = transition_matrix @ state
        covariance = transition_matrix @ covariance @ transition_matrix.T + process_noise

        if observed[index]:
            innovation = observations[index] - (measurement_matrix @ state)
            innovation_covariance = measurement_matrix @ covariance @ measurement_matrix.T + measurement_noise
            kalman_gain = covariance @ measurement_matrix.T @ np.linalg.inv(innovation_covariance)
            state = state + kalman_gain @ innovation
            covariance = (np.eye(4) - kalman_gain @ measurement_matrix) @ covariance

        filtered_states.append(state.copy())

    filtered = pd.DataFrame(filtered_states, columns=["kf_x", "kf_y", "kf_vx", "kf_vy"])
    return pd.concat([frame.reset_index(drop=True), filtered], axis=1)


def build_process_noise(delta_time: float, acceleration_variance: float) -> np.ndarray:
    dt2 = delta_time ** 2
    dt3 = delta_time ** 3
    dt4 = delta_time ** 4
    return acceleration_variance * np.array(
        [
            [dt4 / 4.0, 0.0, dt3 / 2.0, 0.0],
            [0.0, dt4 / 4.0, 0.0, dt3 / 2.0],
            [dt3 / 2.0, 0.0, dt2, 0.0],
            [0.0, dt3 / 2.0, 0.0, dt2],
        ]
    )


def estimate_measurement_variance(frame: pd.DataFrame) -> float:
    residuals = (
        frame[["noisy_x", "noisy_y"]].to_numpy() - frame[["x", "y"]].to_numpy()
    )
    variance = np.nanvar(residuals)
    return float(max(variance, 1.0))

