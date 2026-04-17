from __future__ import annotations

import numpy as np
import pandas as pd


def add_measurement_corruption(
    frame: pd.DataFrame,
    noise_std: float,
    missing_ratio: float,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    corrupted = frame.copy()

    noise = rng.normal(0.0, noise_std, size=(len(corrupted), 2))
    corrupted["noisy_x"] = corrupted["x"] + noise[:, 0]
    corrupted["noisy_y"] = corrupted["y"] + noise[:, 1]

    observation_mask = create_segment_dropout_mask(len(corrupted), missing_ratio, rng)
    corrupted["observed"] = observation_mask.astype(int)
    corrupted.loc[~observation_mask, ["noisy_x", "noisy_y"]] = np.nan
    return corrupted


def create_segment_dropout_mask(length: int, missing_ratio: float, rng: np.random.Generator) -> np.ndarray:
    mask = np.ones(length, dtype=bool)
    target_missing = int(length * missing_ratio)

    if target_missing == 0:
        return mask

    missing_count = 0
    while missing_count < target_missing:
        segment_length = int(rng.integers(5, 25))
        start_index = int(rng.integers(0, max(1, length - segment_length)))
        end_index = min(length, start_index + segment_length)
        newly_missing = mask[start_index:end_index].sum()
        mask[start_index:end_index] = False
        missing_count += int(newly_missing)

    return mask

