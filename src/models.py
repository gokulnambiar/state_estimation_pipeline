from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = ["noisy_x", "noisy_y", "observed", "kf_x", "kf_y", "kf_vx", "kf_vy"]


@dataclass
class CorrectionModel:
    pipeline: Pipeline

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        residual_predictions = self.pipeline.predict(frame[FEATURE_COLUMNS].fillna(0.0))
        corrected_positions = frame[["kf_x", "kf_y"]].to_numpy() + residual_predictions
        return pd.DataFrame(corrected_positions, columns=["reg_x", "reg_y"], index=frame.index)


def train_regression_correction_model(train_frame: pd.DataFrame) -> CorrectionModel:
    residual_targets = train_frame[["x", "y"]].to_numpy() - train_frame[["kf_x", "kf_y"]].to_numpy()
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    )
    pipeline.fit(train_frame[FEATURE_COLUMNS].fillna(0.0), residual_targets)
    return CorrectionModel(pipeline=pipeline)
