from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


EARTH_RADIUS_METERS = 6_371_000.0


def load_robotcar_trajectory(data_dir: Path) -> pd.DataFrame:
    gps_path = data_dir / "gps.csv"
    ins_path = data_dir / "ins.csv"

    gps_frame = pd.read_csv(gps_path)
    gps_frame = standardize_columns(gps_frame)
    gps_timestamp = select_column(gps_frame, ["timestamp"])
    latitude_column = select_column(gps_frame, ["latitude"])
    longitude_column = select_column(gps_frame, ["longitude"])

    local_x, local_y = gps_to_local_xy(
        gps_frame[latitude_column].to_numpy(),
        gps_frame[longitude_column].to_numpy(),
    )

    trajectory = pd.DataFrame(
        {
            "timestamp": gps_frame[gps_timestamp].astype(np.int64),
            "latitude": gps_frame[latitude_column].astype(float),
            "longitude": gps_frame[longitude_column].astype(float),
            "x": local_x,
            "y": local_y,
        }
    )

    if ins_path.exists():
        ins_frame = pd.read_csv(ins_path)
        ins_frame = standardize_columns(ins_frame)
        ins_timestamp = select_column(ins_frame, ["timestamp"])
        velocity_x_column = select_column(ins_frame, ["velocity", "east"], optional=True)
        velocity_y_column = select_column(ins_frame, ["velocity", "north"], optional=True)

        if velocity_x_column and velocity_y_column:
            ins_velocity = (
                ins_frame[[ins_timestamp, velocity_x_column, velocity_y_column]]
                .rename(
                    columns={
                        ins_timestamp: "timestamp",
                        velocity_x_column: "velocity_x",
                        velocity_y_column: "velocity_y",
                    }
                )
                .sort_values("timestamp")
            )
            trajectory = pd.merge_asof(
                trajectory.sort_values("timestamp"),
                ins_velocity,
                on="timestamp",
                direction="nearest",
            )

    trajectory = trajectory.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    trajectory["time_seconds"] = (trajectory["timestamp"] - trajectory["timestamp"].iloc[0]) / 1e6

    if {"velocity_x", "velocity_y"}.issubset(trajectory.columns):
        trajectory[["velocity_x", "velocity_y"]] = trajectory[["velocity_x", "velocity_y"]].astype(float)
    else:
        trajectory[["velocity_x", "velocity_y"]] = estimate_velocity(trajectory[["x", "y"]].to_numpy(), trajectory["time_seconds"].to_numpy())

    return trajectory.dropna().reset_index(drop=True)


def standardize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = {column: column.strip().lower().replace(" ", "_").replace("/", "_") for column in frame.columns}
    return frame.rename(columns=normalized)


def select_column(frame: pd.DataFrame, tokens: list[str], optional: bool = False) -> str | None:
    for column in frame.columns:
        if all(token in column for token in tokens):
            return column
    if optional:
        return None
    raise KeyError(f"Could not find column containing tokens {tokens}. Available columns: {list(frame.columns)}")


def gps_to_local_xy(latitude: np.ndarray, longitude: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lat_rad = np.radians(latitude)
    lon_rad = np.radians(longitude)

    origin_lat = lat_rad[0]
    origin_lon = lon_rad[0]

    x = (lon_rad - origin_lon) * np.cos(origin_lat) * EARTH_RADIUS_METERS
    y = (lat_rad - origin_lat) * EARTH_RADIUS_METERS
    return x, y


def estimate_velocity(position_xy: np.ndarray, time_seconds: np.ndarray) -> np.ndarray:
    velocity_x = np.gradient(position_xy[:, 0], time_seconds)
    velocity_y = np.gradient(position_xy[:, 1], time_seconds)
    return np.column_stack([velocity_x, velocity_y])
