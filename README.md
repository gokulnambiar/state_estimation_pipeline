# State Estimation Pipeline

This project builds a simple state estimation pipeline on a small subset of the Oxford RobotCar Dataset. It starts from GPS trajectory data, adds synthetic measurement noise and missing segments, runs a constant-velocity Kalman filter, and then applies a lightweight regression model as a correction step.

## Problem

Raw position measurements are noisy and can disappear for short stretches. The goal here is to compare three levels of estimation quality on the same route:

- noisy observations
- Kalman filter estimates
- regression-corrected estimates built on top of the Kalman state

The implementation stays intentionally small and interview-friendly. There is no SLAM stack, no robotics framework, and no deep learning model.

## Dataset

The pipeline uses the official Oxford RobotCar small sample archive:

- source: `https://robotcar-dataset.robots.ox.ac.uk/downloads/sample_small.tar`
- extracted files: `gps.csv` and `ins.csv`

`gps.csv` provides timestamps and latitude/longitude. `ins.csv` is used to attach velocity estimates when available. GPS coordinates are converted into a local planar `(x, y)` frame so the rest of the pipeline can work in meters.

## Methods

### 1. Preprocessing

- load RobotCar GPS and INS files from `data/`
- sort by timestamp and remove duplicates
- convert latitude/longitude to local `x, y`
- use INS velocity when present, otherwise estimate velocity from position differences

### 2. Corruption Model

- add Gaussian noise to position measurements
- randomly remove contiguous observation segments to simulate missing GPS

### 3. State Estimation

- state vector: position and velocity in 2D
- motion model: constant velocity
- estimator: basic Kalman filter with position updates when observations are available

### 4. Learned Correction

- train a linear regression model on top of the Kalman state
- input features: noisy position, observation flag, Kalman position, Kalman velocity
- target: residual correction from Kalman estimate back to ground-truth position

## Experiments

The pipeline evaluates all combinations of:

- noise standard deviation: `1.5`, `3.0`, `6.0` meters
- missing data ratio: `0.0`, `0.1`, `0.25`

For each scenario it computes RMSE for:

- noisy observations
- Kalman filter output
- regression-corrected output

## Results

Saved metrics in `outputs/metrics.csv` show that the Kalman filter consistently improves over the noisy observations, and the regression correction gives a small gain in some settings.

Example RMSE values:

| Scenario | Noisy | Kalman | Regression |
| --- | ---: | ---: | ---: |
| noise `1.5`, missing `0.00` | 2.11 | 1.47 | 1.48 |
| noise `3.0`, missing `0.10` | 4.17 | 3.61 | 3.59 |
| noise `6.0`, missing `0.25` | 8.03 | 5.89 | 5.76 |

The correction model helps most in the higher-noise scenarios. In easier cases, the Kalman filter already captures most of the available improvement.

## Outputs

Running the pipeline writes the following artifacts to `outputs/`:

- `metrics.csv`
- `clean_trajectory.csv`
- representative trajectory CSVs
- representative trajectory plots
- `rmse_vs_noise.png`
- `rmse_vs_missing_ratio.png`

## Project Structure

```text
state_estimation_pipeline/
├── data/
├── outputs/
├── src/
├── main.py
├── requirements.txt
└── README.md
```
