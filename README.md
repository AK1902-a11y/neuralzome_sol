# EKF Robot Pose and Velocity Estimation

## Problem Statement

This project implements an Extended Kalman Filter (EKF) to fuse drive feedback and IMU data for real-time estimation of robot pose and velocity. The EKF effectively integrates both data sources to estimate the robot's state at each time step, and its performance is validated by comparing its outputs with ground truth data.

## Project Structure

- **EKF_Prediction.ipynb**: Jupyter notebook implementation of the EKF algorithm.
- **EKF_Prediction.py**: Python script version of the EKF algorithm.
- **data.mcap**: MCAP dataset containing sensor data and ground truth pose for validation.
- **data.toml**: TOML file containing robot parameters like wheelbase and track width.

## Tools and Requirements

- Python 3.x
- Jupyter Notebook (for running `EKF_Prediction.ipynb`)
- Required Python libraries:
  - `numpy`
  - `scipy`
  - `matplotlib`
  - `pandas`
  - `toml`
  - `mcap` (for reading MCAP files)

### Installation (Using Virtual Environment)

1. Clone this repository:
   ```bash
   git clone https://github.com/AK1902-a11y/neuralzome_sol.git
   cd neuralzome_sol

