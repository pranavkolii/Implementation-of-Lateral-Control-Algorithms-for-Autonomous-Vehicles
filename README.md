# Implementation of Lateral Control Algorithms for Autonomous Vehicles

A comprehensive lateral control comparison suite for autonomous vehicle path following in CARLA simulator. This project implements and compares three control algorithms:

- **Pure Pursuit (PP)** - A geometric path-tracking algorithm
- **Linear Quadratic Regulator (LQR)** - An optimal control technique based on linear state feedback
- **Model Predictive Controller (MPC)** - A predictive control algorithm that optimizes over a finite horizon

## Overview

This project simulates a vehicle in the CARLA simulator and evaluates the performance of three different lateral control strategies for path following. The vehicle receives target waypoints, and each controller generates steering commands to minimize lateral and heading errors.

### Quick Video

<a href="https://www.youtube.com/watch?v=4Kh8L4rUMD0" target="_blank">
  <img src="https://img.youtube.com/vi/4Kh8L4rUMD0/maxresdefault.jpg" alt="Lateral Control Comparison - PP vs LQR vs MPC" width="600">
</a>

Click the image above or [watch on YouTube](https://www.youtube.com/watch?v=4Kh8L4rUMD0) to see a quick video of the simulation.

## Features

- **Three Control Algorithms**: Compare Pure Pursuit, LQR, and MPC controllers
- **CARLA Integration**: Runs in the open-source CARLA autonomous driving simulator
- **Configurable Parameters**: All simulation and control parameters are defined in YAML config files
- **Performance Logging**: Outputs CSV files with metrics including:
  - Timestamp
  - Active controller algorithm
  - Vehicle speed
  - Lateral error
  - Heading error
  - Execution time
- **Sensor Simulation**: Includes RGB cameras, LiDAR, GNSS, IMU, and Radar sensors
- **Path Planning**: Automatic route generation using CARLA's GlobalRoutePlanner

## Requirements

- Python 3.8+
- CARLA Simulator 0.9.x or later
- Python packages:
  - `carla` - CARLA simulator Python API
  - `pyyaml` - YAML configuration file parsing
  - `numpy` - Numerical computations
  - `scipy` - Scientific computing utilities (for DARE solver)
  - `cvxpy` - Convex optimization for MPC

## Installation

1. **Install CARLA Simulator**:
   ```bash
   # Download and install CARLA from https://carla.org/
   # Or use Docker: docker pull carlasim/carla:latest
   ```

2. **Install Python Dependencies**:
   ```bash
   pip install pyyaml numpy scipy cvxpy
   ```

3. **Add CARLA Python API to Path** (if not already in PATH):
   ```bash
   export PYTHONPATH=$PYTHONPATH:/home/aakash/CARLA_0.9.16/PythonAPI/carla
   ```

## Usage

### Basic Launch

Start the CARLA server first:
```bash
./CarlaUE4.sh
```

Then run the lateral control suite with your configuration:
```bash
python3 lateral_control_suite.py --config_file config/config.yaml --output_file output/MPC_7.csv --verbose
```

### Command Line Arguments

- `--config_file` (required): Path to YAML configuration file
- `--output_file` (required): Path to output CSV file for results
- `--host` (optional): CARLA server host IP (default: localhost)
- `--port` (optional): CARLA server port (default: 2000)
- `--verbose` or `-v` (optional): Enable debug logging

### Configuration File Structure

The configuration file (`config/config.yaml`) contains three main sections:

#### Vehicle Configuration
```yaml
vehicle:
  type: vehicle.lincoln.mkz_2017  # CARLA vehicle model
  id: hero                         # Vehicle identifier
  spawn_point: 139                 # Spawn point index on map
```

#### Planning Configuration
```yaml
planning_config:
  vehicle_wheelbase: 2.8           # Wheelbase in meters
  target_speed: 3.0                # Target speed in m/s
  lateral_controller: LQR          # PP, LQR, or MPC
  
  # Route configuration
  route_points: [129, 28, 33, 40]  # Waypoint indices or random seed
  randomly_select_route_points_seed: 7
  
  # Controller-specific parameters
  # Pure Pursuit
  PP_lookahead: 4                  # Lookahead distance
  
  # LQR
  Q_cost_lateral: 2.0              # Lateral error penalty
  Q_cost_heading: 0.5              # Heading error penalty
  R_cost: 3.0                      # Control effort penalty
  
  # MPC
  mpc_horizon: 10                  # Prediction horizon steps
  mpc_Q_lateral: 3.0               # Lateral error penalty
  mpc_Q_heading: 1.0               # Heading error penalty
  mpc_R_steer: 0.5                 # Steering effort penalty
  mpc_max_steer_rate: 0.5          # Max steering rate (rad/s)
```

#### General Configuration
```yaml
config:
  simulation_timestep_sec: 0.05    # Simulation time step
  simulation_length: 6000          # Number of steps (<=0 for infinite)
  visualization_timeout_sec: 3600  # Debug visualization duration
```

## Controller Algorithms

### Pure Pursuit (PP)
- Geometric algorithm that computes steering angle to track a lookahead point
- Computationally efficient and simple to implement
- Good for high-speed scenarios
- Parameters: `PP_lookahead`

### Linear Quadratic Regulator (LQR)
- Optimal control theory approach
- Minimizes cost function: `J = ∫(x^T*Q*x + u^T*R*u)dt`
- Uses Discrete Algebraic Riccati Equation (DARE) solver
- Good balance between performance and computation
- Parameters: `Q_cost_lateral`, `Q_cost_heading`, `R_cost`

### Model Predictive Controller (MPC)
- Predictive control that optimizes over a finite horizon
- Uses convex optimization (CVXPY solver)
- Incorporates constraints (steering limits, rate constraints)
- Best performance but higher computational cost
- Parameters: `mpc_horizon`, `mpc_Q_lateral`, `mpc_Q_heading`, `mpc_R_steer`, `mpc_max_steer_rate`

## Output

The simulation generates a CSV file with the following columns:

| Column | Description |
|--------|-------------|
| Timestamp | Simulation step number |
| Algorithm | Active lateral controller (PP/LQR/MPC) |
| Speed_m_s | Current vehicle speed in m/s |
| Lateral_Error_m | Cross-track error from reference path (meters) |
| Heading_Error_rad | Yaw angle error from reference heading (radians) |
| Exec_Time_ms | Controller execution time (milliseconds) |

## Project Structure

```
Compare-PP-LQR-MPC/
├── lateral_control_suite.py    # Main simulation script
├── config/
│   └── config.yaml             # Configuration file
├── output/
│   └── MPC_7.csv               # Example output file
└── README.md                   # This file
```

## Key Methods

### LateralControlSuite Class

- `setup_vehicle()` - Spawns vehicle in CARLA world
- `setup_sensors()` - Attaches sensors to vehicle
- `generate_path()` - Plans path using GlobalRoutePlanner
- `pure_pursuit_controller()` - Implements Pure Pursuit algorithm
- `LQR_controller()` - Implements LQR algorithm
- `MPC_controller()` - Implements MPC algorithm
- `longitudinal_control()` - PID controller for speed regulation
- `get_error_states()` - Computes lateral and heading errors
- `main_loop()` - Main simulation loop

## Outcomes
* Evaluated and implemented three lateral control algorithms—Model Predictive Control (MPC), Pure Pursuit (PP), and Linear Quadratic Regulator (LQR)—using the CARLA simulator and Python.
* Optimized tracking precision and real-time efficiency by developing a latency-compensated LQR controller and a linearized MPC strategy.
