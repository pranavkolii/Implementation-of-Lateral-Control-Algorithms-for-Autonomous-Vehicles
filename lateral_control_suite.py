import carla
from agents.navigation.global_route_planner import GlobalRoutePlanner

import argparse
import yaml
import logging
import math
import time
import csv
import random
import numpy as np
import scipy.linalg as la
import cvxpy as cp


class LateralControlSuite:
    def __init__(self, args):
        self.world = None
        self.vehicle = None
        self.sensors = []
        self.original_settings = None

        # Data Logging Setup (override existing file is present)
        self.log_file = open(args.output_file, 'w', newline='')
        self.csv_writer = csv.writer(self.log_file)
        self.csv_writer.writerow(['Timestamp', 'Algorithm', 'Speed_m_s',
                                 'Lateral_Error_m', 'Heading_Error_rad', 'Exec_Time_ms'])

        # variables
        # Previous Steering for Incremental Control (MPC/LQR smoothing)
        self.last_steer = 0.0
        self.max_steer_rad = math.radians(70)  # for vehicle.lincoln.mkz_2017

        self.client = carla.Client(args.host, args.port)
        self.client.set_timeout(20.0)

        logging.info("Connected with the Carla server")

        # Load the config from the file
        with open(args.config_file) as f:
            config_file = yaml.safe_load(f)

        self.config = config_file["config"]
        self.planning_config = config_file["planning_config"]

        # Load the world and set the synchronous mode true
        self.world = self.client.get_world()
        self.original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.config["simulation_timestep_sec"]
        self.world.apply_settings(settings)

        self.traffic_manager = self.client.get_trafficmanager()
        self.traffic_manager.set_synchronous_mode(True)

        # Load the vehicle and attach sensors to it
        self.vehicle = self.setup_vehicle(config_file.get("vehicle"))
        self.sensors = self.setup_sensors(config_file.get("sensors", []))

        # plan the path if random path selected
        if self.planning_config["randomly_select_route_points_seed"] > 0:
            # Seed RNG
            random.seed(
                self.planning_config["randomly_select_route_points_seed"])

            map = self.world.get_map()
            spawn_points = map.get_spawn_points()
            self.route_points_random = random.sample(
                range(0, len(spawn_points)), 10)   # 10 unique numbers the spawn points

        logging.info("Initialization completed")

        self.main_loop()

    def cleanup(self):
        logging.info("Exiting the code, cleaning up everything.")

        if self.log_file:
            self.log_file.close()

        if self.original_settings:
            self.world.apply_settings(self.original_settings)

        for sensor in self.sensors:
            sensor.destroy()

        if self.vehicle:
            self.vehicle.destroy()
            self.vehicle = None

        logging.info("Cleanup complete")

    # --------------------------------------- Vehicle and Sensor setup ---------------------------------------

    def setup_vehicle(self, config):
        """
        Spawn the vehicle, according to the config file
        """
        logging.debug("Spawning vehicle: {}".format(config.get("type")))

        bp_library = self.world.get_blueprint_library()
        map_ = self.world.get_map()

        bp = bp_library.filter(config.get("type"))[0]
        bp.set_attribute("role_name", config.get("id"))
        bp.set_attribute("ros_name", config.get("id"))

        return self.world.spawn_actor(
            bp,
            map_.get_spawn_points()[config.get("spawn_point")],
            attach_to=None)

    def setup_sensors(self, sensors_config):
        """
        Attach the required sensors in the given location on the vehicle, according to the config file
        """
        bp_library = self.world.get_blueprint_library()

        sensors = []
        for sensor in sensors_config:
            logging.debug("Spawning sensor: {}".format(sensor))

            bp = bp_library.filter(sensor.get("type"))[0]
            bp.set_attribute("ros_name", sensor.get("id"))
            bp.set_attribute("role_name", sensor.get("id"))
            for key, value in sensor.get("attributes", {}).items():
                bp.set_attribute(str(key), str(value))

            wp = carla.Transform(
                location=carla.Location(
                    x=sensor["spawn_point"]["x"], y=-sensor["spawn_point"]["y"], z=sensor["spawn_point"]["z"]),
                rotation=carla.Rotation(
                    roll=sensor["spawn_point"]["roll"], pitch=-sensor["spawn_point"]["pitch"], yaw=-sensor["spawn_point"]["yaw"])
            )

            sensors.append(
                self.world.spawn_actor(
                    bp,
                    wp,
                    attach_to=self.vehicle
                )
            )

            sensors[-1].enable_for_ros()

        return sensors

    # --------------------------------------- Path Generation ---------------------------------------

    def visualize_all_spawn_points(self):
        """
        Visualize all the spawn points in the current world, helper function to determine the route
        """
        map = self.world.get_map()
        spawn_points = map.get_spawn_points()

        logging.info(
            f"There are {len(spawn_points)} spawn points in this map, visualizing them")

        visualization_timeout_sec = self.config["visualization_timeout_sec"]
        for idx in range(len(spawn_points)):
            loc = spawn_points[idx].location
            self.world.debug.draw_string(loc, str(idx),
                                         color=carla.Color(0, 0, 255),
                                         life_time=visualization_timeout_sec)
            self.world.debug.draw_point(loc, size=0.2, color=carla.Color(
                255, 0, 0), life_time=visualization_timeout_sec)

    def generate_path(self, route_waypoints_list: list[int], visualize=True) -> list:
        """
        function to take the list of route waypoints (spawn points), and give out fine
        poses (with say 20cm distance b/w poses), for the car to follow the poses acurately
        """
        full_route_poses = []

        # Get the List of Spawn Points, and verify if these waypoints are valid
        map = self.world.get_map()
        spawn_points = map.get_spawn_points()
        logging.info(f'This map has {len(spawn_points)} spawn points')
        if len(spawn_points) < max(route_waypoints_list) + 1:
            logging.info(
                f"Error: Map only has {len(spawn_points)} points. Cannot reach index {max(route_waypoints_list)}.")
            return

        resolution = self.planning_config["path_resolution"]
        visualization_timeout_sec = self.config["visualization_timeout_sec"]

        # Initialize the global route planner

        grp = GlobalRoutePlanner(map, sampling_resolution=resolution)
        logging.info(
            f"Planning path for sequence: {route_waypoints_list} with {resolution} meter resolution...")

        # Loop through the sequence and plan segments
        for i in range(len(route_waypoints_list) - 1):
            start_index = route_waypoints_list[i]
            end_index = route_waypoints_list[i+1]

            start_loc = spawn_points[start_index].location
            end_loc = spawn_points[end_index].location

            # trace_route returns a list of (waypoint, road_option)
            route_segment = grp.trace_route(start_loc, end_loc)

            # Extract just the locations from the waypoints
            for waypoint, _ in route_segment:
                full_route_poses.append(waypoint.transform)

        if (visualize):
            arrow_freq = self.config["path_arrow_freq"]
            arrow_len = self.config["path_arrow_len"]
            arrow_thick = self.config["path_arrow_dia"]

            # Highlight the route waypoint list (spawn points)
            for idx in route_waypoints_list:
                loc = spawn_points[idx].location + carla.Location(z=0.5)
                self.world.debug.draw_string(loc, str(idx),
                                             color=carla.Color(255, 0, 0),
                                             life_time=visualization_timeout_sec)
                self.world.debug.draw_point(loc, size=0.2, color=carla.Color(
                    255, 0, 0), life_time=visualization_timeout_sec)

            logging.info(
                f"Drawing path with {len(full_route_poses)} poses. Showing arrows every {arrow_freq} poses.")

            # Visualize the poses between the route waypoints
            for i in range(len(full_route_poses)):
                if i % arrow_freq != 0:
                    continue

                # compute the start and end location for the arrow
                # lift up slightly to make it more visible
                loc = full_route_poses[i].location + carla.Location(z=0.5)
                forward_vec = full_route_poses[i].get_forward_vector()
                end_loc = loc + (forward_vec * arrow_len)

                # draw the arrow
                self.world.debug.draw_arrow(loc, end_loc,
                                            thickness=arrow_thick,
                                            arrow_size=arrow_thick,
                                            color=carla.Color(255, 0, 0),
                                            life_time=visualization_timeout_sec)

        return full_route_poses

    # --------------------------------------- Path Following Helpers ---------------------------------------

    def longitudinal_control(self, target_speed):
        """
        Longitudinal PID controller for speed.
        Returns (throttle, brake) where throttle and brake are in [0,1].
        """
        # Current speed (m/s)
        vel = self.vehicle.get_velocity()
        current_speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

        # Error
        error = target_speed - current_speed

        # Time step
        dt = float(self.config.get("simulation_timestep_sec", 0.05))
        if dt <= 0:
            dt = 0.05

        # Get PID gains from the config file
        Kp = float(self.planning_config.get("speed_Kp", 1.0))
        Ki = float(self.planning_config.get("speed_Ki", 0.0))
        Kd = float(self.planning_config.get("speed_Kd", 0.0))

        # Persistent integrator and last error
        integ = getattr(self, "speed_integral", 0.0)
        last_err = getattr(self, "last_speed_error", 0.0)

        # Integrator update with simple clamp
        integ += error * dt
        integ_limit = float(self.planning_config.get(
            "speed_integral_limit", 10.0))
        integ = max(-integ_limit, min(integ_limit, integ))

        # Derivative
        deriv = (error - last_err) / dt if dt > 0 else 0.0

        # PID output
        pid_out = Kp * error + Ki * integ + Kd * deriv

        # Store back the current output
        self.speed_integral = integ
        self.last_speed_error = error

        # Map PID output to throttle/brake
        # Positive pid_out -> throttle, Negative -> brake
        throttle = 0.0
        brake = 0.0

        if pid_out >= 0.0:
            throttle = max(0.0, min(1.0, pid_out))
        else:
            brake = max(0.0, min(1.0, -pid_out))

        return (throttle, brake)

    def compute_closest_path_index(self, target_path):
        """
        Compute the pose closest to the vehicle in target path, and return its index
        """
        vehicle_loc = self.vehicle.get_transform().location
        min_dist = float('inf')
        closest_idx = 0

        # This search can be optimized to search only near the car's previous known closest index

        for i in range(len(target_path)):
            dist = vehicle_loc.distance(target_path[i].location)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        self.closest_idx = closest_idx

    def get_error_states(self, target_path, lookahead=0):
        """
        Computes Lateral Error (e) and Heading Error (theta_e) based on the closest waypoint.
        Used by LQR and MPC controllers.
        """
        idx = min(self.closest_idx + lookahead, len(target_path) - 1)
        target_wp = target_path[idx]
        vehicle_trans = self.vehicle.get_transform()

        # Vehicle State
        x_v, y_v = vehicle_trans.location.x, vehicle_trans.location.y
        yaw_v = math.radians(vehicle_trans.rotation.yaw)

        # Path State
        x_r, y_r = target_wp.location.x, target_wp.location.y
        yaw_r = math.radians(target_wp.rotation.yaw)

        # Heading Error, normalized to -pi, pi
        heading_error = self.normalize_angle(yaw_v - yaw_r)

        # 2. Lateral Error / Cross Track Error
        # Vector from waypoint to vehicle
        dx = x_v - x_r
        dy = y_v - y_r
        # Project onto the normal vector of the path [-sin(yaw), cos(yaw)]
        lat_error = -dx * math.sin(yaw_r) + dy * math.cos(yaw_r)

        return lat_error, heading_error

    def normalize_angle(self, angle):
        """
        Normalize angle to [-pi, pi]
        """
        while angle > np.pi:
            angle -= 2.0 * np.pi
        while angle < -np.pi:
            angle += 2.0 * np.pi
        return angle

    # --------------------------------------- Controllers ---------------------------------------

    def pure_pursuit_controller(self, target_path):
        """
        It takes in the target path, it knows the vehicle location. 
        It computes the steering angle using pure pursuit algorithm.
        """

        # Get the vehicle's current location
        transform = self.vehicle.get_transform()
        vehicle_loc = transform.location
        vehicle_yaw = math.radians(transform.rotation.yaw)
        lookahead = self.planning_config.get("PP_lookahead", 5)
        target = target_path[min(
            self.closest_idx + lookahead, len(target_path)-1)]
        target_loc = target.location

        # Vector from car to target waypoint
        dx = target_loc.x - vehicle_loc.x
        dy = target_loc.y - vehicle_loc.y

        # Transform to vehicle coordinates
        x = dx * math.cos(vehicle_yaw) + dy * math.sin(vehicle_yaw)
        y = -dx * math.sin(vehicle_yaw) + dy * math.cos(vehicle_yaw)

        # Pure pursuit steering formula
        L = self.planning_config["vehicle_wheelbase"]
        steer = math.atan2(2 * L * y, x * x + y * y)

        # Set limits for the steering cmd generated
        steer = max(-1.0, min(1.0, steer))

        return steer

    def LQR_controller(self, target_path):
        """
        Implements LQR that Minimizes Cost J = Integral(x.T*Q*x + u.T*R*u).
        """
        if v < 0.1:
            return 0.0  # Avoid singularity at zero speed

        # Load the parameters
        dt = self.config["simulation_timestep_sec"]
        L = self.planning_config["vehicle_wheelbase"]
        v_vec = self.vehicle.get_velocity()
        v = math.sqrt(v_vec.x**2 + v_vec.y**2 + v_vec.z**2)

        # Get current State Errors: x = [e_y, e_phi]
        lat_error, head_error = self.get_error_states(target_path, lookahead=0)

        # Predict state 1 step into the future to compensate for delay
        # Future Lat Error = Current Lat + (v * sin(Head Error) * dt)
        pred_lat_error = lat_error + (v * head_error * dt)
        pred_head_error = head_error  # Heading changes slower, usually ok to keep

        state_vector = np.array([[pred_lat_error], [pred_head_error]])

        # Define Discrete State Space Matrices A and B
        # Continuous A = [[0, v], [0, 0]], B = [[0], [v/L]]
        # Discrete Approximation: A_d = I + A*dt, B_d = B*dt
        A = np.array([[1.0, dt * v],
                      [0.0, 1.0]])
        B = np.array([[0.5 * dt * dt * v * v / L],  # Effect on Lateral Error (2nd order)
                      [dt * v / L]])                # Effect on Heading Error

        # Define Q and R Costs
        # Q penalizes lateral error heavily, R penalizes steering effort
        Q = np.array([[self.planning_config["Q_cost_lateral"], 0.0],
                      [0.0, self.planning_config["Q_cost_heading"]]])
        R = np.array([[self.planning_config["R_cost"]]])

        # Solve Discrete Algebraic Riccati Equation
        P = la.solve_discrete_are(A, B, Q, R)

        # Compute Gain K = (R + B.T*P*B)^-1 * (B.T*P*A)
        K = la.inv(R + B.T @ P @ B) @ (B.T @ P @ A)

        # Compute Optimal Control u = -Kx
        state_vector = np.array([[pred_lat_error], [pred_head_error]])

        steer_rad = -(K @ state_vector)[0, 0]
        # Invert sign because CARLA steer<0 often means Left, while Math steer<0 means Right

        # Normalize to CARLA steering [-1, 1] (Assuming max steer ~30-40 deg, e.g. 0.6 rad)
        steer_cmd = np.clip(steer_rad / self.max_steer_rad, -1.0, 1.0)

        return steer_cmd

    def MPC_controller(self, target_path):
        """
        Model Predictive Controller (MPC) using CVXPY.
        Minimizes the objective function subject to vehicle dynamics and input constraints.
        """

        if v < 0.1:
            return 0.0

        # Load Parameters from the config
        dt = self.config["simulation_timestep_sec"]
        L = self.planning_config["vehicle_wheelbase"]

        N = self.planning_config.get(
            "MPC_horizon", 10)         # Prediction Horizon
        Q_lat = self.planning_config.get(
            "Q_cost_lateral", 1.0)  # Weight on lateral error
        Q_head = self.planning_config.get(
            "Q_cost_heading", 1.0)  # Weight on heading error
        # Weight on steering magnitude
        R_steer = self.planning_config.get("R_cost", 0.1)
        # Weight on steering rate (smoothness)
        R_rate = self.planning_config.get("R_cost_rate", 10.0)

        # Get current Velocity
        v_vec = self.vehicle.get_velocity()
        v = math.sqrt(v_vec.x**2 + v_vec.y**2 + v_vec.z**2)

        # 2. Define State and Reference
        # Get current errors [lat_error, head_error]
        # We use lookahead=0 because MPC handles the "looking ahead" internally via the horizon
        e_y, e_psi = self.get_error_states(target_path, lookahead=0)

        x_init = np.array([e_y, e_psi])

        # Define Discrete State Space Matrices (Linearized Bicycle Model)
        # A = [[1, v*dt], [0, 1]]
        # B = [[0.5*dt^2*v^2/L], [dt*v/L]]

        A = np.array([[1.0, dt * v],
                      [0.0, 1.0]])

        B = np.array([[0.5 * dt * dt * v * v / L],
                      [dt * v / L]])

        # 3. Construct the Optimization Problem (CVXPY)
        # State variables: 2 states x (N+1) steps
        x = cp.Variable((2, N + 1))
        # Control variables: 1 control input x N steps
        u = cp.Variable((1, N))

        cost = 0
        constraints = []

        # Initial State Constraint
        constraints.append(x[:, 0] == x_init)

        # Cost Matrices
        Q = np.diag([Q_lat, Q_head])

        # Previous control input for rate limiting (converted to radians)
        prev_u = self.last_steer * self.max_steer_rad

        for k in range(N):
            # Cost Function summation
            # J = x^T Q x + u^T R u + (du)^T R_rate (du)
            cost += cp.quad_form(x[:, k], Q) + \
                cp.sum_squares(u[:, k]) * R_steer

            # Steering Rate Cost (Smoothness)
            if k == 0:
                cost += cp.sum_squares(u[:, k] - prev_u) * R_rate
            else:
                cost += cp.sum_squares(u[:, k] - u[:, k-1]) * R_rate

            # Dynamics Constraints
            # x_{k+1} = A x_k + B u_k
            constraints.append(x[:, k+1] == A @ x[:, k] + B @ u[:, k])

            # Actuation Constraints
            # Clip steering to physical limits of the vehicle
            constraints.append(cp.abs(u[:, k]) <= self.max_steer_rad)

        # Optional, terminal Cost (penalize the final state heavily to ensure convergence)
        cost += cp.quad_form(x[:, N], Q) * 5

        # Solve the optimization problem
        prob = cp.Problem(cp.Minimize(cost), constraints)

        try:
            # OSQP is generally robust for path following QPs
            prob.solve(solver=cp.OSQP, verbose=False)
        except cp.SolverError:
            # Fallback if OSQP fails
            prob.solve(solver=cp.SCS, verbose=False)

        # Extract and Return Control
        if u.value is None or prob.status not in ["optimal", "optimal_inaccurate"]:
            # If solver fails, maintain previous steering to avoid erratic jumps
            logging.warning(
                f"MPC Solver failed ({prob.status}). Maintaining last steer.")
            return self.last_steer

        # Get the first control action (steering angle in radians)
        steer_rad = u.value[0, 0]

        # Convert to CARLA Steering Command [-1.0, 1.0], also left turn negative conversion
        steer_cmd = -steer_rad / self.max_steer_rad

        return np.clip(steer_cmd, -1.0, 1.0)

    # --------------------------------------- Control loop ---------------------------------------

    def main_loop(self):
        try:
            self.visualize_all_spawn_points()

            # plan the path, either pseudo random or through given spawn points
            if self.planning_config["randomly_select_route_points_seed"] > 0:
                target_path = self.generate_path(
                    self.route_points_random, True)
            else:
                target_path = self.generate_path(
                    self.planning_config["route_points"], True)

            simulation_length = self.config["simulation_length"]
            simulation_loop_count = 0

            while True:
                # check the simulation length constraint
                simulation_loop_count += 1
                if simulation_length >= 0:
                    if simulation_loop_count > simulation_length:

                        logging.warning(
                            f"Simulation ran for {simulation_loop_count-1} steps, ending now")

                        break

                # Compute longitudial control (accelerator)
                target_speed = self.planning_config["target_speed"]
                throttle, brake = self.longitudinal_control(target_speed)

                # Pre-computation for lateral control (steering)
                self.compute_closest_path_index(target_path)
                current_lat_error, current_head_error = self.get_error_states(
                    target_path)
                v_vec = self.vehicle.get_velocity()
                current_speed = math.sqrt(v_vec.x**2 + v_vec.y**2 + v_vec.z**2)

                lateral_controller = self.planning_config["lateral_controller"]

                steer_start = time.time()

                # Choose the lateral controller
                match lateral_controller:
                    case "PP":
                        steer = self.pure_pursuit_controller(target_path)
                    case "LQR":
                        steer = self.LQR_controller(target_path)
                    case "MPC":
                        steer = self.MPC_controller(target_path)
                    case _:
                        logging.error("Unknown lateral controller")
                        brake = 1.0
                        throttle = 0.0
                        steer = 0.0

                steer_duration_ms = (time.time() - steer_start) * 1000.0
                self.last_steer = steer  # Store for MPC incremental update

                # Apply Control
                control = carla.VehicleControl()
                control.throttle = throttle
                control.brake = brake
                control.steer = steer
                self.vehicle.apply_control(control)

                # Log Data to CSV for comparison
                self.csv_writer.writerow([
                    simulation_loop_count,
                    lateral_controller,
                    f"{current_speed:.2f}",
                    f"{current_lat_error:.4f}",
                    f"{current_head_error:.4f}",
                    f"{steer_duration_ms:.2f}"
                ])

                logging.info(
                    f"Controller: {lateral_controller} | Speed: {current_speed:.2f} m/s | Lat Error: {current_lat_error:.4f} m | Head Error: {current_head_error:.4f} rad | Exec Time: {steer_duration_ms:.2f} ms")

                # Tick the carla world, for synchronous mode
                _ = self.world.tick()

        except KeyboardInterrupt:
            logging.info("KeyboardInterrupt caught")
        finally:
            self.cleanup()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='CARLA ROS2 native')
    argparser.add_argument('--host', metavar='H', default='localhost',
                           help='IP of the host CARLA Simulator (default: localhost)')
    argparser.add_argument('--port', metavar='P', default=2000,
                           type=int, help='TCP port of CARLA Simulator (default: 2000)')
    argparser.add_argument('-c', '--config_file', default='',
                           required=True, help='Config json file')
    argparser.add_argument('-v', '--verbose', action='store_true',
                           dest='debug', help='print debug information')
    argparser.add_argument('-o', '--output_file', default='',
                           required=True, help='Output csv file')

    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('Listening to server %s:%s', args.host, args.port)

    LateralControlSuite(args)
