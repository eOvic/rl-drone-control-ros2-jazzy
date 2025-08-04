# File: ~/ws_ros2/src/px4_rl_project/px4_rl_project/environment/px4_nav_env.py

import rclpy, numpy as np, gymnasium as gym, time
from gymnasium import spaces
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleOdometry

class DroneEnv(gym.Env, Node):
    metadata = {'render_modes': ['human']}
    
    def __init__(self):
        Node.__init__(self, 'drone_env_node')
        gym.Env.__init__(self)
        
        # Environment parameters
        self.observation_shape = (3,)  # [Pos_z, Vel_z, Thrust]
        self.target_altitude = 20.0  # Target altitude in meters (positive up)
        self.max_episode_steps = 1000
        self.current_step = 0
        
        # Constraints
        self.thrust_min = 0.40
        self.thrust_max = 0.78
        self.vel_z_min = -3.0
        self.vel_z_max = 3.0
        self.pos_z_min = 10.0
        self.pos_z_max = 30.0
        self.target_zone_min = 19.7
        self.target_zone_max = 20.3
        
        # Discrete action space: {0: decrease thrust, 1: increase thrust, 2: restart}
        self.action_space = spaces.Discrete(3)  # 0, 1, 2 (where 2 = restart/-1)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.observation_shape, dtype=np.float32)
        
        # Thrust control
        self.current_thrust = 0.59  # Start in middle of range [0.40, 0.78]
        self.thrust_delta = 0.05    # Thrust change per action
        
        # QoS Profile
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Publishers
        self.offboard_control_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', 10)
        self.trajectory_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', 10)
        self.command_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', 10)
        
        # Subscribers
        self.odom_sub = self.create_subscription(VehicleOdometry, '/fmu/out/vehicle_odometry', self.odom_callback, qos_profile)
        
        # State variables
        self.drone_pos = np.zeros(3)
        self.drone_orientation_q = np.array([1.0, 0.0, 0.0, 0.0])
        self.drone_vel = np.zeros(3)
        self.is_armed = False
        self.offboard_active = False
        
        # Timer for continuous offboard heartbeat
        self.timer = self.create_timer(0.02, self.timer_callback)  # 50Hz
        
        # Debug printing settings
        self.print_every_n_steps = 10  # Print every 10 steps
        self.step_counter = 0
        
        self.get_logger().info('DroneEnv has been initialized.')

    def odom_callback(self, msg):
        self.drone_pos = np.array(msg.position)
        # Convert NED to standard coordinates (z-up)
        # PX4 uses NED (North-East-Down), we want z positive up
        self.drone_pos[2] = -self.drone_pos[2]  # Flip z-axis
        
        self.drone_orientation_q = np.array(msg.q)
        self.drone_vel = np.array(msg.velocity)
        # Also flip velocity z-axis
        self.drone_vel[2] = -self.drone_vel[2]
    
    def timer_callback(self):
        """Continuous offboard control mode heartbeat"""
        if self.offboard_active:
            offboard_msg = OffboardControlMode()
            # Switch between position and velocity control based on mode
            if hasattr(self, 'use_position_control') and self.use_position_control:
                offboard_msg.position = True
                offboard_msg.velocity = False
            else:
                offboard_msg.position = False
                offboard_msg.velocity = True
            offboard_msg.acceleration = False
            offboard_msg.attitude = False
            offboard_msg.body_rate = False
            offboard_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
            self.offboard_control_pub.publish(offboard_msg)
    
    def publish_vehicle_command(self, command, **kwargs):
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = float(kwargs.get("param1", 0.0))
        msg.param2 = float(kwargs.get("param2", 0.0))
        msg.param7 = float(kwargs.get("param7", 0.0))
        msg.target_system = 1
        msg.target_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.command_pub.publish(msg)

    def print_drone_state(self, action_taken=None, reward=None, forced_action=False):
        """Print current drone position, velocity, and thrust"""
        pos_z = self.drone_pos[2]
        vel_z = self.drone_vel[2]
        
        print(f"\n--- STEP {self.current_step} ---")
        print(f"Pos_z:   {pos_z:+7.2f} m  (Target: {self.target_altitude:.1f} m)")
        print(f"Vel_z:   {vel_z:+7.2f} m/s  (Range: [{self.vel_z_min}, {self.vel_z_max}])")
        print(f"Thrust:  {self.current_thrust:7.3f} (Range: [{self.thrust_min}, {self.thrust_max}])")
        
        if action_taken is not None:
            action_names = ["DECREASE", "INCREASE", "RESTART"]
            action_str = f"{action_taken} ({action_names[action_taken]})"
            if forced_action:
                action_str += " [FORCED]"
            print(f"Action:  {action_str}")
        
        if reward is not None:
            print(f"Reward:  {reward:+7.2f}")
        
        # Check if in target zone
        if self.target_zone_min < pos_z < self.target_zone_max:
            print("🎯 IN TARGET ZONE! 🎯")
        else:
            error = abs(pos_z - self.target_altitude)
            print(f"Target Error: {error:6.2f} m")
        
        # Show constraint violations
        violations = []
        if vel_z > self.vel_z_max:
            violations.append(f"Vel_z too high ({vel_z:.2f})")
        if vel_z < self.vel_z_min:
            violations.append(f"Vel_z too low ({vel_z:.2f})")
        if pos_z < self.pos_z_min or pos_z > self.pos_z_max:
            violations.append(f"Pos_z out of range ({pos_z:.2f})")
        
        if violations:
            print(f"⚠️  Violations: {', '.join(violations)}")
        
        print("-" * 40)



    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.step_counter = 0
        
        self.get_logger().info("--- New Episode: Starting Arming & Takeoff Sequence ---")
        
        # Reset state
        self.is_armed = False
        self.offboard_active = False
        self.current_thrust = 0.59  # Reset thrust to middle of valid range
        self.use_position_control = True  # Use position control for takeoff
        
        # Short delay and get current position
        time.sleep(0.2)
        rclpy.spin_once(self, timeout_sec=0.1)
        initial_position = self.drone_pos.copy()
        
        # Step 1: Start offboard heartbeat with position setpoints
        self.get_logger().info("Starting offboard heartbeat with position control...")
        self.offboard_active = True
        
        # Stream initial position setpoints
        for i in range(25):  # 0.5 seconds
            traj_msg = TrajectorySetpoint()
            traj_msg.position = [initial_position[0], initial_position[1], -initial_position[2]]  # NED
            traj_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
            self.trajectory_pub.publish(traj_msg)
            
            rclpy.spin_once(self, timeout_sec=0.01)
            time.sleep(0.02)
        
        # Step 2: Set to offboard mode
        self.get_logger().info("Setting to Offboard mode...")
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
            param1=1.0,  # MAV_MODE_FLAG_CUSTOM_MODE_ENABLED
            param2=6.0   # PX4_CUSTOM_MAIN_MODE_OFFBOARD
        )
        
        # Continue streaming position
        for i in range(15):  # 0.3 seconds
            traj_msg = TrajectorySetpoint()
            traj_msg.position = [initial_position[0], initial_position[1], -initial_position[2]]  # NED
            traj_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
            self.trajectory_pub.publish(traj_msg)
            
            rclpy.spin_once(self, timeout_sec=0.01)
            time.sleep(0.02)
        
        # Step 3: Force arm the vehicle
        self.get_logger().info("Force arming vehicle...")
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            param1=1.0,      # 1 to arm, 0 to disarm
            param2=21196.0   # Force arm (bypass preflight checks)
        )
        
        # Continue streaming position
        for i in range(15):  # 0.3 seconds
            traj_msg = TrajectorySetpoint()
            traj_msg.position = [initial_position[0], initial_position[1], -initial_position[2]]  # NED
            traj_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
            self.trajectory_pub.publish(traj_msg)
            
            rclpy.spin_once(self, timeout_sec=0.01)
            time.sleep(0.02)
        
        self.is_armed = True
        self.get_logger().info("--- Arming Complete, Starting Takeoff ---")
        
        # Step 4: Takeoff using position setpoints
        self.get_logger().info(f"Taking off to {self.target_altitude}m using position control...")
        
        takeoff_start_time = time.time()
        takeoff_timeout = 30.0  # 30 seconds max for takeoff
        target_pos_ned = [initial_position[0], initial_position[1], -self.target_altitude]  # NED coordinates
        
        # Gradual climb to avoid aggressive maneuvers
        climb_steps = 50  # Number of intermediate positions
        for step in range(climb_steps + 1):
            if time.time() - takeoff_start_time > takeoff_timeout:
                self.get_logger().warning("Takeoff timeout reached")
                break
                
            # Interpolate between current and target altitude
            progress = step / climb_steps
            current_target_alt = initial_position[2] + progress * (self.target_altitude - initial_position[2])
            current_pos_ned = [initial_position[0], initial_position[1], -current_target_alt]
            
            # Send position setpoint for 0.5 seconds per step
            for _ in range(25):  # 0.5 seconds at 50Hz
                traj_msg = TrajectorySetpoint()
                traj_msg.position = current_pos_ned
                traj_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
                self.trajectory_pub.publish(traj_msg)
                
                rclpy.spin_once(self, timeout_sec=0.01)
                time.sleep(0.02)
            
            # Check current altitude
            current_alt = self.drone_pos[2]
            
            # Print progress every 10 steps
            if step % 10 == 0:
                self.get_logger().info(f"Takeoff progress: {current_alt:.1f}m / {self.target_altitude:.1f}m")
            
            # If we're close enough to target, break early
            if abs(current_alt - self.target_altitude) < 1.0:
                self.get_logger().info(f"Target altitude reached early at step {step}")
                break
        
        # Hold final position for stabilization
        self.get_logger().info("Stabilizing at target altitude...")
        for _ in range(50):  # 1 second
            traj_msg = TrajectorySetpoint()
            traj_msg.position = target_pos_ned
            traj_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
            self.trajectory_pub.publish(traj_msg)
            
            rclpy.spin_once(self, timeout_sec=0.01)
            time.sleep(0.02)
        
        # Switch back to velocity control for training
        self.get_logger().info("Switching to velocity control for training...")
        self.use_position_control = False
        
        # Start velocity control with zero velocity
        for _ in range(25):  # 0.5 seconds
            traj_msg = TrajectorySetpoint()
            traj_msg.velocity = [0.0, 0.0, 0.0]  # NED
            traj_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
            self.trajectory_pub.publish(traj_msg)
            
            rclpy.spin_once(self, timeout_sec=0.01)
            time.sleep(0.02)
        
        # Final position check
        rclpy.spin_once(self, timeout_sec=0.1)
        final_alt = self.drone_pos[2]
        
        if abs(final_alt - self.target_altitude) < 3.0:
            self.get_logger().info(f"Takeoff successful! Final altitude: {final_alt:.2f}m")
        else:
            self.get_logger().warning(f"Takeoff incomplete. Altitude: {final_alt:.2f}m (target: {self.target_altitude:.1f}m)")
        
        # Print initial state
        print(f"\n🚁 NEW EPISODE STARTED 🚁")
        self.print_drone_state()
        
        return self._get_observation(), {}

    def step(self, action):
        self.current_step += 1
        self.step_counter += 1
        
        original_action = action
        forced_action = False
        
        # Apply velocity constraints with forced actions
        vel_z = self.drone_vel[2]
        if vel_z > self.vel_z_max:  # vel_z > 3.0
            action = 0  # Force decrease thrust
            forced_action = True
            self.get_logger().info(f"Forced action=0 due to vel_z={vel_z:.2f} > {self.vel_z_max}")
        elif vel_z < self.vel_z_min:  # vel_z < -3.0
            action = 1  # Force increase thrust
            forced_action = True
            self.get_logger().info(f"Forced action=1 due to vel_z={vel_z:.2f} < {self.vel_z_min}")
        
        # Handle restart action (action 2 maps to -1 in your specification)
        if original_action == 2:  # Restart action
            self.get_logger().info("Agent requested environment restart")
            print(f"\n🔄 RESTART REQUESTED (Step {self.current_step}) 🔄")
            self.print_drone_state(action_taken=2, forced_action=False)
            observation = self._get_observation()
            return observation, 0.0, True, False, {"restart_requested": True}
        
        # Update thrust based on action with constraints
        if action == 0:  # Decrease thrust
            self.current_thrust = max(self.thrust_min, self.current_thrust - self.thrust_delta)
        elif action == 1:  # Increase thrust
            self.current_thrust = min(self.thrust_max, self.current_thrust + self.thrust_delta)
        
        # Ensure we're still in offboard mode
        if not self.offboard_active:
            self.offboard_active = True
        
        # Convert thrust to velocity command
        # Map thrust [0.40, 0.78] to reasonable vertical velocities
        thrust_normalized = (self.current_thrust - self.thrust_min) / (self.thrust_max - self.thrust_min)
        thrust_velocity = -4.0 + (thrust_normalized * 8.0)  # Range: [-4, +4] m/s
        
        # Send velocity setpoint (in NED coordinates for PX4)
        setpoint_msg = TrajectorySetpoint()
        setpoint_msg.velocity = [0.0, 0.0, -thrust_velocity]  # Negative for NED up
        setpoint_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_pub.publish(setpoint_msg)
        
        # Process ROS messages
        rclpy.spin_once(self, timeout_sec=0.1)
        
        # Get observation
        observation = self._get_observation()
        
        # Calculate reward (exact specification)
        reward = self._calculate_reward()
        
        # Print drone state periodically
        if self.step_counter % self.print_every_n_steps == 0:
            self.print_drone_state(action_taken=action, reward=reward, forced_action=forced_action)
        
        # Check termination conditions
        terminated = self._check_termination()
        
        # Check if max episode steps reached
        truncated = self.current_step >= self.max_episode_steps
        
        # Print final state if episode ends
        if terminated or truncated:
            print(f"\n🏁 EPISODE ENDED (Step {self.current_step}) 🏁")
            self.print_drone_state(action_taken=action, reward=reward, forced_action=forced_action)
            if terminated:
                print("Reason: Termination condition met")
            if truncated:
                print("Reason: Max episode steps reached")
            print("=" * 50)
        
        return observation, reward, terminated, truncated, {}

    def _calculate_reward(self):
        """Exact binary reward specification"""
        pos_z = self.drone_pos[2]
        
        if self.target_zone_min < pos_z < self.target_zone_max:  # 19.7 < Pos_z < 20.3
            return 1.0
        else:
            return 0.0

    def _check_termination(self):
        """Check for episode termination based on constraints"""
        # Give the drone time to take off
        if self.current_step < 50:
            return False
        
        pos_z = self.drone_pos[2]
        
        # Terminate if altitude is out of training range [10, 30]
        if pos_z < self.pos_z_min or pos_z > self.pos_z_max:
            self.get_logger().warning(f"Episode terminated: Pos_z={pos_z:.2f} out of range [{self.pos_z_min}, {self.pos_z_max}]")
            return True
        
        # Terminate if drone hits ground (safety)
        if pos_z < 0.5:
            self.get_logger().warning("Episode terminated: Drone hit ground")
            return True
        
        return False

    def _get_observation(self):
        """Return state as [Pos_z, Vel_z, Thrust]"""
        pos_z = self.drone_pos[2]
        vel_z = self.drone_vel[2]
        
        # Ensure vel_z is within bounds for observation
        vel_z_clamped = np.clip(vel_z, self.vel_z_min, self.vel_z_max)
        
        return np.array([pos_z, vel_z_clamped, self.current_thrust], dtype=np.float32)

    def close(self):
        self.get_logger().info('Closing DroneEnv.')
        self.offboard_active = False
        if hasattr(self, 'timer'):
            self.timer.destroy()
        self.destroy_node()