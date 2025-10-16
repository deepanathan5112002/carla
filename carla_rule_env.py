import numpy as np
import carla
import gymnasium as gym
from gymnasium import spaces
import yaml
import time
from collections import deque
import threading
import queue
from typing import Dict, Tuple, Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CarlaRuleAwareEnv(gym.Env):
    """Optimized CARLA environment with LIDAR, waypoint following, and collision prediction"""
    
    metadata = {'render_modes': ['rgb_array']}
    
    def __init__(self, config_path: str, port: int = 2000, difficulty: str = 'easy'):
        super().__init__()
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.difficulty = difficulty  # 'easy', 'medium', 'hard'
        
        # Connect to CARLA with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"Connecting to CARLA on port {port} (attempt {attempt+1}/{max_retries})...")
                self.client = carla.Client('localhost', port)
                self.client.set_timeout(15.0)
                self.world = self.client.get_world()
                break
            except RuntimeError as e:
                if attempt == max_retries - 1:
                    raise
                logger.warning(f"Connection failed, retrying in 2s...")
                time.sleep(2)
        
        self._orig_settings = self.world.get_settings()

        # Set synchronous mode with optimized settings
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / 20.0
        settings.substepping = True
        self.world.apply_settings(settings)
        self._fps = 20

        logger.info(f"Connected to map: {self.world.get_map().name}")
        logger.info(f"Difficulty: {self.difficulty}")
        
        # Action space: [steer, throttle, brake]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # ✅ EXPANDED Observation space: vehicle(10) + rule(5) + lidar(6) + waypoint(4) = 25
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(25,), 
            dtype=np.float32
        )
        
        # Initialize components
        self.vehicle = None
        self.sensors = {}
        self.collision_hist = deque(maxlen=100)
        self.lane_invasion_hist = deque(maxlen=100)
        
        # ✅ LIDAR data storage
        self.lidar_data = None
        self.lidar_lock = threading.Lock()
        
        # Perception queue (thread-safe)
        self.perception_queue = queue.Queue(maxsize=2)
        self.last_rule_state = self._default_rule_state()
        self.perception_thread = None
        self.stop_perception = threading.Event()
        
        # ✅ Route planning with CARLA planner
        self.map = self.world.get_map()
        self.spawn_points = self.map.get_spawn_points()
        self.destination = None
        self.last_dist_to_dest = None
        self.route_waypoints = []
        self.current_waypoint_idx = 0
        self.waypoint_spacing = 5.0  # meters between waypoints
        
        # Metrics tracking
        self.episode_metrics = self._reset_metrics()
        self.step_count = 0
        
        # Perception module
        self.detector = None
        self.use_perception = False
        
        # Performance optimization
        self._last_obs_cache = None
        self._cache_valid = False
        self._last_action = None
        
        # ✅ Traffic management based on difficulty
        self.traffic_manager = self.client.get_trafficmanager(8000)
        self.traffic_vehicles = []
    
    def _default_rule_state(self) -> Dict:
        """Default rule state when no signs detected"""
        return {
            'speed_limit': 50,
            'must_stop': False,
            'no_entry': False,
            'traffic_light': 'green',
            'confidence': 0.0
        }

    def _tick_world(self):
        """Advance the world by exactly one frame (sync mode)."""
        self.world.tick()

    def _reset_metrics(self) -> Dict:
        """Reset episode metrics"""
        return {
            'distance_traveled': 0.0,
            'collisions': 0,
            'lane_invasions': 0,
            'speed_violations': 0,
            'stop_violations': 0,
            'route_completion': 0.0,
            'episode_reward': 0.0,
            'steps': 0,
            'off_road_steps': 0,
            'max_speed': 0.0,
            'waypoints_reached': 0,
            'close_calls': 0  # ✅ Track near-collisions
        }
    
    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)
        
        # Clean up previous episode
        self._cleanup()
        
        # ✅ Spawn traffic based on difficulty
        self._spawn_traffic()
        
        # Choose spawn point with better strategy
        spawn_point = self._choose_spawn_point()
        
        # Spawn vehicle with retry
        bp_library = self.world.get_blueprint_library()
        vehicle_bp = bp_library.find('vehicle.tesla.model3')
        
        for attempt in range(3):
            try:
                self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
                break
            except RuntimeError:
                if attempt < 2:
                    spawn_point = np.random.choice(self.spawn_points)
                else:
                    raise
        
        # Small delay for spawn to settle
        time.sleep(0.05)
        self.world.tick()
        
        # Attach sensors (including LIDAR)
        self._setup_sensors()
        
        # ✅ Setup route with CARLA's planner
        self._setup_route()
        
        # Reset metrics
        self.episode_metrics = self._reset_metrics()
        self.last_dist_to_dest = None
        self.step_count = 0
        self._cache_valid = False
        self._last_action = None
        self.current_waypoint_idx = 0
        
        # Start perception thread if enabled
        if self.use_perception and self.detector and not self.perception_thread:
            self._start_perception_thread()
        
        # Tick world to initialize sensors
        self.world.tick()
        time.sleep(0.1)
        
        obs = self._get_observation()
        info = {}
        
        return obs, info
    
    def _spawn_traffic(self):
        """✅ Spawn traffic vehicles based on difficulty"""
        # Clear old traffic
        for vehicle in self.traffic_vehicles:
            if vehicle.is_alive:
                vehicle.destroy()
        self.traffic_vehicles.clear()
        
        # Determine number of vehicles based on difficulty
        if self.difficulty == 'easy':
            num_vehicles = 0  # Empty roads
        elif self.difficulty == 'medium':
            num_vehicles = 20
        elif self.difficulty == 'hard':
            num_vehicles = 50
        else:
            num_vehicles = 0
        
        if num_vehicles == 0:
            logger.info("Easy mode: No traffic vehicles")
            return
        
        bp_library = self.world.get_blueprint_library()
        vehicle_bps = bp_library.filter('vehicle.*')
        
        spawn_points = self.spawn_points.copy()
        np.random.shuffle(spawn_points)
        
        spawned = 0
        for spawn_point in spawn_points[:num_vehicles]:
            try:
                bp = np.random.choice(vehicle_bps)
                vehicle = self.world.spawn_actor(bp, spawn_point)
                vehicle.set_autopilot(True, self.traffic_manager.get_port())
                self.traffic_vehicles.append(vehicle)
                spawned += 1
            except RuntimeError:
                continue
        
        logger.info(f"Spawned {spawned}/{num_vehicles} traffic vehicles")
    
    def _choose_spawn_point(self) -> carla.Transform:
        """Choose a good spawn point (avoid junctions and busy areas)"""
        good_spawns = []
        for sp in self.spawn_points:
            waypoint = self.map.get_waypoint(sp.location)
            if not waypoint.is_junction:
                good_spawns.append(sp)
        
        if not good_spawns:
            good_spawns = self.spawn_points
        
        return np.random.choice(good_spawns)
    
    def _setup_sensors(self):
        """✅ Setup all sensors including LIDAR"""
        bp_library = self.world.get_blueprint_library()
        
        # RGB Camera (lower resolution for performance)
        cam_bp = bp_library.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(self.config['sensors']['camera']['width']))
        cam_bp.set_attribute('image_size_y', str(self.config['sensors']['camera']['height']))
        cam_bp.set_attribute('fov', str(self.config['sensors']['camera']['fov']))
        
        cam_transform = carla.Transform(
            carla.Location(
                x=self.config['sensors']['camera']['x'],
                z=self.config['sensors']['camera']['z']
            )
        )
        
        self.sensors['camera'] = self.world.spawn_actor(
            cam_bp, cam_transform, attach_to=self.vehicle
        )
        self.sensors['camera'].listen(lambda image: self._process_camera(image))
        
        # ✅ LIDAR Sensor
        lidar_bp = bp_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('channels', '32')
        lidar_bp.set_attribute('range', '50.0')
        lidar_bp.set_attribute('points_per_second', '56000')
        lidar_bp.set_attribute('rotation_frequency', '10.0')
        lidar_bp.set_attribute('upper_fov', '10.0')
        lidar_bp.set_attribute('lower_fov', '-30.0')
        
        lidar_transform = carla.Transform(carla.Location(x=0.0, z=2.5))
        self.sensors['lidar'] = self.world.spawn_actor(
            lidar_bp, lidar_transform, attach_to=self.vehicle
        )
        self.sensors['lidar'].listen(lambda data: self._process_lidar(data))
        
        # Collision sensor
        col_bp = bp_library.find('sensor.other.collision')
        self.sensors['collision'] = self.world.spawn_actor(
            col_bp, carla.Transform(), attach_to=self.vehicle
        )
        self.sensors['collision'].listen(lambda event: self.collision_hist.append(event))
        
        # Lane invasion sensor
        lane_bp = bp_library.find('sensor.other.lane_invasion')
        self.sensors['lane_invasion'] = self.world.spawn_actor(
            lane_bp, carla.Transform(), attach_to=self.vehicle
        )
        self.sensors['lane_invasion'].listen(lambda event: self.lane_invasion_hist.append(event))
        
        logger.info("All sensors attached (Camera, LIDAR, Collision, Lane)")
    
    def _process_lidar(self, lidar_measurement):
        """✅ Process LIDAR point cloud for obstacle detection"""
        try:
            # Convert to numpy array
            points = np.frombuffer(lidar_measurement.raw_data, dtype=np.float32)
            points = points.reshape((-1, 4))  # [x, y, z, intensity]
            
            # Filter ground points (z < -1.5m relative to sensor)
            points = points[points[:, 2] > -1.5]
            
            # Define sectors for analysis
            # Front: x > 0, |y| < 4m, x < 30m
            front_mask = (points[:, 0] > 0) & (np.abs(points[:, 1]) < 4.0) & (points[:, 0] < 30.0)
            front_points = points[front_mask]
            
            # Left and Right
            left_mask = (points[:, 0] > 0) & (points[:, 1] < -2.0) & (points[:, 0] < 20.0)
            right_mask = (points[:, 0] > 0) & (points[:, 1] > 2.0) & (points[:, 0] < 20.0)
            
            left_points = points[left_mask]
            right_points = points[right_mask]
            
            # Calculate metrics
            lidar_info = {
                'min_distance_front': 50.0,
                'obstacles_close': 0,      # < 5m
                'obstacles_medium': 0,     # 5-15m
                'obstacles_far': 0,        # 15-30m
                'left_clear': True,
                'right_clear': True
            }
            
            if len(front_points) > 0:
                distances = np.sqrt(front_points[:, 0]**2 + front_points[:, 1]**2)
                lidar_info['min_distance_front'] = float(np.min(distances))
                lidar_info['obstacles_close'] = int(np.sum(distances < 5.0))
                lidar_info['obstacles_medium'] = int(np.sum((distances >= 5.0) & (distances < 15.0)))
                lidar_info['obstacles_far'] = int(np.sum((distances >= 15.0) & (distances < 30.0)))
            
            lidar_info['left_clear'] = len(left_points) < 50
            lidar_info['right_clear'] = len(right_points) < 50
            
            # Thread-safe update
            with self.lidar_lock:
                self.lidar_data = lidar_info
                
        except Exception as e:
            logger.error(f"LIDAR processing error: {e}")
    
    def _process_camera(self, image):
        """Process camera image - runs in CARLA callback thread"""
        try:
            if not self.use_perception:
                return
            
            if self.step_count % 3 != 0:
                return
            
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))
            array = array[:, :, :3]
            
            if not self.perception_queue.full():
                try:
                    self.perception_queue.put_nowait(array)
                except queue.Full:
                    pass
        except Exception as e:
            logger.error(f"Camera processing error: {e}")
    
    def _start_perception_thread(self):
        """Start background perception processing thread"""
        def perception_worker():
            logger.info("Perception thread started")
            while not self.stop_perception.is_set():
                try:
                    frame = self.perception_queue.get(timeout=0.5)
                    rule_state = self.detector.infer(frame)
                    self.last_rule_state = rule_state
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Perception inference error: {e}")
            logger.info("Perception thread stopped")
        
        self.perception_thread = threading.Thread(target=perception_worker, daemon=True)
        self.perception_thread.start()
    
    def _setup_route(self):
        """✅ Setup route using CARLA's global route planner"""
        start_location = self.vehicle.get_location()
        start_waypoint = self.map.get_waypoint(start_location)
        
        # Choose destination based on difficulty
        if self.difficulty == 'easy':
            min_dist, max_dist = 50.0, 100.0
        elif self.difficulty == 'medium':
            min_dist, max_dist = 100.0, 200.0
        else:  # hard
            min_dist, max_dist = 150.0, 300.0
        
        # Find valid destinations
        valid_destinations = [
            sp for sp in self.spawn_points 
            if min_dist < start_location.distance(sp.location) < max_dist
        ]
        
        if not valid_destinations:
            valid_destinations = [
                sp for sp in self.spawn_points 
                if start_location.distance(sp.location) > 30.0
            ]
        
        if valid_destinations:
            destination_transform = np.random.choice(valid_destinations)
        else:
            destination_transform = np.random.choice(self.spawn_points)
        
        self.destination = destination_transform.location
        
        # ✅ Use CARLA's route planner to generate waypoints
        destination_waypoint = self.map.get_waypoint(self.destination)
        
        # Generate route using A* planner
        from agents.navigation.global_route_planner import GlobalRoutePlanner
        
        grp = GlobalRoutePlanner(self.map, self.waypoint_spacing)
        route = grp.trace_route(start_waypoint.transform.location, destination_waypoint.transform.location)
        
        # Extract waypoints from route
        self.route_waypoints = [waypoint for waypoint, _ in route]
        self.current_waypoint_idx = 0
        
        logger.info(f"Route planned: {len(self.route_waypoints)} waypoints, "
                   f"distance: {start_location.distance(self.destination):.1f}m")
    
    def _get_next_waypoint_info(self) -> Dict:
        """✅ Get information about next waypoint to reach"""
        if not self.route_waypoints or self.current_waypoint_idx >= len(self.route_waypoints):
            return {
                'distance': 0.0,
                'angle': 0.0,
                'lane_offset': 0.0,
                'progress': 1.0
            }
        
        vehicle_transform = self.vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        
        # Get next waypoint
        next_wp = self.route_waypoints[self.current_waypoint_idx]
        
        # Distance to next waypoint
        wp_distance = vehicle_location.distance(next_wp.transform.location)
        
        # Check if we reached this waypoint
        if wp_distance < self.waypoint_spacing * 1.5:
            self.current_waypoint_idx += 1
            self.episode_metrics['waypoints_reached'] += 1
            if self.current_waypoint_idx < len(self.route_waypoints):
                next_wp = self.route_waypoints[self.current_waypoint_idx]
                wp_distance = vehicle_location.distance(next_wp.transform.location)
        
        # Calculate angle to waypoint (relative to vehicle heading)
        wp_vector = next_wp.transform.location - vehicle_location
        wp_angle = np.arctan2(wp_vector.y, wp_vector.x)
        
        vehicle_yaw = np.deg2rad(vehicle_transform.rotation.yaw)
        angle_diff = wp_angle - vehicle_yaw
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
        
        # Lane offset from waypoint
        lane_offset = vehicle_location.distance(next_wp.transform.location)
        
        # Route progress
        progress = self.current_waypoint_idx / max(len(self.route_waypoints), 1)
        
        return {
            'distance': wp_distance,
            'angle': angle_diff,
            'lane_offset': lane_offset,
            'progress': progress
        }
    
    def _get_observation(self) -> np.ndarray:
        """✅ Get observation with LIDAR and waypoint info"""
        if self._cache_valid:
            return self._last_obs_cache
        
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        speed_ms = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        speed_kmh = 3.6 * speed_ms
        
        # Update max speed metric
        self.episode_metrics['max_speed'] = max(self.episode_metrics['max_speed'], speed_kmh)
        
        vehicle_location = transform.location
        waypoint = self.map.get_waypoint(vehicle_location)
        
        # Heading error
        waypoint_yaw = waypoint.transform.rotation.yaw
        vehicle_yaw = transform.rotation.yaw
        heading_error = np.deg2rad(waypoint_yaw - vehicle_yaw)
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
        
        # Lane offset
        waypoint_loc = waypoint.transform.location
        lane_offset = vehicle_location.distance(waypoint_loc)
        
        # Distance to destination
        dist_to_dest = vehicle_location.distance(self.destination)
        
        # Flags
        collision_flag = 1.0 if len(self.collision_hist) > 0 else 0.0
        lane_invasion_flag = 1.0 if len(self.lane_invasion_hist) > 0 else 0.0
        
        rule_state = self.last_rule_state
        
        # ✅ LIDAR features
        with self.lidar_lock:
            lidar = self.lidar_data
        
        if lidar is None:
            lidar_features = [1.0, 0.0, 0.0, 0.0, 1.0, 1.0]
        else:
            lidar_features = [
                np.clip(lidar['min_distance_front'] / 50.0, 0.0, 1.0),
                np.clip(lidar['obstacles_close'] / 100.0, 0.0, 1.0),
                np.clip(lidar['obstacles_medium'] / 200.0, 0.0, 1.0),
                np.clip(lidar['obstacles_far'] / 300.0, 0.0, 1.0),
                float(lidar['left_clear']),
                float(lidar['right_clear'])
            ]
        
        # ✅ Waypoint features
        wp_info = self._get_next_waypoint_info()
        waypoint_features = [
            np.clip(wp_info['distance'] / 20.0, 0.0, 2.0),
            np.clip(wp_info['angle'] / np.pi, -1.0, 1.0),
            np.clip(wp_info['lane_offset'] / 5.0, 0.0, 2.0),
            wp_info['progress']
        ]
        
        # Construct observation (normalized)
        obs = np.array([
            # Vehicle state (10)
            speed_kmh / 100.0,
            heading_error / np.pi,
            np.clip(lane_offset / 5.0, -1.0, 1.0),
            np.clip(dist_to_dest / 100.0, 0.0, 5.0),
            collision_flag,
            lane_invasion_flag,
            np.clip(transform.rotation.pitch / 90.0, -1.0, 1.0),
            np.clip(transform.rotation.roll / 90.0, -1.0, 1.0),
            np.clip(velocity.x / 30.0, -1.0, 1.0),
            np.clip(velocity.y / 30.0, -1.0, 1.0),
            
            # Rule state (5)
            rule_state['speed_limit'] / 100.0,
            float(rule_state['must_stop']),
            float(rule_state['no_entry']),
            float(rule_state['traffic_light'] == 'red'),
            rule_state['confidence'],
            
            # ✅ LIDAR state (6)
            *lidar_features,
            
            # ✅ Waypoint state (4)
            *waypoint_features
        ], dtype=np.float32)
        
        self._last_obs_cache = obs
        self._cache_valid = True
        
        return obs
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute action and return observation"""
        self._cache_valid = False
        
        # Apply control
        control = carla.VehicleControl()
        control.steer = float(np.clip(action[0], -1.0, 1.0))
        control.throttle = float(np.clip(action[1], 0.0, 1.0))
        control.brake = float(np.clip(action[2], 0.0, 1.0))
        control.hand_brake = False
        control.manual_gear_shift = False
        
        self.vehicle.apply_control(control)
        
        # Tick simulation
        self.world.tick()
        self.step_count += 1
        
        # Get observation FIRST
        obs = self._get_observation()
        
        # Update collision counter
        if len(self.collision_hist) > 0:
            self.episode_metrics['collisions'] += len(self.collision_hist)
            self.collision_hist.clear()
        
        # Update lane invasion counter
        if len(self.lane_invasion_hist) > 0:
            self.episode_metrics['lane_invasions'] += len(self.lane_invasion_hist)
            self.lane_invasion_hist.clear()
        
        # Calculate reward
        reward = self._compute_reward(action, obs)
        
        # Check termination
        terminated = self._check_termination(obs)
        truncated = self.step_count >= self.config['rl']['max_steps']
        
        # Update metrics
        self.episode_metrics['steps'] += 1
        self.episode_metrics['episode_reward'] += reward
        
        # Calculate speed and distance for metrics
        velocity = self.vehicle.get_velocity()
        speed_ms = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        speed_kmh = 3.6 * speed_ms
        self.episode_metrics['distance_traveled'] += speed_ms * 0.05
        
        # Logging
        if self.step_count % 200 == 0:
            logger.info(
                f"Step {self.step_count}: "
                f"speed={speed_kmh:.1f} km/h, "
                f"dist={self.episode_metrics['distance_traveled']:.1f}m, "
                f"waypoints={self.episode_metrics['waypoints_reached']}/{len(self.route_waypoints)}, "
                f"reward={reward:.2f}"
            )
        
        info = self.episode_metrics.copy()
        
        return obs, reward, terminated, truncated, info
    
    def _compute_reward(self, action: np.ndarray, obs: np.ndarray) -> float:
        """✅ Improved reward with waypoint following and collision prediction"""
        reward = 0.0
        
        # Get vehicle state
        vehicle_location = self.vehicle.get_location()
        velocity = self.vehicle.get_velocity()
        
        speed_ms = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        speed_kmh = 3.6 * speed_ms
        
        # Get LIDAR data
        with self.lidar_lock:
            lidar = self.lidar_data
        
        # ✅ 1. COLLISION PENALTY (terminal)
        if len(self.collision_hist) > 0:
            reward -= 200.0
            return float(np.clip(reward, -200.0, 100.0))
        
        # ✅ 2. PREDICTIVE COLLISION PENALTY (approaching obstacle too fast)
        if lidar is not None:
            min_dist = lidar['min_distance_front']
            
            # Calculate time-to-collision
            if speed_ms > 0.1:
                ttc = min_dist / speed_ms  # seconds to collision
            else:
                ttc = 999.0
            
            # Penalize based on danger level
            if min_dist < 3.0:  # Critical: < 3m
                reward -= 20.0
                self.episode_metrics['close_calls'] += 1
            elif min_dist < 5.0 and speed_kmh > 20.0:  # Danger: < 5m and going fast
                reward -= 10.0
            elif min_dist < 10.0 and speed_kmh > 40.0:  # Warning: < 10m and very fast
                reward -= 5.0
            elif ttc < 2.0 and speed_kmh > 10.0:  # Time-to-collision < 2s
                reward -= 3.0
        
        # ✅ 3. WAYPOINT FOLLOWING REWARD (primary objective)
        wp_info = self._get_next_waypoint_info()
        
        # Reward for reaching waypoints
        if self.current_waypoint_idx > 0:
            # Big reward for each waypoint reached
            waypoints_this_step = self.current_waypoint_idx - getattr(self, '_last_wp_idx', 0)
            if waypoints_this_step > 0:
                reward += 15.0 * waypoints_this_step
        self._last_wp_idx = self.current_waypoint_idx
        
        # Reward for moving toward next waypoint
        wp_distance = wp_info['distance']
        wp_angle = abs(wp_info['angle'])
        
        # Reward for being aligned with waypoint direction
        if wp_angle < np.pi / 6:  # Within 30 degrees
            reward += 2.0
        elif wp_angle < np.pi / 3:  # Within 60 degrees
            reward += 1.0
        else:  # Facing wrong way
            reward -= 1.0
        
        # Reward for being close to route
        wp_lane_offset = wp_info['lane_offset']
        if wp_lane_offset < 2.0:
            reward += 1.0
        elif wp_lane_offset > 5.0:
            reward -= 0.5
        
        # ✅ 4. SPEED REWARD (contextual - only when safe)
        if lidar is not None and lidar['min_distance_front'] > 15.0:
            # Safe to go fast - reward optimal speed
            target_speed = 40.0
            speed_diff = abs(speed_kmh - target_speed)
            if speed_diff < 10.0:
                reward += 1.5
            elif speed_kmh > 10.0:  # At least moving
                reward += 0.5
        elif lidar is not None and lidar['min_distance_front'] < 10.0:
            # Obstacle ahead - reward slowing down
            if speed_kmh < 20.0:
                reward += 1.0
            elif speed_kmh > 40.0:
                reward -= 2.0
        
        # Penalty for being too slow when path is clear
        if (lidar is not None and lidar['min_distance_front'] > 20.0 and 
            speed_kmh < 5.0):
            reward -= 2.0
        
        # ✅ 5. LANE KEEPING (gentle)
        lane_offset = abs(obs[2] * 5.0)
        if lane_offset > 3.0:
            reward -= 0.3 * (lane_offset - 3.0)
        
        # ✅ 6. SMOOTH DRIVING (penalize jerky actions)
        if self._last_action is not None:
            steer_change = abs(action[0] - self._last_action[0])
            if steer_change > 0.5:  # Big steering change
                reward -= 0.2
        self._last_action = action.copy()
        
        # ✅ 7. ROUTE PROGRESS BONUS
        if wp_info['progress'] > 0.9:
            reward += 10.0  # Almost at destination
        elif wp_info['progress'] > 0.7:
            reward += 5.0
        elif wp_info['progress'] > 0.5:
            reward += 2.0
        
        # ✅ 8. SURVIVAL BONUS (reward staying alive)
        reward += 0.1
        
        return float(np.clip(reward, -200.0, 100.0))
    
    def _check_termination(self, obs: np.ndarray) -> bool:
        """Check if episode should terminate"""
        # Collision
        if self.episode_metrics['collisions'] > 0:
            logger.info("Episode terminated: Collision")
            return True
        
        # Reached destination
        vehicle_location = self.vehicle.get_location()
        if vehicle_location.distance(self.destination) < 5.0:
            self.episode_metrics['route_completion'] = 1.0
            logger.info("Episode terminated: Destination reached!")
            return True
        
        # Too far off road
        lane_offset = abs(obs[2] * 5.0)
        if lane_offset > 10.0:
            self.episode_metrics['off_road_steps'] += 1
            if self.episode_metrics['off_road_steps'] > 50:
                logger.info("Episode terminated: Off-road too long")
                return True
        else:
            self.episode_metrics['off_road_steps'] = 0
        
        # Stuck (not moving for too long)
        velocity = self.vehicle.get_velocity()
        speed_kmh = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        if speed_kmh < 1.0:
            if not hasattr(self, '_stuck_counter'):
                self._stuck_counter = 0
            self._stuck_counter += 1
            if self._stuck_counter > 100:  # Stuck for 5 seconds
                logger.info("Episode terminated: Vehicle stuck")
                return True
        else:
            self._stuck_counter = 0
        
        return False
    
    def _cleanup(self):
        """Clean up actors and sensors"""
        # Stop perception thread
        if self.perception_thread and self.perception_thread.is_alive():
            self.stop_perception.set()
            self.perception_thread.join(timeout=2.0)
            self.stop_perception.clear()
            self.perception_thread = None
        
        # Clear queue
        while not self.perception_queue.empty():
            try:
                self.perception_queue.get_nowait()
            except queue.Empty:
                break
        
        # Destroy sensors
        for sensor in list(self.sensors.values()):
            if sensor is not None and sensor.is_alive:
                try:
                    sensor.stop()
                    sensor.destroy()
                except:
                    pass
        self.sensors.clear()
        
        # Destroy vehicle
        if self.vehicle is not None and self.vehicle.is_alive:
            try:
                self.vehicle.destroy()
            except:
                pass
            self.vehicle = None
        
        # Destroy traffic vehicles
        for vehicle in self.traffic_vehicles:
            if vehicle.is_alive:
                try:
                    vehicle.destroy()
                except:
                    pass
        self.traffic_vehicles.clear()
        
        # Clear histories
        self.collision_hist.clear()
        self.lane_invasion_hist.clear()
        
        # Reset LIDAR data
        with self.lidar_lock:
            self.lidar_data = None
        
        # Tick to process destruction
        try:
            self.world.tick()
        except:
            pass
    
    def close(self):
        """Close the environment"""
        logger.info("Closing CARLA environment")
        self._cleanup()
        
        try:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
        except:
            pass
    
    def __del__(self):
        try:
            self.close()
        except:
            pass
