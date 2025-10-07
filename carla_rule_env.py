import numpy as np
import carla
import cv2
import gymnasium as gym
from gymnasium import spaces
import yaml
import time
from collections import deque
import threading
import queue
from typing import Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CarlaRuleAwareEnv(gym.Env):
    """CARLA environment with traffic rule awareness via CV perception"""
    
    metadata = {'render_modes': ['rgb_array']}
    
    def __init__(self, config_path: str, port: int = 2000):
        super().__init__()
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Connect to CARLA
        logger.info(f"Connecting to CARLA on port {port}...")
        self.client = carla.Client('localhost', port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        
        # Set synchronous mode
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 FPS
        self.world.apply_settings(settings)
        logger.info(f"Connected to map: {self.world.get_map().name}")
        
        # Action space: [steer, throttle, brake]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # Observation space: [vehicle_state(10) + rule_state(5)]
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(15,), 
            dtype=np.float32
        )
        
        # Initialize components
        self.vehicle = None
        self.sensors = {}
        self.collision_hist = []
        self.lane_invasion_hist = []
        
        # Perception queue and state
        self.perception_queue = queue.Queue(maxsize=2)
        self.last_rule_state = self._default_rule_state()
        self.perception_thread = None
        self.stop_perception = threading.Event()
        
        # Route planning
        self.map = self.world.get_map()
        self.spawn_points = self.map.get_spawn_points()
        self.destination = None
        self.last_dist_to_dest = None
        
        # Metrics tracking
        self.episode_metrics = self._reset_metrics()
        
        # Perception module (will be set by ShieldedCarlaEnv)
        self.detector = None
        self.use_perception = False
    
    def _default_rule_state(self) -> Dict:
        """Default rule state when no signs detected"""
        return {
            'speed_limit': 50,  # km/h
            'must_stop': False,
            'no_entry': False,
            'traffic_light': 'green',
            'confidence': 0.0
        }
    
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
            'steps': 0
        }
    
    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)
        
        # Clean up previous episode
        self._cleanup()
        
        # Spawn vehicle at random location
        spawn_point = np.random.choice(self.spawn_points)
        bp_library = self.world.get_blueprint_library()
        vehicle_bp = bp_library.find('vehicle.tesla.model3')
        
        try:
            self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        except RuntimeError as e:
            logger.warning(f"Spawn failed: {e}. Retrying with different point...")
            spawn_point = np.random.choice(self.spawn_points)
            self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        
        # Attach sensors
        self._setup_sensors()
        
        # Setup route
        self._setup_route()
        
        # Reset metrics
        self.episode_metrics = self._reset_metrics()
        self.last_dist_to_dest = None
        
        # Start perception thread if perception is enabled
        if self.use_perception and self.detector and not self.perception_thread:
            self._start_perception_thread()
        
        # Tick world to initialize
        self.world.tick()
        time.sleep(0.1)  # Small delay for sensors to initialize
        
        # Get initial observation
        obs = self._get_observation()
        info = {}
        
        return obs, info
    
    def _setup_sensors(self):
        """Setup all sensors"""
        bp_library = self.world.get_blueprint_library()
        
        # RGB Camera for perception
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
    
    def _process_camera(self, image):
        """Process camera image - runs in CARLA callback thread"""
        try:
            # Convert to numpy array
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))
            array = array[:, :, :3]  # Remove alpha channel (RGB only)
            
            # Queue for perception thread (non-blocking)
            if not self.perception_queue.full():
                try:
                    self.perception_queue.put_nowait(array)
                except queue.Full:
                    pass  # Skip frame if queue full
        except Exception as e:
            logger.error(f"Camera processing error: {e}")
    
    def _start_perception_thread(self):
        """Start background perception processing thread"""
        if not self.use_perception or not self.detector:
            return
        
        def perception_worker():
            """Background thread for running perception inference"""
            logger.info("Perception thread started")
            while not self.stop_perception.is_set():
                try:
                    # Get frame from queue (blocking with timeout)
                    frame = self.perception_queue.get(timeout=0.5)
                    
                    # Run inference
                    rule_state = self.detector.infer(frame)
                    
                    # Update last rule state (thread-safe for simple dict assignment)
                    self.last_rule_state = rule_state
                    
                except queue.Empty:
                    continue  # No frame available, keep waiting
                except Exception as e:
                    logger.error(f"Perception inference error: {e}")
            
            logger.info("Perception thread stopped")
        
        self.perception_thread = threading.Thread(target=perception_worker, daemon=True)
        self.perception_thread.start()
    
    def _setup_route(self):
        """Setup a random route"""
        start = self.vehicle.get_location()
        
        # Pick a random destination far enough away
        valid_destinations = [sp for sp in self.spawn_points 
                            if start.distance(sp.location) > 50.0]
        
        if valid_destinations:
            destination_transform = np.random.choice(valid_destinations)
        else:
            destination_transform = np.random.choice(self.spawn_points)
        
        self.destination = destination_transform.location
        logger.debug(f"Route set: distance to destination = {start.distance(self.destination):.1f}m")
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation vector"""
        # Vehicle transform and velocity
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        speed_ms = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        speed_kmh = 3.6 * speed_ms
        
        # Get current waypoint
        vehicle_location = transform.location
        waypoint = self.map.get_waypoint(vehicle_location)
        
        # Calculate heading error (angle to waypoint)
        waypoint_yaw = waypoint.transform.rotation.yaw
        vehicle_yaw = transform.rotation.yaw
        heading_error = np.deg2rad(waypoint_yaw - vehicle_yaw)
        # Normalize to [-pi, pi]
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
        
        # Lane offset (lateral distance from lane center)
        waypoint_loc = waypoint.transform.location
        lane_offset = vehicle_location.distance(waypoint_loc)
        
        # Distance to destination
        dist_to_dest = vehicle_location.distance(self.destination)
        
        # Collision and lane invasion flags
        collision_flag = 1.0 if len(self.collision_hist) > 0 else 0.0
        lane_invasion_flag = 1.0 if len(self.lane_invasion_hist) > 0 else 0.0
        
        # Get rule state (latest from perception)
        rule_state = self.last_rule_state
        
        # Construct observation vector (15 dimensions)
        obs = np.array([
            # Vehicle state (10 features)
            speed_kmh / 100.0,                      # [0] Normalized speed
            heading_error / np.pi,                  # [1] Normalized heading error
            lane_offset / 5.0,                      # [2] Normalized lane offset
            dist_to_dest / 100.0,                   # [3] Normalized distance to goal
            collision_flag,                          # [4] Collision flag
            lane_invasion_flag,                      # [5] Lane invasion flag
            transform.rotation.pitch / 90.0,        # [6] Normalized pitch
            transform.rotation.roll / 90.0,         # [7] Normalized roll
            velocity.x / 30.0,                      # [8] Normalized vx
            velocity.y / 30.0,                      # [9] Normalized vy
            
            # Rule state (5 features)
            rule_state['speed_limit'] / 100.0,      # [10] Normalized speed limit
            float(rule_state['must_stop']),         # [11] Must stop flag
            float(rule_state['no_entry']),          # [12] No entry flag
            float(rule_state['traffic_light'] == 'red'),  # [13] Red light flag
            rule_state['confidence']                # [14] Detection confidence
        ], dtype=np.float32)
        
        return obs
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute action and return (obs, reward, terminated, truncated, info)"""
        # Apply control action
        control = carla.VehicleControl()
        control.steer = float(np.clip(action[0], -1.0, 1.0))
        control.throttle = float(np.clip(action[1], 0.0, 1.0))
        control.brake = float(np.clip(action[2], 0.0, 1.0))
        self.vehicle.apply_control(control)
        
        # Tick simulation (synchronous mode)
        self.world.tick()
        
        # Get new observation
        obs = self._get_observation()
        
        # Calculate reward
        reward = self._compute_reward(action)
        
        # Check termination conditions
        terminated = self._check_termination()
        truncated = self.episode_metrics['steps'] >= self.config['rl']['max_steps']
        
        # Update metrics
        self.episode_metrics['steps'] += 1
        self.episode_metrics['episode_reward'] += reward
        
        # Calculate distance traveled
        velocity = self.vehicle.get_velocity()
        speed_ms = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        self.episode_metrics['distance_traveled'] += speed_ms * 0.05  # dt = 0.05s
        
        info = self.episode_metrics.copy()
        
        return obs, reward, terminated, truncated, info
    
    def _compute_reward(self, action: np.ndarray) -> float:
        """Compute reward based on current state and action"""
        reward = 0.0
        reward_config = self.config['rewards']
        
        vehicle_location = self.vehicle.get_location()
        velocity = self.vehicle.get_velocity()
        speed_ms = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        speed_kmh = 3.6 * speed_ms
        
        # 1. Progress reward (encourage moving toward destination)
        dist_to_dest = vehicle_location.distance(self.destination)
        if self.last_dist_to_dest is not None:
            progress = self.last_dist_to_dest - dist_to_dest
            reward += reward_config['progress'] * max(0, progress)
        self.last_dist_to_dest = dist_to_dest
        
        # 2. Lane keeping penalty
        waypoint = self.map.get_waypoint(vehicle_location)
        lane_offset = vehicle_location.distance(waypoint.transform.location)
        reward -= reward_config['lane_keeping'] * lane_offset
        
        # 3. Speed compliance (penalize exceeding speed limit)
        speed_limit = self.last_rule_state['speed_limit']
        if speed_kmh > speed_limit:
            speed_violation = (speed_kmh - speed_limit) / speed_limit
            reward -= reward_config['speed_compliance'] * speed_violation
            self.episode_metrics['speed_violations'] += 1
        
        # 4. Stop compliance (must stop at stop signs / red lights)
        if self.last_rule_state['must_stop'] and speed_kmh > 5.0:
            reward -= reward_config['stop_compliance']
            self.episode_metrics['stop_violations'] += 1
        
        # 5. Collision penalty (terminal)
        if len(self.collision_hist) > 0:
            reward += reward_config['collision']  # Large negative
            self.episode_metrics['collisions'] += len(self.collision_hist)
            self.collision_hist.clear()
        
        # 6. Off-road penalty
        if not waypoint.is_junction and lane_offset > 3.0:
            reward += reward_config['off_road']  # Negative
        
        # 7. Lane invasion penalty
        if len(self.lane_invasion_hist) > 0:
            reward -= 10.0
            self.episode_metrics['lane_invasions'] += len(self.lane_invasion_hist)
            self.lane_invasion_hist.clear()
        
        return reward
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate"""
        # Collision termination
        if self.episode_metrics['collisions'] > 0:
            logger.debug("Episode terminated: collision")
            return True
        
        # Reached destination
        vehicle_location = self.vehicle.get_location()
        dist = vehicle_location.distance(self.destination)
        if dist < 5.0:
            self.episode_metrics['route_completion'] = 1.0
            logger.debug("Episode terminated: destination reached")
            return True
        
        # Off-road termination (too far from lane)
        waypoint = self.map.get_waypoint(vehicle_location)
        lane_offset = vehicle_location.distance(waypoint.transform.location)
        if lane_offset > 5.0:
            logger.debug("Episode terminated: off-road")
            return True
        
        return False
    
    def _cleanup(self):
        """Clean up actors and sensors"""
        # Stop perception thread
        if self.perception_thread and self.perception_thread.is_alive():
            self.stop_perception.set()
            self.perception_thread.join(timeout=2.0)
            self.stop_perception.clear()
            self.perception_thread = None
        
        # Clear perception queue
        while not self.perception_queue.empty():
            try:
                self.perception_queue.get_nowait()
            except queue.Empty:
                break
        
        # Destroy sensors
        for sensor_name, sensor in self.sensors.items():
            if sensor is not None and sensor.is_alive:
                try:
                    sensor.stop()
                    sensor.destroy()
                except Exception as e:
                    logger.warning(f"Error destroying sensor {sensor_name}: {e}")
        self.sensors.clear()
        
        # Destroy vehicle
        if self.vehicle is not None and self.vehicle.is_alive:
            try:
                self.vehicle.destroy()
            except Exception as e:
                logger.warning(f"Error destroying vehicle: {e}")
            self.vehicle = None
        
        # Clear histories
        self.collision_hist.clear()
        self.lane_invasion_hist.clear()
        
        # Tick world to process destruction
        self.world.tick()
    
    def close(self):
        """Close the environment"""
        logger.info("Closing CARLA environment")
        self._cleanup()
        
        # Reset to asynchronous mode
        try:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
        except Exception as e:
            logger.warning(f"Error resetting synchronous mode: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            self.close()
        except:
            pass


