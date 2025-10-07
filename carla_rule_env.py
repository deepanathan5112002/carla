import numpy as np
import carla
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
    """Optimized CARLA environment with traffic rule awareness"""
    
    metadata = {'render_modes': ['rgb_array']}
    
    def __init__(self, config_path: str, port: int = 2000):
        super().__init__()
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
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
        
        # Set synchronous mode with optimized settings
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 FPS
        settings.no_rendering_mode = False  # Keep rendering for stability
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
        self.collision_hist = deque(maxlen=100)
        self.lane_invasion_hist = deque(maxlen=100)
        
        # Perception queue (thread-safe)
        self.perception_queue = queue.Queue(maxsize=2)
        self.last_rule_state = self._default_rule_state()
        self.perception_thread = None
        self.stop_perception = threading.Event()
        
        # Route planning
        self.map = self.world.get_map()
        self.spawn_points = self.map.get_spawn_points()
        self.destination = None
        self.last_dist_to_dest = None
        self.route_waypoints = []
        
        # Metrics tracking
        self.episode_metrics = self._reset_metrics()
        self.step_count = 0
        
        # Perception module
        self.detector = None
        self.use_perception = False
        
        # Performance optimization
        self._last_obs_cache = None
        self._cache_valid = False
    
    def _default_rule_state(self) -> Dict:
        """Default rule state when no signs detected"""
        return {
            'speed_limit': 50,
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
            'steps': 0,
            'off_road_steps': 0,
            'max_speed': 0.0
        }
    
    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)
        
        # Clean up previous episode
        self._cleanup()
        
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
        
        # Attach sensors
        self._setup_sensors()
        
        # Setup route
        self._setup_route()
        
        # Reset metrics
        self.episode_metrics = self._reset_metrics()
        self.last_dist_to_dest = None
        self.step_count = 0
        self._cache_valid = False
        
        # Start perception thread if enabled
        if self.use_perception and self.detector and not self.perception_thread:
            self._start_perception_thread()
        
        # Tick world to initialize sensors
        self.world.tick()
        time.sleep(0.1)
        
        obs = self._get_observation()
        info = {}
        
        return obs, info
    
    def _choose_spawn_point(self) -> carla.Transform:
        """Choose a good spawn point (avoid junctions and busy areas)"""
        # Filter out junction spawns for stability
        good_spawns = []
        for sp in self.spawn_points:
            waypoint = self.map.get_waypoint(sp.location)
            if not waypoint.is_junction:
                good_spawns.append(sp)
        
        if not good_spawns:
            good_spawns = self.spawn_points
        
        return np.random.choice(good_spawns)
    
    def _setup_sensors(self):
        """Setup all sensors with optimized settings"""
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
            # Only process if perception is enabled
            if not self.use_perception:
                return
            
            # Subsample frames (process every Nth frame)
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
        """Setup route with better destination selection"""
        start = self.vehicle.get_location()
        
        # Choose destination far enough away (50-200m)
        valid_destinations = [
            sp for sp in self.spawn_points 
            if 50.0 < start.distance(sp.location) < 200.0
        ]
        
        if not valid_destinations:
            valid_destinations = [
                sp for sp in self.spawn_points 
                if start.distance(sp.location) > 30.0
            ]
        
        if valid_destinations:
            destination_transform = np.random.choice(valid_destinations)
        else:
            destination_transform = np.random.choice(self.spawn_points)
        
        self.destination = destination_transform.location
        self.route_waypoints = []  # Could use GlobalRoutePlanner here
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation vector with caching"""
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
        
        # Construct observation (normalized)
        obs = np.array([
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
            rule_state['speed_limit'] / 100.0,
            float(rule_state['must_stop']),
            float(rule_state['no_entry']),
            float(rule_state['traffic_light'] == 'red'),
            rule_state['confidence']
        ], dtype=np.float32)
        
        self._last_obs_cache = obs
        self._cache_valid = True
        
        return obs
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute action and return observation"""
        self._cache_valid = False
        
        # Apply control with smoothing
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
        
        # Get observation
        obs = self._get_observation()
        
        # Calculate reward
        reward = self._compute_reward(action, obs)
        
        # Check termination
        terminated = self._check_termination(obs)
        truncated = self.step_count >= self.config['rl']['max_steps']
        
        # Update metrics
        self.episode_metrics['steps'] += 1
        self.episode_metrics['episode_reward'] += reward
        
        velocity = self.vehicle.get_velocity()
        speed_ms = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        self.episode_metrics['distance_traveled'] += speed_ms * 0.05
        
        info = self.episode_metrics.copy()
        
        return obs, reward, terminated, truncated, info
    
    def _compute_reward(self, action: np.ndarray, obs: np.ndarray) -> float:
        """Optimized reward computation"""
        reward = 0.0
        reward_config = self.config['rewards']
        
        vehicle_location = self.vehicle.get_location()
        velocity = self.vehicle.get_velocity()
        speed_ms = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        speed_kmh = 3.6 * speed_ms
        
        # 1. Progress reward (shaped)
        dist_to_dest = vehicle_location.distance(self.destination)
        if self.last_dist_to_dest is not None:
            progress = self.last_dist_to_dest - dist_to_dest
            if progress > 0:
                reward += reward_config['progress'] * progress
        self.last_dist_to_dest = dist_to_dest
        
        # 2. Lane keeping (quadratic penalty for smoothness)
        lane_offset = obs[2] * 5.0  # Denormalize
        reward -= reward_config['lane_keeping'] * (lane_offset ** 2)
        
        # Track off-road
        if abs(lane_offset) > 2.0:
            self.episode_metrics['off_road_steps'] += 1
        
        # 3. Speed compliance (only penalize significant violations)
        speed_limit = self.last_rule_state['speed_limit']
        if speed_kmh > speed_limit + 10:  # 10 km/h tolerance
            excess = speed_kmh - speed_limit
            reward -= reward_config['speed_compliance'] * (excess / 10.0)
            self.episode_metrics['speed_violations'] += 1
        
        # 4. Stop compliance
        if self.last_rule_state['must_stop'] and speed_kmh > 3.0:
            reward -= reward_config['stop_compliance']
            self.episode_metrics['stop_violations'] += 1
        
        # 5. Collision penalty (terminal)
        if len(self.collision_hist) > 0:
            reward += reward_config['collision']
            self.episode_metrics['collisions'] += len(self.collision_hist)
            self.collision_hist.clear()
        
        # 6. Off-road penalty (smoothed)
        waypoint = self.map.get_waypoint(vehicle_location)
        if not waypoint.is_junction and abs(lane_offset) > 3.0:
            reward += reward_config['off_road'] * 0.1  # Soft penalty per step
        
        # 7. Lane invasion penalty
        if len(self.lane_invasion_hist) > 0:
            reward -= 5.0  # Reduced from 10
            self.episode_metrics['lane_invasions'] += len(self.lane_invasion_hist)
            self.lane_invasion_hist.clear()
        
        # 8. Smooth control bonus (reduce jerkiness)
        if hasattr(self, '_last_action'):
            action_diff = np.abs(action - self._last_action)
            smoothness_penalty = -0.1 * np.sum(action_diff)
            reward += smoothness_penalty
        self._last_action = action.copy()
        
        # 9. Forward progress bonus (encourage moving)
        if self.step_count > 100:  # After initial settling
            if speed_kmh < 5.0:  # Too slow
                reward -= 2.0  # Strong penalty for not moving
            elif speed_kmh < 15.0:  # Moving but too cautious
                reward -= 0.5
    
        # 10. Bonus for maintaining good speed
        if 20.0 < speed_kmh < speed_limit:
            reward += 0.5
        return float(np.clip(reward, -100.0, 10.0))
    
    def _check_termination(self, obs: np.ndarray) -> bool:
        """Check if episode should terminate"""
        # Collision termination
        if self.episode_metrics['collisions'] > 0:
            return True
        
        # Reached destination
        vehicle_location = self.vehicle.get_location()
        if vehicle_location.distance(self.destination) < 5.0:
            self.episode_metrics['route_completion'] = 1.0
            return True
        
        # Off-road termination (more lenient)
        lane_offset = obs[2] * 5.0
        if abs(lane_offset) > 7.0:  # ✅ INCREASED from 6.0 - more forgiving
            return True
        
        # ✅ NEW: Terminate if stuck (not moving for too long)
        if self.step_count > 500:  # After 500 steps
            if self.episode_metrics['distance_traveled'] < 20.0:  # Moved less than 20m
                logger.debug("Episode terminated: stuck/not moving")
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
        
        # Clear histories
        self.collision_hist.clear()
        self.lane_invasion_hist.clear()
        
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
