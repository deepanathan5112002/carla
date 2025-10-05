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
        self.client = carla.Client('localhost', port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        
        # Set synchronous mode
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 FPS
        self.world.apply_settings(settings)
        
        # Action and observation spaces
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0]),  # steer, throttle, brake
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # State: [vehicle_state(10), rule_state(5)]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32
        )
        
        # Initialize components
        self.vehicle = None
        self.sensors = {}
        self.collision_hist = []
        self.lane_invasion_hist = []
        self.perception_queue = queue.Queue(maxsize=1)
        self.last_rule_state = self._default_rule_state()
        
        # Route planning
        self.map = self.world.get_map()
        self.spawn_points = self.map.get_spawn_points()
        self.route_planner = None
        self.current_waypoint = None
        self.destination = None
        
        # Metrics tracking
        self.episode_metrics = self._reset_metrics()
        
        # Start perception thread
        self.perception_thread = None
        self.stop_perception = threading.Event()
        
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
        
        # Spawn vehicle
        spawn_point = np.random.choice(self.spawn_points)
        bp_library = self.world.get_blueprint_library()
        vehicle_bp = bp_library.find('vehicle.tesla.model3')
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        
        # Attach sensors
        self._setup_sensors()
        
        # Setup route
        self._setup_route()
        
        # Reset metrics
        self.episode_metrics = self._reset_metrics()
        
        # Get initial observation
        self.world.tick()
        obs = self._get_observation()
        info = {}
        
        return obs, info
    
    def _setup_sensors(self):
        """Setup all sensors"""
        bp_library = self.world.get_blueprint_library()
        
        # RGB Camera
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
        """Process camera image for perception"""
        # Convert to numpy array
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        array = array[:, :, :3]  # Remove alpha channel
        
        # Queue for perception thread
        if not self.perception_queue.full():
            self.perception_queue.put(array)
    
    def _setup_route(self):
        """Setup a random route"""
        start = self.vehicle.get_location()
        
        # Pick a random destination
        spawn_points = self.map.get_spawn_points()
        destination_transform = np.random.choice(spawn_points)
        self.destination = destination_transform.location
        
        # Simple waypoint following (can be enhanced with GlobalRoutePlanner)
        self.current_waypoint = self.map.get_waypoint(start)
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation vector"""
        # Vehicle state
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        
        # Get waypoint info
        vehicle_location = transform.location
        waypoint = self.map.get_waypoint(vehicle_location)
        
        # Calculate heading error
        waypoint_yaw = waypoint.transform.rotation.yaw
        vehicle_yaw = transform.rotation.yaw
        heading_error = np.deg2rad(waypoint_yaw - vehicle_yaw)
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
        
        # Lane offset
        waypoint_loc = waypoint.transform.location
        lane_offset = vehicle_location.distance(waypoint_loc)
        
        # Distance to destination
        dist_to_dest = vehicle_location.distance(self.destination)
        
        # Collision and lane invasion flags
        collision_flag = 1.0 if len(self.collision_hist) > 0 else 0.0
        lane_invasion_flag = 1.0 if len(self.lane_invasion_hist) > 0 else 0.0
        
        # Get rule state
        if not self.perception_queue.empty():
            try:
                self.last_rule_state = self.perception_queue.get_nowait()
            except:
                pass
        
        rule_state = self.last_rule_state
        
        # Construct observation vector
        obs = np.array([
            speed / 30.0,  # Normalized speed
            heading_error,
            lane_offset / 5.0,  # Normalized lane offset
            dist_to_dest / 100.0,  # Normalized distance
            collision_flag,
            lane_invasion_flag,
            transform.rotation.pitch / 180.0,
            transform.rotation.roll / 180.0,
            velocity.x / 30.0,
            velocity.y / 30.0,
            # Rule state
            rule_state['speed_limit'] / 100.0,
            float(rule_state['must_stop']),
            float(rule_state['no_entry']),
            float(rule_state['traffic_light'] == 'red'),
            rule_state['confidence']
        ], dtype=np.float32)
        
        return obs
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute action and return observation"""
        # Apply action
        control = carla.VehicleControl()
        control.steer = float(action[0])
        control.throttle = float(action[1])
        control.brake = float(action[2])
        self.vehicle.apply_control(control)
        
        # Tick simulation
        self.world.tick()
        
        # Get new observation
        obs = self._get_observation()
        
        # Calculate reward
        reward = self._compute_reward(action)
        
        # Check termination
        terminated = self._check_termination()
        truncated = self.episode_metrics['steps'] >= self.config['rl']['max_steps']
        
        # Update metrics
        self.episode_metrics['steps'] += 1
        self.episode_metrics['episode_reward'] += reward
        
        info = self.episode_metrics.copy()
        
        return obs, reward, terminated, truncated, info
    
    def _compute_reward(self, action: np.ndarray) -> float:
        """Compute reward based on current state"""
        reward = 0.0
        reward_config = self.config['rewards']
        
        # Progress reward
        vehicle_location = self.vehicle.get_location()
        dist_to_dest = vehicle_location.distance(self.destination)
        if hasattr(self, 'last_dist_to_dest'):
            progress = self.last_dist_to_dest - dist_to_dest
            reward += reward_config['progress'] * max(0, progress)
        self.last_dist_to_dest = dist_to_dest
        
        # Lane keeping
        waypoint = self.map.get_waypoint(vehicle_location)
        lane_offset = vehicle_location.distance(waypoint.transform.location)
        reward -= reward_config['lane_keeping'] * lane_offset
        
        # Speed compliance
        velocity = self.vehicle.get_velocity()
        speed_kmh = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        speed_limit = self.last_rule_state['speed_limit']
        
        if speed_kmh > speed_limit:
            speed_violation = (speed_kmh - speed_limit) / speed_limit
            reward -= reward_config['speed_compliance'] * speed_violation
            self.episode_metrics['speed_violations'] += 1
        
        # Stop compliance
        if self.last_rule_state['must_stop'] and speed_kmh > 5:
            reward -= reward_config['stop_compliance']
            self.episode_metrics['stop_violations'] += 1
        
        # Collision penalty
        if len(self.collision_hist) > 0:
            reward += reward_config['collision']
            self.episode_metrics['collisions'] += len(self.collision_hist)
            self.collision_hist.clear()
        
        # Off-road penalty
        if not waypoint.is_junction and lane_offset > 3.0:
            reward += reward_config['off_road']
        
        # Lane invasion penalty
        if len(self.lane_invasion_hist) > 0:
            reward -= 10.0
            self.episode_metrics['lane_invasions'] += len(self.lane_invasion_hist)
            self.lane_invasion_hist.clear()
        
        return reward
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate"""
        # Collision
        if self.episode_metrics['collisions'] > 0:
            return True
        
        # Reached destination
        vehicle_location = self.vehicle.get_location()
        if vehicle_location.distance(self.destination) < 5.0:
            self.episode_metrics['route_completion'] = 1.0
            return True
        
        # Off-road for too long
        waypoint = self.map.get_waypoint(vehicle_location)
        if vehicle_location.distance(waypoint.transform.location) > 5.0:
            return True
        
        return False
    
    def _cleanup(self):
        """Clean up actors and sensors"""
        # Stop perception thread
        if self.perception_thread:
            self.stop_perception.set()
            self.perception_thread.join()
        
        # Destroy sensors
        for sensor in self.sensors.values():
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        self.sensors.clear()
        
        # Destroy vehicle
        if self.vehicle is not None:
            self.vehicle.destroy()
            self.vehicle = None
        
        # Clear histories
        self.collision_hist.clear()
        self.lane_invasion_hist.clear()
    
    def close(self):
        """Close the environment"""
        self._cleanup()
        
        # Reset to asynchronous mode
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)