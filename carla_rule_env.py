# carla_rule_env.py
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np

try:
    import carla  # CARLA PythonAPI must be on PYTHONPATH
except Exception as e:
    raise RuntimeError(
        "CARLA PythonAPI not found. Add CARLA .egg to PYTHONPATH before importing this env."
    ) from e


# =========================
# Config / Logging
# =========================
@dataclass
class EnvConfig:
    host: str = "127.0.0.1"
    port: int = 2000
    town: Optional[str] = None          # e.g. "Town03"; None = keep current map
    fps: int = 20                       # sync fixed-dt Hz
    max_steps: int = 2000
    offroad_threshold_m: float = 4.0    # lateral offset to terminate
    spawn_index: int = 0
    warmup_ticks: int = 3
    seed: int = 42
    no_rendering_mode: Optional[bool] = None  # leave None to keep current


def _logger(name="carla_rule_env"):
    log = logging.getLogger(name)
    if not log.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
        log.addHandler(h)
        log.setLevel(logging.INFO)
    return log


# =========================
# Environment
# =========================
class CarlaRuleEnv(gym.Env):
    """
    CARLA env with 3D action: [ steer, accel, brake ] in [-1,1].
      - steer in [-1,1]
      - accel, brake in [-1,1] mapped to [0,1]
      - Dominance rule: if brake > accel + eps -> brake only;
                        if accel > brake + eps -> throttle only;
                        else neither.

    Observation (minimal example):
      [ speed_kmh/50, lateral_offset/5, yaw_err/90deg, throttle, brake, steer ]
    """
    metadata = {"render_modes": []}

    def __init__(self, cfg: EnvConfig, shield: Optional[object] = None):
        super().__init__()
        self.cfg = cfg
        self._log = _logger("carla_rule_env")
        self._shield = shield

        # ----- Connect client/world -----
        self.client = carla.Client(cfg.host, cfg.port)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()

        if cfg.town and self.world.get_map().name.split("/")[-1] != cfg.town:
            self._log.info(f"Loading map {cfg.town} ...")
            self.world = self.client.load_world(cfg.town)

        self.map = self.world.get_map()
        self._log.info(f"Connected to map: {self.map.name}")

        # ----- Sync settings (once) -----
        self._orig_settings = self.world.get_settings()
        s = self.world.get_settings()
        s.synchronous_mode = True
        s.fixed_delta_seconds = 1.0 / float(cfg.fps)
        s.substepping = True
        if cfg.no_rendering_mode is not None:
            s.no_rendering_mode = bool(cfg.no_rendering_mode)
        self.world.apply_settings(s)
        self._fps = cfg.fps

        # RNG
        self.np_random, _ = gym.utils.seeding.np_random(cfg.seed)

        # Actors / state
        self.vehicle: Optional[carla.Vehicle] = None
        self.sensors = []
        self._episode_steps = 0
        self._stuck_steps = 0
        self._metrics = {"shield_interventions": 0}

        # ----- Spaces -----
        # Actions: steer, accel, brake in [-1, 1]
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([ 1.0,  1.0,  1.0], dtype=np.float32),
            dtype=np.float32,
        )
        # Normalized obs
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, -1.0, -1.0, 0.0, 0.0, -1.0], dtype=np.float32),
            high=np.array([1.0,  1.0,  1.0, 1.0, 1.0,  1.0], dtype=np.float32),
            dtype=np.float32,
        )

    # -------------- Gym API --------------
    def reset(self, *, seed: Optional[int] = None, options=None):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        self._destroy_actors()
        self._episode_steps = 0
        self._stuck_steps = 0
        self._metrics = {"shield_interventions": 0}

        self.vehicle = self._spawn_vehicle(self.cfg.spawn_index)

        # sane control state (no handbrake, auto gear)
        ctrl = carla.VehicleControl(throttle=0.0, brake=0.0,
                                    hand_brake=False, manual_gear_shift=False, reverse=False)
        self.vehicle.apply_control(ctrl)

        # warm-up ticks for sensor/physics
        for _ in range(self.cfg.warmup_ticks):
            self.world.tick()
        self._log.info(
        f"sync={self.world.get_settings().synchronous_mode} "
        f"dt={self.world.get_settings().fixed_delta_seconds:.3f}s fps={self._fps}"
)


        return self._get_observation(), {"shield_interventions": 0}

    def step(self, action: np.ndarray):
        self._episode_steps += 1

        # Map 3D action -> steer, throttle, brake
        steer, throttle, brake = self._map_action_3d(action)

        speed_kmh = self._speed_kmh()

        # Optional shield (even if rule confidence is 0)
        if self._shield and hasattr(self._shield, "guard"):
            s0, t0, b0 = steer, throttle, brake
            steer, throttle, brake = self._shield.guard(steer, throttle, brake, speed_kmh, rules=None)
            if (steer, throttle, brake) != (s0, t0, b0):
                self._metrics["shield_interventions"] += 1

        # Apply control (mutually exclusive already, but guard again)
        vc = carla.VehicleControl()
        vc.steer = float(np.clip(steer, -1.0, 1.0))
        if brake > throttle and brake > 0.05:
            vc.throttle, vc.brake = 0.0, float(np.clip(brake, 0.0, 1.0))
        elif throttle > 0.05:
            vc.throttle, vc.brake = float(np.clip(throttle, 0.0, 1.0)), 0.0
        else:
            vc.throttle, vc.brake = 0.0, 0.0

        vc.hand_brake = False
        vc.manual_gear_shift = False
        vc.reverse = False
        self.vehicle.apply_control(vc)

        # Single physics step in sync mode
        self.world.tick()

        # Observation / reward / done
        obs = self._get_observation()

        # Stuck tracking & gentle ‘unstuck’ nudge
        speed_kmh = self._speed_kmh()
        if speed_kmh < 0.2:
            self._stuck_steps += 1
        else:
            self._stuck_steps = 0

        if self._stuck_steps > 200:
            kick = carla.VehicleControl(steer=0.0, throttle=0.35, brake=0.0,
                                        hand_brake=False, manual_gear_shift=False)
            self.vehicle.apply_control(kick)
            self.world.tick()
            self._stuck_steps = 0

        offset_m, yaw_err_norm = self._lane_offset_and_yaw_err()

        # Reward shaping (simple, tweak to taste)
        reward = 0.0
        if 5.0 <= speed_kmh <= 25.0:
            reward += 5.0                                   # useful motion
        reward += max(0.0, 1.0 - abs(offset_m) / 3.0)       # lane keeping bonus
        if speed_kmh < 0.5 and abs(steer) > 0.8:
            reward -= 2.0                                   # sawing wheel at standstill
        if abs(offset_m) > 3.5:
            reward -= 10.0 if speed_kmh >= 1.0 else 2.0     # softer if barely moving

        # Termination conditions
        terminated = False
        truncated = False
        if abs(offset_m) > self.cfg.offroad_threshold_m and speed_kmh >= 0.5:
            self._log.info(f"Episode terminated: Off-road (offset={offset_m:.1f}m)")
            terminated = True
        if self._episode_steps >= self.cfg.max_steps:
            truncated = True

        if self._episode_steps % 200 == 0:
        vc = self.vehicle.get_control()
        self._log.info(
            f"t={self._episode_steps} speed={self._speed_kmh():.1f} km/h "
            f"ctrl[steer={vc.steer:.2f}, thr={vc.throttle:.2f}, brk={vc.brake:.2f}] "
            f"stuck={getattr(self,'_stuck_count',0)} "
            f"shield={self._metrics.get('shield_interventions',0)}"
        )


        info = {
            "speed_kmh": speed_kmh,
            "offset_m": offset_m,
            "yaw_err_norm": yaw_err_norm,
            "shield_interventions": self._metrics["shield_interventions"],
        }
        return obs, reward, terminated, truncated, info

    def close(self):
        try:
            self._destroy_actors()
        finally:
            try:
                self.world.apply_settings(self._orig_settings)
            except Exception as e:
                self._log.warning(f"Failed to restore world settings: {e}")

    # -------------- Helpers --------------
    def _map_action_3d(self, a: np.ndarray, eps: float = 0.05) -> Tuple[float, float, float]:
        """
        a = [ steer, accel, brake ] in [-1,1].
        accel_raw/brake_raw mapped to [0,1] via (x+1)/2.
        Dominance: if brake > accel + eps -> brake only; if accel > brake + eps -> throttle only; else neither.
        """
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        steer = float(np.clip(a[0], -1.0, 1.0))
        accel = float((np.clip(a[1], -1.0, 1.0) + 1.0) * 0.5)  # [0,1]
        brk   = float((np.clip(a[2], -1.0, 1.0) + 1.0) * 0.5)  # [0,1]

        if brk > accel + eps:
            throttle, brake = 0.0, brk
        elif accel > brk + eps:
            throttle, brake = accel, 0.0
        else:
            throttle, brake = 0.0, 0.0

        # Low-speed safety: never hold brake at standstill; small nudge if both zero repeatedly is handled in step()
        return steer, throttle, brake

    def _get_observation(self) -> np.ndarray:
        speed = self._speed_kmh()
        offset_m, yaw_err_norm = self._lane_offset_and_yaw_err()
        vc = self.vehicle.get_control()

        obs = np.array([
            np.clip(speed / 50.0, 0.0, 1.0),
            np.clip(offset_m / 5.0, -1.0, 1.0),
            np.clip(yaw_err_norm, -1.0, 1.0),
            float(vc.throttle),
            float(vc.brake),
            float(np.clip(vc.steer, -1.0, 1.0)),
        ], dtype=np.float32)
        return obs

    def _speed_kmh(self) -> float:
        v = self.vehicle.get_velocity()
        return 3.6 * (v.x**2 + v.y**2 + v.z**2) ** 0.5

    def _lane_offset_and_yaw_err(self) -> Tuple[float, float]:
        # Nearest driving-lane waypoint each step
        wp = self.map.get_waypoint(
            self.vehicle.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )

        # Lateral offset (signed) relative to waypoint transform
        loc = self.vehicle.get_location()
        right = wp.transform.get_right_vector()
        right_vec = np.array([right.x, right.y, right.z], dtype=np.float32)
        delta = np.array([loc.x - wp.transform.location.x,
                          loc.y - wp.transform.location.y,
                          loc.z - wp.transform.location.z], dtype=np.float32)
        offset_m = float(np.dot(delta, right_vec))

        # Heading error normalized to [-1,1] by /90deg
        veh_yaw = np.deg2rad(self.vehicle.get_transform().rotation.yaw)
        wp_yaw = np.deg2rad(wp.transform.rotation.yaw)
        yaw_err = np.arctan2(np.sin(veh_yaw - wp_yaw), np.cos(veh_yaw - wp_yaw))
        yaw_err_norm = float(np.clip(yaw_err / (np.pi / 2.0), -1.0, 1.0))
        return offset_m, yaw_err_norm

    def _spawn_vehicle(self, spawn_index: int) -> carla.Vehicle:
        spawns = self.map.get_spawn_points()
        if not spawns:
            raise RuntimeError("No spawn points found on this map.")
        spawn = spawns[spawn_index % len(spawns)]

        bp_lib = self.world.get_blueprint_library()
        bps = bp_lib.filter("vehicle.*.*")
        if not bps:
            raise RuntimeError("No vehicle blueprints found.")
        bp = bps[0]
        actor = self.world.try_spawn_actor(bp, spawn)
        if actor is None:
            # retry at a random spawn
            actor = self.world.try_spawn_actor(bp, self.np_random.choice(spawns))
            if actor is None:
                raise RuntimeError("Failed to spawn vehicle.")
        return actor

    def _destroy_actors(self):
        if self.vehicle is not None:
            try:
                self.vehicle.destroy()
            except Exception:
                pass
            self.vehicle = None
        for s in self.sensors:
            try:
                s.destroy()
            except Exception:
                pass
        self.sensors = []
