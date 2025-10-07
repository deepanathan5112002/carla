import numpy as np
from typing import Dict, Tuple

class SafetyShield:
    """Safety shield for filtering dangerous actions"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.stop_timer = 0
        self.min_stop_duration = 40  # steps (2 seconds at 20 FPS)
    
    def filter_action(
        self, 
        action: np.ndarray, 
        observation: np.ndarray,
        rule_state: Dict
    ) -> np.ndarray:
        """Apply safety filters to action"""
        
        if len(observation) < 15:
            return action.copy()
        
        filtered_action = action.copy()
        speed_kmh = observation[0] * 100.0
        
        # Rule 1: Stop sign/red light (KEEP THIS)
        if rule_state.get('must_stop', False) or rule_state.get('traffic_light') == 'red':
            if speed_kmh > 1.0:
                filtered_action[1] = 0.0
                filtered_action[2] = 1.0
                self.stop_timer = 0
            elif self.stop_timer < self.min_stop_duration:
                filtered_action[1] = 0.0
                filtered_action[2] = 0.5
                self.stop_timer += 1
        else:
            self.stop_timer = 0
        
        # Rule 2: Speed limit (MUCH MORE LENIENT)
        speed_limit = rule_state.get('speed_limit', 50)
        tolerance = 15  # ✅ INCREASED from 5 to 15 km/h
        
        if speed_kmh > speed_limit + tolerance:
            filtered_action[1] = min(filtered_action[1], 0.5)  # ✅ Allow more throttle
            
            if speed_kmh > speed_limit + 2 * tolerance:
                filtered_action[1] = 0.0
                filtered_action[2] = max(filtered_action[2], 0.3)  # ✅ Gentler braking
        
        # Rule 3: No-entry (KEEP THIS)
        if rule_state.get('no_entry', False):
            filtered_action[1] = 0.0
            filtered_action[2] = 1.0
            if abs(filtered_action[0]) < 0.5:
                filtered_action[0] = 1.0 if np.random.random() > 0.5 else -1.0
        
        # ✅ REMOVE anti-collision and lane keeping from shield
        # Let the RL agent learn these behaviors!
        
        return filtered_action
    
    def reset(self):
        """Reset shield state"""
        self.stop_timer = 0
