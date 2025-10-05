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
        
        filtered_action = action.copy()
        
        # Extract relevant info from observation
        speed = observation[0] * 30.0  # De-normalize
        
        # Stop sign/red light enforcement
        if rule_state['must_stop'] or rule_state['traffic_light'] == 'red':
            if speed > 1.0:  # Still moving
                filtered_action[1] = 0.0  # No throttle
                filtered_action[2] = 1.0  # Full brake
                self.stop_timer = 0
            elif self.stop_timer < self.min_stop_duration:
                filtered_action[1] = 0.0  # Hold stop
                filtered_action[2] = 0.5  # Light brake
                self.stop_timer += 1
        else:
            self.stop_timer = 0
        
        # Speed limit enforcement
        speed_limit = rule_state['speed_limit']
        if speed > speed_limit + 5:  # 5 km/h tolerance
            filtered_action[1] = min(filtered_action[1], 0.3)  # Limit throttle
            if speed > speed_limit + 10:
                filtered_action[1] = 0.0
                filtered_action[2] = max(filtered_action[2], 0.5)
        
        # No-entry zone protection
        if rule_state['no_entry']:
            filtered_action[1] = 0.0  # No forward movement
            filtered_action[2] = 1.0  # Stop
            # Force turn away (simplified)
            if abs(filtered_action[0]) < 0.5:
                filtered_action[0] = 1.0 if np.random.random() > 0.5 else -1.0
        
        # Anti-collision (basic)
        collision_flag = observation[4]
        if collision_flag > 0.5:
            filtered_action[2] = 1.0  # Emergency brake
        
        return filtered_action
    
    def reset(self):
        """Reset shield state"""
        self.stop_timer = 0