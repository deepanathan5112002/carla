
from collections import deque
import numpy as np
from typing import Dict

class RuleStateBuffer:
    def __init__(self, buffer_size=10):
        self.buffer = deque(maxlen=buffer_size)
        
    def add(self, state: Dict):
        self.buffer.append(state)
        
    def get_aggregated(self) -> Dict:
        if not self.buffer:
            return {
                'speed_limit': 50,
                'must_stop': False,
                'no_entry': False,
                'traffic_light': 'green',
                'confidence': 0.0
            }
        
        # Simple majority voting
        speed_limits = [s['speed_limit'] for s in self.buffer]
        must_stops = [s['must_stop'] for s in self.buffer]
        
        return {
            'speed_limit': max(set(speed_limits), key=speed_limits.count),
            'must_stop': sum(must_stops) > len(must_stops) // 2,
            'no_entry': False,
            'traffic_light': 'green',
            'confidence': 0.8
        }