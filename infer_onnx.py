import numpy as np
import cv2
import onnxruntime as ort
from collections import deque
import time
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class TrafficSignDetector:
    """ONNX-based traffic sign/light detector"""
    
    def __init__(
        self, 
        model_path: str,
        confidence_threshold: float = 0.5,
        temporal_buffer_size: int = 10
    ):
        # Initialize ONNX runtime
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        
        self.confidence_threshold = confidence_threshold
        
        # Temporal buffer for stabilization
        self.detection_buffer = deque(maxlen=temporal_buffer_size)
        
        # Class mapping (adjust based on your model)
        self.class_map = {
            0: 'speed_20',
            1: 'speed_30',
            2: 'speed_50',
            3: 'speed_60',
            4: 'speed_70',
            5: 'speed_80',
            6: 'speed_100',
            7: 'speed_120',
            8: 'no_entry',
            9: 'stop',
            10: 'traffic_light_red',
            11: 'traffic_light_yellow',
            12: 'traffic_light_green'
        }
        
        # Rule state decay parameters
        self.decay_rate = 0.95
        self.last_detection_time = {}
        self.persistent_state = self._default_state()
    
    def _default_state(self) -> Dict:
        """Default rule state"""
        return {
            'speed_limit': 50,
            'must_stop': False,
            'no_entry': False,
            'traffic_light': 'green',
            'confidence': 0.0
        }
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for inference"""
        # Resize to model input size
        height, width = self.input_shape[2:4]
        resized = cv2.resize(image, (width, height))
        
        # Normalize
        normalized = resized.astype(np.float32) / 255.0
        
        # Transpose to NCHW format
        transposed = np.transpose(normalized, (2, 0, 1))
        
        # Add batch dimension
        batched = np.expand_dims(transposed, axis=0)
        
        return batched
    
    def postprocess(self, outputs: np.ndarray) -> Dict:
        """Process model outputs into rule state"""
        # This is a simplified version - adjust based on your model's output format
        # Assuming YOLOv8 style outputs
        
        detections = []
        
        # Parse detections (simplified)
        for detection in outputs[0]:
            confidence = detection[4]
            if confidence > self.confidence_threshold:
                class_probs = detection[5:]
                class_id = np.argmax(class_probs)
                class_conf = class_probs[class_id]
                
                if class_conf > self.confidence_threshold:
                    detections.append({
                        'class_id': class_id,
                        'confidence': float(class_conf),
                        'bbox': detection[:4]
                    })
        
        # Update detection buffer
        self.detection_buffer.append(detections)
        
        # Aggregate detections over time
        rule_state = self._aggregate_detections()
        
        return rule_state
    
    def _aggregate_detections(self) -> Dict:
        """Aggregate temporal detections into stable rule state"""
        current_time = time.time()
        
        # Count detections by class
        class_counts = {}
        total_confidence = 0.0
        
        for frame_detections in self.detection_buffer:
            for det in frame_detections:
                class_name = self.class_map.get(det['class_id'], 'unknown')
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                total_confidence += det['confidence']
        
        # Update persistent state with decay
        state = self.persistent_state.copy()
        
        # Speed limits (persistent until new one detected)
        for speed_class in ['speed_20', 'speed_30', 'speed_50', 'speed_60', 
                           'speed_70', 'speed_80', 'speed_100', 'speed_120']:
            if class_counts.get(speed_class, 0) > len(self.detection_buffer) // 3:
                speed_value = int(speed_class.split('_')[1])
                state['speed_limit'] = speed_value
                self.last_detection_time['speed_limit'] = current_time
        
        # Stop signs (temporary state)
        if class_counts.get('stop', 0) > len(self.detection_buffer) // 3:
            state['must_stop'] = True
            self.last_detection_time['stop'] = current_time
        elif current_time - self.last_detection_time.get('stop', 0) > 3.0:
            state['must_stop'] = False
        
        # Traffic lights
        red_count = class_counts.get('traffic_light_red', 0)
        yellow_count = class_counts.get('traffic_light_yellow', 0)
        green_count = class_counts.get('traffic_light_green', 0)
        
        if red_count > max(yellow_count, green_count):
            state['traffic_light'] = 'red'
            state['must_stop'] = True
        elif yellow_count > max(red_count, green_count):
            state['traffic_light'] = 'yellow'
        elif green_count > 0:
            state['traffic_light'] = 'green'
        
        # No entry zones
        if class_counts.get('no_entry', 0) > len(self.detection_buffer) // 3:
            state['no_entry'] = True
            self.last_detection_time['no_entry'] = current_time
        elif current_time - self.last_detection_time.get('no_entry', 0) > 2.0:
            state['no_entry'] = False
        
        # Average confidence
        if len(self.detection_buffer) > 0:
            state['confidence'] = total_confidence / max(1, sum(len(d) for d in self.detection_buffer))
        
        self.persistent_state = state
        return state
    
    def infer(self, image: np.ndarray) -> Dict:
        """Run inference on image and return rule state"""
        # Preprocess
        input_tensor = self.preprocess(image)
        
        # Run inference
        outputs = self.session.run(None, {self.input_name: input_tensor})
        
        # Postprocess
        rule_state = self.postprocess(outputs[0])
        
        return rule_state