#!/usr/bin/env python3
import os
import sys
import cv2
import numpy as np
from datetime import datetime
import carla
from stable_baselines3 import PPO
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_ppo import ShieldedCarlaEnv

def record_demo(
    model_path: str,
    config_path: str = "configs/town03_easy.yaml",
    duration_seconds: int = 120,
    output_path: str = None,
    use_shield: bool = True
):
    """Record a demo video of the trained agent"""
    
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"demo_{timestamp}.mp4"
    
    # Load model
    model = PPO.load(model_path)
    
    # Create environment with rendering
    env = ShieldedCarlaEnv(config_path, port=2000, use_shield=use_shield)
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 20
    frame_size = (1280, 720)  # Output video size
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    
    print(f"Recording demo to {output_path}")
    print(f"Duration: {duration_seconds} seconds")
    print(f"Press 'q' to stop early")
    
    # Setup spectator camera
    world = env.world
    spectator = world.get_spectator()
    
    # Recording loop
    obs, _ = env.reset()
    start_time = time.time()
    frame_count = 0
    episode_count = 0
    
    try:
        while time.time() - start_time < duration_seconds:
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Update spectator to follow vehicle
            if env.vehicle:
                transform = env.vehicle.get_transform()
                spectator_transform = carla.Transform(
                    transform.location + carla.Location(z=50, x=-30),
                    carla.Rotation(pitch=-45, yaw=transform.rotation.yaw)
                )
                spectator.set_transform(spectator_transform)
            
            # Capture frame (simplified - in practice, set up a camera sensor)
            # This is a placeholder - implement proper camera capture
            frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
            
            # Add overlay information
            overlay = frame.copy()
            
            # Add text overlays
            texts = [
                f"Episode: {episode_count}",
                f"Step: {info.get('steps', 0)}",
                f"Reward: {reward:.2f}",
                f"Speed: {obs[0] * 30:.1f} km/h",
                f"Speed Limit: {int(obs[10] * 100)} km/h",
                f"Must Stop: {'Yes' if obs[11] > 0.5 else 'No'}",
                f"Shield Active: {'Yes' if use_shield else 'No'}"
            ]
            
            y_offset = 30
            for text in texts:
                cv2.putText(overlay, text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_offset += 25
            
            # Blend overlay
            frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
            
            # Write frame
            out.write(frame)
            frame_count += 1
            
            # Show preview (optional)
            cv2.imshow('CARLA Demo', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Reset if episode ends
            if terminated or truncated:
                obs, _ = env.reset()
                episode_count += 1
                print(f"Episode {episode_count} completed. "
                      f"Completion: {info.get('route_completion', 0):.1%}")
    
    finally:
        # Clean up
        out.release()
        cv2.destroyAllWindows()
        env.close()
        
        print(f"\nDemo recording completed!")
        print(f"Total frames: {frame_count}")
        print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Record demo of trained agent')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--config', type=str, default='configs/town03_easy.yaml')
    parser.add_argument('--duration', type=int, default=120)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--shield', action='store_true')
    parser.add_argument('--no-shield', dest='shield', action='store_false')
    parser.set_defaults(shield=True)
    
    args = parser.parse_args()
    
    record_demo(
        model_path=args.model,
        config_path=args.config,
        duration_seconds=args.duration,
        output_path=args.output,
        use_shield=args.shield
    )

