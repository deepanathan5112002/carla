#!/usr/bin/env python3
import os
import sys
import cv2
import numpy as np
from datetime import datetime
import carla
from stable_baselines3 import PPO
import time
import queue
import threading

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_ppo import ShieldedCarlaEnv


class VideoRecorder:
    """Helper class to record video from CARLA camera"""
    
    def __init__(self, output_path: str, fps: int = 20, frame_size: tuple = (1280, 720)):
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        
        # Frame queue
        self.frame_queue = queue.Queue(maxsize=10)
        self.latest_frame = None
        self.stop_event = threading.Event()
        
    def add_frame(self, frame: np.ndarray):
        """Add frame to queue (called from CARLA callback)"""
        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            pass  # Skip frame if queue full
    
    def get_latest_frame(self) -> np.ndarray:
        """Get most recent frame"""
        # Drain queue and keep latest
        while not self.frame_queue.empty():
            try:
                self.latest_frame = self.frame_queue.get_nowait()
            except queue.Empty:
                break
        
        return self.latest_frame
    
    def write_frame(self, frame: np.ndarray):
        """Write frame to video"""
        if frame is not None:
            # Resize if needed
            if frame.shape[:2][::-1] != self.frame_size:
                frame = cv2.resize(frame, self.frame_size)
            self.writer.write(frame)
    
    def release(self):
        """Release video writer"""
        self.writer.release()


def record_demo(
    model_path: str,
    config_path: str = "configs/town03_easy.yaml",
    duration_seconds: int = 120,
    output_path: str = None,
    use_shield: bool = True,
    show_preview: bool = True
):
    """Record a demo video of the trained agent"""
    
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"demo_{timestamp}.mp4"
    
    print("="*60)
    print(f"🎥 CARLA Demo Recording")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Output: {output_path}")
    print(f"Duration: {duration_seconds}s")
    print(f"Shield: {'Enabled' if use_shield else 'Disabled'}")
    print("="*60)
    
    # Load model
    print("Loading model...")
    model = PPO.load(model_path)
    print("✅ Model loaded")
    
    # Create environment
    print("Creating environment...")
    env = ShieldedCarlaEnv(config_path, port=2000, use_shield=use_shield)
    world = env.world
    print("✅ Environment ready")
    
    # Setup video recorder
    recorder = VideoRecorder(output_path, fps=20, frame_size=(1280, 720))
    
    # Setup recording camera (separate from perception camera)
    print("Setting up recording camera...")
    bp_library = world.get_blueprint_library()
    camera_bp = bp_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '1280')
    camera_bp.set_attribute('image_size_y', '720')
    camera_bp.set_attribute('fov', '90')
    
    # Camera transform (third-person view)
    camera_transform = carla.Transform(
        carla.Location(x=-8.0, z=3.0),  # Behind and above vehicle
        carla.Rotation(pitch=-15.0)
    )
    
    recording_camera = None
    
    def camera_callback(image):
        """Process camera images for recording"""
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        array = array[:, :, :3]  # RGB only
        array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
        recorder.add_frame(array)
    
    # Recording loop
    try:
        obs, _ = env.reset()
        
        # Attach recording camera to vehicle
        recording_camera = world.spawn_actor(
            camera_bp, 
            camera_transform, 
            attach_to=env.vehicle
        )
        recording_camera.listen(camera_callback)
        
        print("\n🎬 Recording started! Press Ctrl+C to stop early\n")
        
        start_time = time.time()
        frame_count = 0
        episode_count = 0
        last_info = {}
        
        while time.time() - start_time < duration_seconds:
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            last_info = info
            
            # Get latest frame
            frame = recorder.get_latest_frame()
            
            if frame is not None:
                # Add overlay information
                overlay = frame.copy()
                
                # Info texts
                texts = [
                    f"Episode: {episode_count}",
                    f"Step: {info.get('steps', 0)}",
                    f"Reward: {reward:.2f} (Total: {info.get('episode_reward', 0):.1f})",
                    f"Speed: {obs[0] * 100:.1f} km/h",
                    f"Speed Limit: {int(obs[10] * 100)} km/h",
                    f"Must Stop: {'YES' if obs[11] > 0.5 else 'No'}",
                    f"Shield: {'ON' if use_shield else 'OFF'}",
                    f"Distance: {info.get('distance_traveled', 0):.1f}m",
                    f"Collisions: {info.get('collisions', 0)}",
                ]
                
                # Draw text on overlay
                y_offset = 30
                for text in texts:
                    cv2.putText(
                        overlay, text, (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                        (0, 255, 0), 2, cv2.LINE_AA
                    )
                    y_offset += 30
                
                # Add status bar at bottom
                status_color = (0, 255, 0) if info.get('collisions', 0) == 0 else (0, 0, 255)
                cv2.rectangle(overlay, (0, 690), (1280, 720), status_color, -1)
                
                elapsed = time.time() - start_time
                status_text = f"Recording: {elapsed:.1f}s / {duration_seconds}s"
                cv2.putText(
                    overlay, status_text, (10, 710),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA
                )
                
                # Blend overlay
                alpha = 0.7
                frame_with_overlay = cv2.addWeighted(frame, 1-alpha, overlay, alpha, 0)
                
                # Write to video
                recorder.write_frame(frame_with_overlay)
                frame_count += 1
                
                # Show preview
                if show_preview:
                    cv2.imshow('CARLA Demo Recording', frame_with_overlay)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\n⚠️  Recording stopped by user")
                        break
            
            # Reset if episode ends
            if terminated or truncated:
                completion = info.get('route_completion', 0)
                collisions = info.get('collisions', 0)
                print(f"Episode {episode_count} complete: "
                      f"Completion={completion:.1%}, Collisions={collisions}")
                obs, _ = env.reset()
                episode_count += 1
                
                # Reattach camera to new vehicle
                if recording_camera and recording_camera.is_alive:
                    recording_camera.stop()
                    recording_camera.destroy()
                
                recording_camera = world.spawn_actor(
                    camera_bp,
                    camera_transform,
                    attach_to=env.vehicle
                )
                recording_camera.listen(camera_callback)
        
        print(f"\n✅ Recording completed!")
        print(f"Total frames: {frame_count}")
        print(f"Episodes: {episode_count}")
        print(f"Final stats: Completion={last_info.get('route_completion', 0):.1%}, "
              f"Collisions={last_info.get('collisions', 0)}")
    
    except KeyboardInterrupt:
        print("\n⚠️  Recording interrupted by user")
    
    except Exception as e:
        print(f"\n❌ Error during recording: {e}")
        raise
    
    finally:
        # Clean up
        print("\nCleaning up...")
        
        if recording_camera and recording_camera.is_alive:
            recording_camera.stop()
            recording_camera.destroy()
        
        recorder.release()
        cv2.destroyAllWindows()
        env.close()
        
        print(f"✅ Video saved to: {output_path}")
        
        # Verify file
        if os.path.exists(output_path):
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"File size: {size_mb:.2f} MB")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Record demo of trained agent')
    parser.add_argument('--model', type=str, required=True, 
                       help='Path to trained model')
    parser.add_argument('--config', type=str, default='town03_easy.yaml',
                       help='Path to config file')
    parser.add_argument('--duration', type=int, default=120,
                       help='Recording duration in seconds')
    parser.add_argument('--output', type=str, default=None,
                       help='Output video path')
    parser.add_argument('--shield', action='store_true',
                       help='Enable safety shield')
    parser.add_argument('--no-shield', dest='shield', action='store_false',
                       help='Disable safety shield')
    parser.add_argument('--no-preview', dest='preview', action='store_false',
                       help='Disable preview window')
    parser.set_defaults(shield=True, preview=True)
    
    args = parser.parse_args()
    
    record_demo(
        model_path=args.model,
        config_path=args.config,
        duration_seconds=args.duration,
        output_path=args.output,
        use_shield=args.shield,
        show_preview=args.preview
    )

