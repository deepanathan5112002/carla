import os
import sys
import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from carla_rule_env import CarlaRuleAwareEnv
from safety_shield import SafetyShield
from infer_onnx import TrafficSignDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ShieldedCarlaEnv(CarlaRuleAwareEnv):
    """CARLA environment with safety shield and perception"""
    
    def __init__(self, config_path: str, port: int = 2000, use_shield: bool = True):
        super().__init__(config_path, port)
        
        self.use_shield = use_shield
        if use_shield:
            self.safety_shield = SafetyShield(self.config)
        
        # Initialize perception
        model_path = self.config['perception']['model_path']
        if os.path.exists(model_path):
            self.detector = TrafficSignDetector(
                model_path,
                confidence_threshold=self.config['perception']['confidence_threshold']
            )
            self.use_perception = True
        else:
            logger.warning(f"Model not found at {model_path}. Running without perception.")
            self.use_perception = False
    
    def step(self, action):
        """Override step to apply safety shield"""
        if self.use_shield and self.use_perception:
            # Get current observation for shield
            obs_pre = self._get_observation()
            
            # Filter action through safety shield
            filtered_action = self.safety_shield.filter_action(
                action, obs_pre, self.last_rule_state
            )
        else:
            filtered_action = action
        
        # Execute filtered action
        return super().step(filtered_action)

def make_env(config_path: str, port: int, seed: int, use_shield: bool = True):
    """Create a single environment instance"""
    def _init():
        env = ShieldedCarlaEnv(config_path, port, use_shield)
        env.reset(seed=seed)
        return env
    
    set_random_seed(seed)
    return _init

def train_ppo(
    config_path: str = "configs/town03_easy.yaml",
    num_envs: int = 4,
    total_timesteps: int = 5_000_000,
    use_shield: bool = True,
    experiment_name: str = "ppo_carla_baseline"
):
    """Main training function"""
    
    logger.info(f"Starting training: {experiment_name}")
    logger.info(f"Config: {config_path}")
    logger.info(f"Starting training: {experiment_name}")
    logger.info(f"Config: {config_path}")
    logger.info(f"Num environments: {num_envs}")
    logger.info(f"Safety shield: {use_shield}")
    
    # Create parallel environments
    base_port = 2000
    env = SubprocVecEnv([
        make_env(config_path, base_port + i * 10, seed=i, use_shield=use_shield)
        for i in range(num_envs)
    ])
    
    # Wrap with monitor for logging
    env = VecMonitor(env, f"logs/{experiment_name}")
    
    # Create eval environment
    eval_env = SubprocVecEnv([
        make_env(config_path, base_port + 100, seed=999, use_shield=use_shield)
    ])
    
    # PPO hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256])
        ),
        verbose=1,
        tensorboard_log=f"./tensorboard/{experiment_name}",
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=f"./models/{experiment_name}",
        name_prefix="checkpoint"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./models/{experiment_name}/best",
        log_path=f"./logs/{experiment_name}/eval",
        eval_freq=10000,
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )
    
    # Train
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True
        )
        
        # Save final model
        model.save(f"models/{experiment_name}/final_model")
        logger.info(f"Training completed! Model saved to models/{experiment_name}/")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        model.save(f"models/{experiment_name}/interrupted_model")
    
    finally:
        env.close()
        eval_env.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train PPO agent in CARLA')
    parser.add_argument('--config', type=str, default='configs/town03_easy.yaml')
    parser.add_argument('--envs', type=int, default=4)
    parser.add_argument('--timesteps', type=int, default=5_000_000)
    parser.add_argument('--shield', action='store_true', help='Use safety shield')
    parser.add_argument('--no-shield', dest='shield', action='store_false')
    parser.add_argument('--name', type=str, default='ppo_carla_baseline')
    parser.set_defaults(shield=True)
    
    args = parser.parse_args()
    
    train_ppo(
        config_path=args.config,
        num_envs=args.envs,
        total_timesteps=args.timesteps,
        use_shield=args.shield,
        experiment_name=args.name
    )