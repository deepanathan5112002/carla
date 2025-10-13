import os
import sys
import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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
            logger.info("Safety shield enabled")
        
        # Initialize perception if model exists
        model_path = self.config['perception']['model_path']
        if os.path.exists(model_path):
            try:
                self.detector = TrafficSignDetector(
                    model_path,
                    confidence_threshold=self.config['perception']['confidence_threshold']
                )
                self.use_perception = True
                logger.info(f"Perception enabled: {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load perception model: {e}")
                self.use_perception = False
        else:
            logger.warning(f"Model not found at {model_path}. Running without perception.")
            self.use_perception = False
    
    def step(self, action):
        """Override step to apply safety shield"""
        if self.use_shield:
            obs_pre = self._get_observation()
            filtered_action = self.safety_shield.filter_action(
                action, obs_pre, self.last_rule_state
            )
        else:
            filtered_action = action
        
        return super().step(filtered_action)
    
    def reset(self, seed=None, options=None):
        """Override reset to reset shield state"""
        if self.use_shield:
            self.safety_shield.reset()
        return super().reset(seed=seed, options=options)


def make_env(config_path: str, port: int, seed: int, use_shield: bool = True):
    """Create a single environment instance with monitoring"""
    def _init():
        env = ShieldedCarlaEnv(config_path, port, use_shield)
        env = Monitor(env)  # Add monitoring wrapper
        env.reset(seed=seed)
        return env
    
    set_random_seed(seed)
    return _init


def train_ppo(
    config_path: str = "configs/town03_optimized.yaml",
    num_envs: int = 1,
    total_timesteps: int = 200_000,
    use_shield: bool = True,
    use_normalization: bool = True,
    experiment_name: str = "ppo_optimized"
):
    """Main training function with optimizations"""
    
    logger.info("="*60)
    logger.info(f"Starting training: {experiment_name}")
    logger.info(f"Config: {config_path}")
    logger.info(f"Num environments: {num_envs}")
    logger.info(f"Total timesteps: {total_timesteps:,}")
    logger.info(f"Safety shield: {use_shield}")
    logger.info(f"Observation normalization: {use_normalization}")
    logger.info(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    logger.info("="*60)
    
    # Create output directories
    os.makedirs(f"./models/{experiment_name}", exist_ok=True)
    os.makedirs(f"./logs/{experiment_name}", exist_ok=True)
    os.makedirs(f"./tensorboard/{experiment_name}", exist_ok=True)
    
    # Create environment
    base_port = 2000
    logger.info(f"Creating {num_envs} environment(s)...")
    
    if num_envs == 1:
        env = DummyVecEnv([make_env(config_path, base_port, seed=0, use_shield=use_shield)])
    else:
        # For multiple envs, use DummyVecEnv (safer than SubprocVecEnv)
        env = DummyVecEnv([
            make_env(config_path, base_port, seed=i, use_shield=use_shield)
            for i in range(num_envs)
        ])
    
    # Add observation normalization for better learning
    if use_normalization:
        env = VecNormalize(
            env, 
            norm_obs=True, 
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=0.99
        )
        logger.info("Observation normalization enabled")
    
    # Wrap with monitor
    env = VecMonitor(env, f"logs/{experiment_name}")
    logger.info("Environment created successfully")
    
    # Create eval environment
    logger.info("Creating evaluation environment...")
    eval_env = DummyVecEnv([
        make_env(config_path, base_port, seed=999, use_shield=use_shield)
    ])
    if use_normalization:
        eval_env = VecNormalize(
            eval_env,
            norm_obs=True,
            norm_reward=False,  # Don't normalize rewards during eval
            clip_obs=10.0,
            training=False
        )
    logger.info("Evaluation environment ready")
    
    # Optimized PPO hyperparameters
    logger.info("Initializing PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        learning_rate=5e-4,      # ✅ INCREASE from 3e-4
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=0.1,           # ✅ INCREASE from 0.01 - MORE exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        #policy_kwargs=dict(net_arch=dict(pi=[256, 256, 128],vf=[256, 256, 128]),activation_fn=torch.nn.ReLU),
        policy_kwargs=dict(
            net_arch=dict(pi=[128, 128], vf=[128, 128]),  # Smaller network
            activation_fn=torch.nn.Tanh,  # Better for continuous control
            log_std_init=-2.0,  # Start with less random actions
        ),
        verbose=1,
        tensorboard_log=f"./tensorboard/{experiment_name}",
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    logger.info(f"Model device: {model.device}")
    logger.info(f"Policy architecture: 3-layer [256, 256, 128]")
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=f"./models/{experiment_name}",
        name_prefix="rl_model",
        save_replay_buffer=False,
        save_vecnormalize=True
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./models/{experiment_name}/best",
        log_path=f"./logs/{experiment_name}/eval",
        eval_freq=5000,  # Evaluate every 5k steps
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )
    
    # Combine callbacks
    callback = CallbackList([checkpoint_callback, eval_callback])
    
    # Train
    logger.info("Starting training loop...")
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=False  # Disable if tqdm causes issues
        )
        
        # Save final model
        final_path = f"models/{experiment_name}/final_model"
        model.save(final_path)
        
        # Save normalization stats
        if use_normalization:
            env.save(f"models/{experiment_name}/vec_normalize.pkl")
        
        logger.info(f"✅ Training completed! Model saved to {final_path}")
        
    except KeyboardInterrupt:
        logger.info("⚠️  Training interrupted by user")
        interrupted_path = f"models/{experiment_name}/interrupted_model"
        model.save(interrupted_path)
        if use_normalization:
            env.save(f"models/{experiment_name}/vec_normalize_interrupted.pkl")
        logger.info(f"Model saved to {interrupted_path}")
    
    except Exception as e:
        logger.error(f"❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        logger.info("Closing environments...")
        env.close()
        eval_env.close()
        logger.info("Cleanup complete")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train PPO agent in CARLA (Optimized)')
    parser.add_argument('--config', type=str, default='configs/town03_optimized.yaml',
                       help='Path to config YAML file')
    parser.add_argument('--envs', type=int, default=1,
                       help='Number of parallel environments')
    parser.add_argument('--timesteps', type=int, default=200_000,
                       help='Total training timesteps')
    parser.add_argument('--shield', action='store_true', 
                       help='Use safety shield')
    parser.add_argument('--no-shield', dest='shield', action='store_false',
                       help='Disable safety shield')
    parser.add_argument('--normalize', action='store_true',
                       help='Use observation normalization')
    parser.add_argument('--no-normalize', dest='normalize', action='store_false',
                       help='Disable normalization')
    parser.add_argument('--name', type=str, default='ppo_optimized',
                       help='Experiment name')
    parser.set_defaults(shield=True, normalize=True)
    
    args = parser.parse_args()
    
    train_ppo(
        config_path=args.config,
        num_envs=args.envs,
        total_timesteps=args.timesteps,
        use_shield=args.shield,
        use_normalization=args.normalize,
        experiment_name=args.name
    )
