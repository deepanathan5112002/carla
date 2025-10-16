import os
import sys
import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList, BaseCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from carla_rule_env_improved import CarlaRuleAwareEnv
from safety_shield import SafetyShield
from infer_onnx import TrafficSignDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ShieldedCarlaEnv(CarlaRuleAwareEnv):
    """CARLA environment with safety shield and perception"""
    
    def __init__(self, config_path: str, port: int = 2000, use_shield: bool = True, difficulty: str = 'easy'):
        super().__init__(config_path, port, difficulty)
        
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


class CurriculumCallback(BaseCallback):
    """✅ Callback to automatically progress difficulty based on performance"""
    
    def __init__(self, env, eval_env, verbose=0):
        super().__init__(verbose)
        self.env = env
        self.eval_env = eval_env
        self.last_difficulty_change = 0
        self.eval_success_rates = []
    
    def _on_step(self) -> bool:
        # Check every 50k steps
        if self.num_timesteps - self.last_difficulty_change < 50000:
            return True
        
        # Get current difficulty from first env
        current_diff = self.env.envs[0].difficulty
        
        # Check recent evaluation success rate
        if len(self.eval_success_rates) >= 3:
            avg_success = np.mean(self.eval_success_rates[-3:])
            
            # Progression criteria
            if current_diff == 'easy' and avg_success > 0.5:  # 50% success
                logger.info(f"🎓 Curriculum: Progressing from EASY → MEDIUM (success rate: {avg_success:.1%})")
                self._set_difficulty('medium')
                self.last_difficulty_change = self.num_timesteps
            elif current_diff == 'medium' and avg_success > 0.4:  # 40% success
                logger.info(f"🎓 Curriculum: Progressing from MEDIUM → HARD (success rate: {avg_success:.1%})")
                self._set_difficulty('hard')
                self.last_difficulty_change = self.num_timesteps
        
        return True
    
    def _set_difficulty(self, difficulty):
        """Set difficulty for all environments"""
        for env in self.env.envs:
            env.difficulty = difficulty
        for env in self.eval_env.envs:
            env.difficulty = difficulty


class MetricsCallback(BaseCallback):
    """✅ Callback to log detailed metrics"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.collision_rates = []
        self.waypoints_reached = []
    
    def _on_step(self) -> bool:
        # Check if episode finished
        if self.locals.get('dones') is not None:
            for i, done in enumerate(self.locals['dones']):
                if done:
                    info = self.locals['infos'][i]
                    self.episode_rewards.append(info.get('episode_reward', 0))
                    self.episode_lengths.append(info.get('steps', 0))
                    
                    collision = 1 if info.get('collisions', 0) > 0 else 0
                    self.collision_rates.append(collision)
                    self.waypoints_reached.append(info.get('waypoints_reached', 0))
                    
                    # Log to tensorboard every 10 episodes
                    if len(self.episode_rewards) % 10 == 0:
                        self.logger.record('metrics/avg_reward_10ep', np.mean(self.episode_rewards[-10:]))
                        self.logger.record('metrics/avg_length_10ep', np.mean(self.episode_lengths[-10:]))
                        self.logger.record('metrics/collision_rate_10ep', np.mean(self.collision_rates[-10:]))
                        self.logger.record('metrics/avg_waypoints_10ep', np.mean(self.waypoints_reached[-10:]))
        
        return True


def make_env(config_path: str, port: int, seed: int, use_shield: bool = True, difficulty: str = 'easy'):
    """Create a single environment instance with monitoring"""
    def _init():
        env = ShieldedCarlaEnv(config_path, port, use_shield, difficulty)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    
    set_random_seed(seed)
    return _init


def train_ppo(
    config_path: str = "configs/town03_optimized.yaml",
    num_envs: int = 1,
    total_timesteps: int = 500_000,  # ✅ Increased from 200k
    use_shield: bool = True,
    use_normalization: bool = True,
    use_curriculum: bool = True,  # ✅ NEW: Automatic difficulty progression
    start_difficulty: str = 'easy',  # ✅ NEW: Start difficulty
    experiment_name: str = "ppo_improved"
):
    """Main training function with all improvements"""
    
    logger.info("="*60)
    logger.info(f"🚀 Starting training: {experiment_name}")
    logger.info(f"Config: {config_path}")
    logger.info(f"Num environments: {num_envs}")
    logger.info(f"Total timesteps: {total_timesteps:,}")
    logger.info(f"Safety shield: {use_shield}")
    logger.info(f"Observation normalization: {use_normalization}")
    logger.info(f"Curriculum learning: {use_curriculum}")
    logger.info(f"Start difficulty: {start_difficulty}")
    logger.info(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    logger.info("="*60)
    
    # Create output directories
    os.makedirs(f"./models/{experiment_name}", exist_ok=True)
    os.makedirs(f"./logs/{experiment_name}", exist_ok=True)
    os.makedirs(f"./tensorboard/{experiment_name}", exist_ok=True)
    
    # Create environment
    base_port = 2000
    logger.info(f"Creating {num_envs} environment(s) with difficulty: {start_difficulty}...")
    
    if num_envs == 1:
        env = DummyVecEnv([make_env(config_path, base_port, seed=0, use_shield=use_shield, difficulty=start_difficulty)])
    else:
        env = DummyVecEnv([
            make_env(config_path, base_port, seed=i, use_shield=use_shield, difficulty=start_difficulty)
            for i in range(num_envs)
        ])
    
    # Wrap with monitor
    env = VecMonitor(env, f"logs/{experiment_name}")
    
    # Add observation normalization
    if use_normalization:
        env = VecNormalize(
            env,
            norm_obs=True,
            norm_reward=False,
            clip_obs=10.0,
            training=True,
            gamma=0.99
        )
    logger.info("Training environment created successfully")
    
    # Create eval environment
    logger.info("Creating evaluation environment...")
    eval_env = DummyVecEnv([
        make_env(config_path, base_port, seed=999, use_shield=use_shield, difficulty=start_difficulty)
    ])
    
    if use_normalization:
        eval_env = VecNormalize(
            eval_env,
            norm_obs=True,
            norm_reward=False,
            clip_obs=10.0,
            training=False,
            gamma=0.99
        )
    logger.info("Evaluation environment ready")
    
    # ✅ Optimized PPO hyperparameters for LIDAR + waypoints
    logger.info("Initializing PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=0.03,  # ✅ Reduced for more focused exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),  # ✅ Larger for 25D observations
            activation_fn=torch.nn.Tanh,
            log_std_init=-1.0,
        ),
        verbose=1,
        tensorboard_log=f"./tensorboard/{experiment_name}",
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    logger.info(f"Model device: {model.device}")
    logger.info(f"Policy architecture: 2-layer [256, 256]")
    logger.info(f"Observation space: {env.observation_space.shape}")
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=20000,  # ✅ Save every 20k steps
        save_path=f"./models/{experiment_name}",
        name_prefix="rl_model",
        save_replay_buffer=False,
        save_vecnormalize=True
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./models/{experiment_name}/best",
        log_path=f"./logs/{experiment_name}/eval",
        eval_freq=10000,  # ✅ Evaluate every 10k steps
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )
    
    metrics_callback = MetricsCallback()
    
    # ✅ Add curriculum callback if enabled
    callbacks = [checkpoint_callback, eval_callback, metrics_callback]
    if use_curriculum:
        curriculum_callback = CurriculumCallback(env, eval_env)
        callbacks.append(curriculum_callback)
        logger.info("Curriculum learning enabled")
    
    callback = CallbackList(callbacks)
    
    # Train
    logger.info("Starting training loop...")
    logger.info("🎯 Training will automatically progress: EASY → MEDIUM → HARD")
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
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
    
    parser = argparse.ArgumentParser(description='Train PPO agent in CARLA (Improved with LIDAR + Waypoints)')
    parser.add_argument('--config', type=str, default='configs/town03_optimized.yaml',
                       help='Path to config YAML file')
    parser.add_argument('--envs', type=int, default=1,
                       help='Number of parallel environments')
    parser.add_argument('--timesteps', type=int, default=500_000,
                       help='Total training timesteps')
    parser.add_argument('--shield', action='store_true', 
                       help='Use safety shield')
    parser.add_argument('--no-shield', dest='shield', action='store_false',
                       help='Disable safety shield')
    parser.add_argument('--normalize', action='store_true',
                       help='Use observation normalization')
    parser.add_argument('--no-normalize', dest='normalize', action='store_false',
                       help='Disable normalization')
    parser.add_argument('--curriculum', action='store_true',
                       help='Use curriculum learning (auto difficulty progression)')
    parser.add_argument('--no-curriculum', dest='curriculum', action='store_false',
                       help='Disable curriculum learning')
    parser.add_argument('--difficulty', type=str, default='easy',
                       choices=['easy', 'medium', 'hard'],
                       help='Starting difficulty level')
    parser.add_argument('--name', type=str, default='ppo_improved',
                       help='Experiment name')
    parser.set_defaults(shield=True, normalize=True, curriculum=True)
    
    args = parser.parse_args()
    
    train_ppo(
        config_path=args.config,
        num_envs=args.envs,
        total_timesteps=args.timesteps,
        use_shield=args.shield,
        use_normalization=args.normalize,
        use_curriculum=args.curriculum,
        start_difficulty=args.difficulty,
        experiment_name=args.name
    )
