import os
import sys
import numpy as np
import pandas as pd
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import logging
import json
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from carla_rule_env_improved import CarlaRuleAwareEnv
from train_ppo_improved import ShieldedCarlaEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_model(
    model_path: str,
    config_path: str = "configs/town03_optimized.yaml",
    num_episodes: int = 50,
    port: int = 2000,
    use_shield: bool = False,
    difficulty: str = 'easy',
    vec_normalize_path: str = None,
    output_prefix: str = "evaluation"
):
    """
    Evaluate a trained PPO model
    
    Args:
        model_path: Path to saved model
        config_path: Path to environment config
        num_episodes: Number of episodes to evaluate
        port: CARLA port
        use_shield: Whether to use safety shield
        difficulty: Difficulty level ('easy', 'medium', 'hard')
        vec_normalize_path: Path to VecNormalize stats (if used during training)
        output_prefix: Prefix for output files
    """
    
    logger.info("="*60)
    logger.info(f"🔍 Evaluating model: {model_path}")
    logger.info(f"Config: {config_path}")
    logger.info(f"Episodes: {num_episodes}")
    logger.info(f"Shield: {use_shield}")
    logger.info(f"Difficulty: {difficulty}")
    logger.info("="*60)
    
    # Load model
    logger.info("Loading model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PPO.load(model_path, device=device)
    logger.info(f"Model loaded on {device}")
    
    # Create environment
    logger.info("Creating evaluation environment...")
    env = ShieldedCarlaEnv(config_path, port, use_shield, difficulty)
    
    # Wrap in vectorized env if normalization was used
    if vec_normalize_path and os.path.exists(vec_normalize_path):
        logger.info(f"Loading VecNormalize stats from {vec_normalize_path}")
        env = DummyVecEnv([lambda: env])
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False  # Don't update stats during eval
        env.norm_reward = False
    
    # Storage for results
    results = []
    
    logger.info(f"\nStarting evaluation for {num_episodes} episodes...")
    
    try:
        for episode in tqdm(range(num_episodes), desc="Evaluating"):
            obs = env.reset()
            done = False
            truncated = False
            episode_data = {
                'episode': episode,
                'total_reward': 0.0,
                'steps': 0,
                'collisions': 0,
                'lane_invasions': 0,
                'speed_violations': 0,
                'stop_violations': 0,
                'off_road_steps': 0,
                'route_completion': 0.0,
                'distance_traveled': 0.0,
                'max_speed': 0.0,
                'waypoints_reached': 0,
                'close_calls': 0,
                'success': False
            }
            
            while not (done or truncated):
                # Get action from model
                action, _ = model.predict(obs, deterministic=True)
                
                # Step environment
                if isinstance(env, VecNormalize):
                    obs, reward, done, info = env.step(action)
                    done = done[0]
                    reward = reward[0]
                    info = info[0]
                else:
                    obs, reward, done, truncated, info = env.step(action)
                
                episode_data['total_reward'] += reward
                episode_data['steps'] += 1
            
            # Extract final metrics from info
            episode_data['collisions'] = info.get('collisions', 0)
            episode_data['lane_invasions'] = info.get('lane_invasions', 0)
            episode_data['speed_violations'] = info.get('speed_violations', 0)
            episode_data['stop_violations'] = info.get('stop_violations', 0)
            episode_data['off_road_steps'] = info.get('off_road_steps', 0)
            episode_data['route_completion'] = info.get('route_completion', 0.0)
            episode_data['distance_traveled'] = info.get('distance_traveled', 0.0)
            episode_data['max_speed'] = info.get('max_speed', 0.0)
            episode_data['waypoints_reached'] = info.get('waypoints_reached', 0)
            episode_data['close_calls'] = info.get('close_calls', 0)
            
            # Determine success (reached destination without collision)
            episode_data['success'] = (episode_data['route_completion'] >= 1.0 and 
                                      episode_data['collisions'] == 0)
            
            # Calculate average lane offset
            episode_data['avg_lane_offset'] = 0.0  # Not easily accessible, placeholder
            
            results.append(episode_data)
            
            # Log progress every 10 episodes
            if (episode + 1) % 10 == 0:
                recent_success = sum(r['success'] for r in results[-10:]) / 10
                recent_collision = sum(r['collisions'] > 0 for r in results[-10:]) / 10
                logger.info(f"Episodes {episode+1}/{num_episodes} | "
                           f"Success: {recent_success:.1%} | "
                           f"Collision: {recent_collision:.1%}")
    
    finally:
        # Clean up
        logger.info("\nCleaning up...")
        if isinstance(env, VecNormalize):
            env.close()
        else:
            env.close()
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Calculate summary statistics
    summary = {
        "Success Rate": f"{df['success'].mean() * 100:.2f}%",
        "Route Completion Rate": f"{(df['route_completion'] >= 1.0).mean() * 100:.2f}%",
        "Average Episode Reward": f"{df['total_reward'].mean():.2f} ± {df['total_reward'].std():.2f}",
        "Average Episode Length": f"{df['steps'].mean():.1f} ± {df['steps'].std():.1f}",
        "Collision Rate": f"{(df['collisions'] > 0).mean() * 100:.2f}%",
        "Avg Collisions per Episode": f"{df['collisions'].mean():.2f}",
        "Lane Invasions per Episode": f"{df['lane_invasions'].mean():.2f}",
        "Speed Violations per Episode": f"{df['speed_violations'].mean():.2f}",
        "Stop Violations per Episode": f"{df['stop_violations'].mean():.2f}",
        "Average Distance Traveled": f"{df['distance_traveled'].mean():.2f} m",
        "Average Max Speed": f"{df['max_speed'].mean():.1f} km/h",
        "Average Waypoints Reached": f"{df['waypoints_reached'].mean():.1f}",
        "Average Close Calls": f"{df['close_calls'].mean():.1f}",
        "Total Distance": f"{df['distance_traveled'].sum() / 1000:.2f} km",
    }
    
    # Calculate infractions per km
    total_infractions = (df['collisions'].sum() + 
                        df['lane_invasions'].sum() + 
                        df['speed_violations'].sum() + 
                        df['stop_violations'].sum())
    total_distance_km = df['distance_traveled'].sum() / 1000
    if total_distance_km > 0:
        summary["Infractions per km"] = f"{total_infractions / total_distance_km:.2f}"
    else:
        summary["Infractions per km"] = "N/A"
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("📊 EVALUATION SUMMARY")
    logger.info("="*60)
    for key, value in summary.items():
        logger.info(f"{key:.<40} {value}")
    logger.info("="*60)
    
    # Save results
    shield_suffix = "shield_True" if use_shield else "shield_False"
    difficulty_suffix = f"difficulty_{difficulty}"
    
    csv_path = f"{output_prefix}_{shield_suffix}_{difficulty_suffix}.csv"
    json_path = f"{output_prefix}_summary_{shield_suffix}_{difficulty_suffix}.json"
    
    df.to_csv(csv_path, index=False)
    logger.info(f"\n✅ Detailed results saved to: {csv_path}")
    
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"✅ Summary saved to: {json_path}")
    
    # Print some episode details
    logger.info("\n📋 Sample Episode Results:")
    logger.info("-" * 60)
    for i in range(min(5, len(df))):
        row = df.iloc[i]
        logger.info(f"Episode {i}: Success={row['success']}, "
                   f"Reward={row['total_reward']:.1f}, "
                   f"Distance={row['distance_traveled']:.1f}m, "
                   f"Waypoints={row['waypoints_reached']}, "
                   f"Collisions={row['collisions']}")
    
    return df, summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained PPO model in CARLA')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.zip file)')
    parser.add_argument('--config', type=str, default='configs/town03_optimized.yaml',
                       help='Path to config YAML file')
    parser.add_argument('--episodes', type=int, default=50,
                       help='Number of evaluation episodes')
    parser.add_argument('--port', type=int, default=2000,
                       help='CARLA server port')
    parser.add_argument('--shield', action='store_true',
                       help='Use safety shield during evaluation')
    parser.add_argument('--no-shield', dest='shield', action='store_false',
                       help='Disable safety shield')
    parser.add_argument('--difficulty', type=str, default='easy',
                       choices=['easy', 'medium', 'hard'],
                       help='Difficulty level for evaluation')
    parser.add_argument('--vec-normalize', type=str, default=None,
                       help='Path to VecNormalize stats file (.pkl)')
    parser.add_argument('--output', type=str, default='evaluation_results',
                       help='Output file prefix')
    parser.set_defaults(shield=False)
    
    args = parser.parse_args()
    
    # Auto-detect vec_normalize path if not provided
    if args.vec_normalize is None:
        model_dir = os.path.dirname(args.model)
        potential_normalize = os.path.join(model_dir, '..', 'vec_normalize.pkl')
        if os.path.exists(potential_normalize):
            args.vec_normalize = potential_normalize
            logger.info(f"Auto-detected VecNormalize stats: {potential_normalize}")
    
    evaluate_model(
        model_path=args.model,
        config_path=args.config,
        num_episodes=args.episodes,
        port=args.port,
        use_shield=args.shield,
        difficulty=args.difficulty,
        vec_normalize_path=args.vec_normalize,
        output_prefix=args.output
    )
