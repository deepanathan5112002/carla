import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import yaml
import json
from tqdm import tqdm
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from carla_rule_env import CarlaRuleAwareEnv
from safety_shield import SafetyShield
from train_ppo import ShieldedCarlaEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CarlaEvaluator:
    """Evaluate trained models in CARLA"""
    
    def __init__(self, model_path: str, config_path: str):
        self.model = PPO.load(model_path)
        self.config_path = config_path
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def evaluate(
        self, 
        num_episodes: int = 100,
        use_shield: bool = True,
        save_metrics: bool = True
    ):
        """Run evaluation episodes"""
        
        # Create environment
        env = ShieldedCarlaEnv(self.config_path, port=2000, use_shield=use_shield)
        
        # Metrics storage
        episode_metrics = []
        
        for episode in tqdm(range(num_episodes), desc="Evaluation"):
            obs, _ = env.reset()
            done = False
            episode_data = {
                'episode': episode,
                'total_reward': 0,
                'steps': 0,
                'collisions': 0,
                'lane_invasions': 0,
                'speed_violations': 0,
                'stop_violations': 0,
                'route_completion': 0,
                'distance_traveled': 0
            }
            
            while not done:
                # Predict action
                action, _ = self.model.predict(obs, deterministic=True)
                
                # Step
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Update metrics
                episode_data['total_reward'] += reward
                episode_data['steps'] += 1
            
            # Get final metrics from environment
            episode_data.update(info)
            episode_metrics.append(episode_data)
            
            logger.info(f"Episode {episode}: Reward={episode_data['total_reward']:.2f}, "
                       f"Completion={episode_data['route_completion']:.2%}")
        
        env.close()
        
        # Convert to DataFrame
        df = pd.DataFrame(episode_metrics)
        
        if save_metrics:
            # Save raw metrics
            df.to_csv(f'evaluation_results_{use_shield}.csv', index=False)
            
            # Generate report
            self._generate_report(df, use_shield)
        
        return df
    
    def _generate_report(self, df: pd.DataFrame, use_shield: bool):
        """Generate evaluation report with plots"""
        
        # Calculate aggregate metrics
        metrics = {
            'Route Completion Rate': f"{df['route_completion'].mean():.2%}",
            'Average Episode Reward': f"{df['total_reward'].mean():.2f} ± {df['total_reward'].std():.2f}",
            'Collisions per Episode': f"{df['collisions'].mean():.2f}",
            'Lane Invasions per Episode': f"{df['lane_invasions'].mean():.2f}",
            'Speed Violations per Episode': f"{df['speed_violations'].mean():.2f}",
            'Stop Violations per Episode': f"{df['stop_violations'].mean():.2f}",
            'Average Distance Traveled': f"{df['distance_traveled'].mean():.2f} m",
            'Success Rate': f"{(df['route_completion'] > 0.9).mean():.2%}"
        }
        
        # Calculate infractions per km
        total_distance_km = df['distance_traveled'].sum() / 1000
        total_infractions = df[['collisions', 'lane_invasions', 'speed_violations', 'stop_violations']].sum().sum()
        infractions_per_km = total_infractions / max(total_distance_km, 1)
        
        metrics['Infractions per km'] = f"{infractions_per_km:.2f}"
        
        # Rule compliance
        total_stops_required = df['stop_violations'].sum() + (df['route_completion'] > 0).sum() * 2  # Estimate
        stops_respected = max(0, total_stops_required - df['stop_violations'].sum())
        stop_compliance = stops_respected / max(total_stops_required, 1)
        metrics['Stop Compliance'] = f"{stop_compliance:.2%}"
        
        # Create plots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Episode rewards
        axes[0, 0].plot(df['total_reward'].rolling(10).mean())
        axes[0, 0].set_title('Episode Rewards (10-episode MA)')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        
        # Route completion
        axes[0, 1].plot(df['route_completion'].rolling(10).mean())
        axes[0, 1].set_title('Route Completion Rate (10-episode MA)')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Completion Rate')
        
        # Violations distribution
        violations = df[['collisions', 'lane_invasions', 'speed_violations', 'stop_violations']].mean()
        axes[0, 2].bar(violations.index, violations.values)
        axes[0, 2].set_title('Average Violations per Episode')
        axes[0, 2].set_ylabel('Count')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # Distance traveled
        axes[1, 0].hist(df['distance_traveled'], bins=20)
        axes[1, 0].set_title('Distance Traveled Distribution')
        axes[1, 0].set_xlabel('Distance (m)')
        axes[1, 0].set_ylabel('Episodes')
        
        # Success vs failure pie chart
        success_counts = pd.Series({
            'Success': (df['route_completion'] > 0.9).sum(),
            'Collision': (df['collisions'] > 0).sum(),
            'Timeout': ((df['route_completion'] < 0.9) & (df['collisions'] == 0)).sum()
        })
        axes[1, 1].pie(success_counts, labels=success_counts.index, autopct='%1.1f%%')
        axes[1, 1].set_title('Episode Outcomes')
        
        # Metrics table
        axes[1, 2].axis('off')
        table_data = [[k, v] for k, v in metrics.items()]
        table = axes[1, 2].table(cellText=table_data, 
                                 colLabels=['Metric', 'Value'],
                                 cellLoc='left',
                                 loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        
        plt.suptitle(f'CARLA Evaluation Results (Shield: {use_shield})')
        plt.tight_layout()
        plt.savefig(f'evaluation_report_shield_{use_shield}.png', dpi=150)
        plt.show()
        
        # Print summary
        print("\n" + "="*50)
        print(f"EVALUATION SUMMARY (Shield: {use_shield})")
        print("="*50)
        for metric, value in metrics.items():
            print(f"{metric:30s}: {value}")
        print("="*50)
        
        return metrics

def compare_experiments(experiments: dict):
    """Compare multiple experiments"""
    
    results = {}
    
    for name, config in experiments.items():
        logger.info(f"\nEvaluating: {name}")
        evaluator = CarlaEvaluator(config['model_path'], config['config_path'])
        df = evaluator.evaluate(
            num_episodes=config.get('num_episodes', 100),
            use_shield=config.get('use_shield', True)
        )
        results[name] = df
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    metrics_to_compare = [
        ('route_completion', 'Route Completion Rate'),
        ('total_reward', 'Episode Reward'),
        ('collisions', 'Collisions per Episode'),
        ('speed_violations', 'Speed Violations per Episode')
    ]
    
    for idx, (metric, title) in enumerate(metrics_to_compare):
        ax = axes[idx // 2, idx % 2]
        
        for name, df in results.items():
            values = df[metric].rolling(10).mean()
            ax.plot(values, label=name, alpha=0.7)
        
        ax.set_title(title)
        ax.set_xlabel('Episode')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Experiment Comparison')
    plt.tight_layout()
    plt.savefig('experiment_comparison.png', dpi=150)
    plt.show()
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained CARLA agent')
    parser.add_argument('--model', type=str, required=True, help='Path to model')
    parser.add_argument('--config', type=str, default='town03_easy.yaml')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--shield', action='store_true')
    parser.add_argument('--no-shield', dest='shield', action='store_false')
    parser.set_defaults(shield=True)
    
    args = parser.parse_args()
    
    evaluator = CarlaEvaluator(args.model, args.config)
    evaluator.evaluate(
        num_episodes=args.episodes,
        use_shield=args.shield
    )
