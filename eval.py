import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import yaml
import json
from tqdm import tqdm
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_ppo_optimized import ShieldedCarlaEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CarlaEvaluator:
    """Optimized evaluator for trained CARLA agents"""
    
    def __init__(self, model_path: str, config_path: str, normalize_path: str = None):
        self.model = PPO.load(model_path)
        self.config_path = config_path
        self.normalize_path = normalize_path
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def evaluate(
        self, 
        num_episodes: int = 50,
        use_shield: bool = True,
        save_metrics: bool = True,
        deterministic: bool = True
    ):
        """Run evaluation episodes with detailed metrics"""
        
        # Create environment
        def make_eval_env():
            env = ShieldedCarlaEnv(self.config_path, port=2000, use_shield=use_shield)
            return env
        
        env = DummyVecEnv([make_eval_env])
        
        # Load normalization stats if available
        if self.normalize_path and os.path.exists(self.normalize_path):
            env = VecNormalize.load(self.normalize_path, env)
            env.training = False
            env.norm_reward = False
            logger.info(f"Loaded normalization from {self.normalize_path}")
        
        # Metrics storage
        episode_metrics = []
        
        logger.info(f"Starting evaluation: {num_episodes} episodes")
        
        for episode in tqdm(range(num_episodes), desc="Evaluation"):
            obs = env.reset()
            done = False
            
            episode_data = {
                'episode': episode,
                'total_reward': 0,
                'steps': 0,
                'collisions': 0,
                'lane_invasions': 0,
                'speed_violations': 0,
                'stop_violations': 0,
                'off_road_steps': 0,
                'route_completion': 0,
                'distance_traveled': 0,
                'max_speed': 0,
                'avg_lane_offset': [],
                'success': False
            }
            
            step_count = 0
            while not done:
                # Predict action
                action, _ = self.model.predict(obs, deterministic=deterministic)
                
                # Step
                obs, reward, done, info = env.step(action)
                
                # Extract info (handle VecEnv format)
                if isinstance(info, list):
                    info = info[0]
                
                # Update metrics
                episode_data['total_reward'] += reward[0] if isinstance(reward, np.ndarray) else reward
                step_count += 1
                
                # Track lane offset
                lane_offset = abs(obs[0][2] * 5.0)  # Denormalize
                episode_data['avg_lane_offset'].append(lane_offset)
                
                if done:
                    break
            
            # Get final metrics from environment
            episode_data['steps'] = info.get('steps', step_count)
            episode_data['collisions'] = info.get('collisions', 0)
            episode_data['lane_invasions'] = info.get('lane_invasions', 0)
            episode_data['speed_violations'] = info.get('speed_violations', 0)
            episode_data['stop_violations'] = info.get('stop_violations', 0)
            episode_data['off_road_steps'] = info.get('off_road_steps', 0)
            episode_data['route_completion'] = info.get('route_completion', 0)
            episode_data['distance_traveled'] = info.get('distance_traveled', 0)
            episode_data['max_speed'] = info.get('max_speed', 0)
            
            # Calculate average lane offset
            if episode_data['avg_lane_offset']:
                episode_data['avg_lane_offset'] = np.mean(episode_data['avg_lane_offset'])
            else:
                episode_data['avg_lane_offset'] = 0
            
            # Success criteria
            episode_data['success'] = (
                episode_data['collisions'] == 0 and 
                episode_data['route_completion'] > 0.5
            )
            
            episode_metrics.append(episode_data)
            
            if episode % 10 == 0:
                logger.info(f"Episode {episode}: Reward={episode_data['total_reward']:.2f}, "
                           f"Completion={episode_data['route_completion']:.2%}, "
                           f"Success={episode_data['success']}")
        
        env.close()
        
        # Convert to DataFrame
        df = pd.DataFrame(episode_metrics)
        
        if save_metrics:
            # Save raw metrics
            output_file = f'evaluation_results_{"shield" if use_shield else "no_shield"}.csv'
            df.to_csv(output_file, index=False)
            logger.info(f"Saved metrics to {output_file}")
            
            # Generate report
            self._generate_report(df, use_shield)
        
        return df
    
    def _generate_report(self, df: pd.DataFrame, use_shield: bool):
        """Generate comprehensive evaluation report with visualizations"""
        
        # Calculate aggregate metrics
        metrics = {
            'Success Rate': f"{df['success'].mean():.2%}",
            'Route Completion Rate': f"{df['route_completion'].mean():.2%}",
            'Average Episode Reward': f"{df['total_reward'].mean():.2f} ± {df['total_reward'].std():.2f}",
            'Average Episode Length': f"{df['steps'].mean():.1f} ± {df['steps'].std():.1f}",
            'Collision Rate': f"{(df['collisions'] > 0).mean():.2%}",
            'Avg Collisions per Episode': f"{df['collisions'].mean():.2f}",
            'Lane Invasions per Episode': f"{df['lane_invasions'].mean():.2f}",
            'Speed Violations per Episode': f"{df['speed_violations'].mean():.2f}",
            'Stop Violations per Episode': f"{df['stop_violations'].mean():.2f}",
            'Average Distance Traveled': f"{df['distance_traveled'].mean():.2f} m",
            'Average Max Speed': f"{df['max_speed'].mean():.1f} km/h",
            'Average Lane Offset': f"{df['avg_lane_offset'].mean():.2f} m"
        }
        
        # Calculate safety metrics
        total_distance_km = df['distance_traveled'].sum() / 1000
        total_infractions = df[['collisions', 'lane_invasions', 'speed_violations', 'stop_violations']].sum().sum()
        infractions_per_km = total_infractions / max(total_distance_km, 1)
        
        metrics['Total Distance'] = f"{total_distance_km:.2f} km"
        metrics['Infractions per km'] = f"{infractions_per_km:.2f}"
        
        # Create comprehensive plots
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Episode rewards over time
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(df['total_reward'].values, alpha=0.3, color='blue')
        ax1.plot(df['total_reward'].rolling(10).mean(), color='darkblue', linewidth=2)
        ax1.set_title('Episode Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.grid(True, alpha=0.3)
        
        # 2. Route completion over time
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(df['route_completion'].rolling(10).mean(), color='green', linewidth=2)
        ax2.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='50% threshold')
        ax2.set_title('Route Completion Rate')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Completion')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Success rate over time
        ax3 = fig.add_subplot(gs[0, 2])
        success_rolling = df['success'].astype(int).rolling(20).mean()
        ax3.plot(success_rolling, color='purple', linewidth=2)
        ax3.set_title('Success Rate (20-episode rolling)')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Success Rate')
        ax3.set_ylim([0, 1])
        ax3.grid(True, alpha=0.3)
        
        # 4. Violations distribution
        ax4 = fig.add_subplot(gs[1, 0])
        violations = df[['collisions', 'lane_invasions', 'speed_violations', 'stop_violations']].mean()
        colors = ['red', 'orange', 'yellow', 'lightcoral']
        ax4.bar(range(len(violations)), violations.values, color=colors)
        ax4.set_xticks(range(len(violations)))
        ax4.set_xticklabels(['Collisions', 'Lane Inv.', 'Speed Viol.', 'Stop Viol.'], rotation=45, ha='right')
        ax4.set_title('Average Violations per Episode')
        ax4.set_ylabel('Count')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. Distance traveled distribution
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.hist(df['distance_traveled'], bins=25, color='skyblue', edgecolor='black')
        ax5.axvline(df['distance_traveled'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
        ax5.set_title('Distance Traveled Distribution')
        ax5.set_xlabel('Distance (m)')
        ax5.set_ylabel('Episodes')
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')
        
        # 6. Episode outcomes pie chart
        ax6 = fig.add_subplot(gs[1, 2])
        outcomes = {
            'Success': df['success'].sum(),
            'Collision': ((df['collisions'] > 0) & (~df['success'])).sum(),
            'Incomplete': ((df['route_completion'] < 0.5) & (df['collisions'] == 0)).sum()
        }
        colors_pie = ['green', 'red', 'orange']
        ax6.pie(outcomes.values(), labels=outcomes.keys(), autopct='%1.1f%%', colors=colors_pie, startangle=90)
        ax6.set_title('Episode Outcomes')
        
        # 7. Steps per episode
        ax7 = fig.add_subplot(gs[2, 0])
        ax7.hist(df['steps'], bins=25, color='lightgreen', edgecolor='black')
        ax7.axvline(df['steps'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
        ax7.set_title('Episode Length Distribution')
        ax7.set_xlabel('Steps')
        ax7.set_ylabel('Episodes')
        ax7.legend()
        ax7.grid(True, alpha=0.3, axis='y')
        
        # 8. Scatter: Distance vs Reward
        ax8 = fig.add_subplot(gs[2, 1])
        scatter = ax8.scatter(df['distance_traveled'], df['total_reward'], 
                             c=df['collisions'], cmap='RdYlGn_r', alpha=0.6, s=50)
        ax8.set_title('Distance vs Reward (colored by collisions)')
        ax8.set_xlabel('Distance Traveled (m)')
        ax8.set_ylabel('Total Reward')
        plt.colorbar(scatter, ax=ax8, label='Collisions')
        ax8.grid(True, alpha=0.3)
        
        # 9. Metrics summary table
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')
        table_data = [[k, v] for k, v in list(metrics.items())[:10]]  # First 10 metrics
        table = ax9.table(cellText=table_data, 
                         colLabels=['Metric', 'Value'],
                         cellLoc='left',
                         loc='center',
                         colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header
        for i in range(2):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        fig.suptitle(f'CARLA Evaluation Results (Shield: {use_shield})', fontsize=16, fontweight='bold')
        
        output_plot = f'evaluation_report_shield_{use_shield}.png'
        plt.savefig(output_plot, dpi=150, bbox_inches='tight')
        logger.info(f"Saved plot to {output_plot}")
        plt.close()
        
        # Print summary to console
        print("\n" + "="*70)
        print(f"EVALUATION SUMMARY (Shield: {use_shield})")
        print("="*70)
        for metric, value in metrics.items():
            print(f"{metric:35s}: {value}")
        print("="*70)
        
        # Save JSON summary
        json_output = f'evaluation_summary_shield_{use_shield}.json'
        with open(json_output, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved JSON summary to {json_output}")
        
        return metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained CARLA agent (Optimized)')
    parser.add_argument('--model', type=str, required=True, help='Path to model .zip file')
    parser.add_argument('--config', type=str, default='configs/town03_optimized.yaml')
    parser.add_argument('--normalize', type=str, default=None, help='Path to vec_normalize.pkl')
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--shield', action='store_true')
    parser.add_argument('--no-shield', dest='shield', action='store_false')
    parser.add_argument('--stochastic', action='store_true', help='Use stochastic policy')
    parser.set_defaults(shield=True, stochastic=False)
    
    args = parser.parse_args()
    
    # Auto-detect normalization file if not provided
    if args.normalize is None:
        model_dir = os.path.dirname(args.model)
        normalize_path = os.path.join(model_dir, '../vec_normalize.pkl')
        if os.path.exists(normalize_path):
            args.normalize = normalize_path
            print(f"Auto-detected normalization: {normalize_path}")
    
    evaluator = CarlaEvaluator(args.model, args.config, args.normalize)
    evaluator.evaluate(
        num_episodes=args.episodes,
        use_shield=args.shield,
        deterministic=not args.stochastic
    )
