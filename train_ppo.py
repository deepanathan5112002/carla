#!/usr/bin/env python3
"""
PPO training script for CARLA (3D action: steer, accel, brake).

- Sync CARLA env assumed (see carla_rule_env.py)
- Uses SB3 (Gymnasium) with VecNormalize, EvalCallback, and checkpoints
- Logs to TensorBoard: --log-dir runs/ppo_<exp>

Requirements:
  stable-baselines3>=2.3, gymnasium, numpy, torch, carla PythonAPI (via PYTHONPATH)
"""

import argparse
import os
import time
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor

from carla_rule_env import CarlaRuleEnv, EnvConfig


# ---------------------------
# Utilities
# ---------------------------
def set_global_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_env(cfg: EnvConfig, record_stats: bool = True):
    """
    Factory for a single CARLA env. Wrapped with Monitor (episode stats).
    """
    def _thunk():
        env = CarlaRuleEnv(cfg=cfg, shield=None)  # plug your shield if you want
        if record_stats:
            env = Monitor(env)  # works with gymnasium in SB3 v2
        return env
    return _thunk


def linear_schedule(initial_value: float):
    """
    Linear LR schedule for SB3 (value * progress_remaining).
    """
    def func(progress_remaining: float):
        return progress_remaining * initial_value
    return func


# ---------------------------
# CLI
# ---------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("PPO training for CARLA (3D action env)")

    # Env / CARLA
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--town", type=str, default=None, help='e.g. "Town03" or None to keep loaded map')
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--spawn-index", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--offroad-threshold", type=float, default=4.0)
    p.add_argument("--warmup-ticks", type=int, default=3)
    p.add_argument("--no-rendering-mode", action="store_true", help="Enable CARLA no_rendering_mode")

    # Train
    p.add_argument("--total-timesteps", type=int, default=500_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-dir", type=str, default="runs/ppo_carla")
    p.add_argument("--tb-log-name", type=str, default="PPO_carla")
    p.add_argument("--save-freq", type=int, default=50_000, help="Steps between checkpoints (vec steps)")
    p.add_argument("--eval-freq", type=int, default=25_000, help="Steps between eval runs (vec steps)")
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--progress-bar", action="store_true", help="Show SB3 progress bar")
    p.add_argument("--resume", type=str, default=None, help="Path to .zip to resume from")

    # PPO hyperparams (safe defaults for driving)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-linear", action="store_true", help="Use linear LR schedule from initial --lr")
    p.add_argument("--clip-range", type=float, default=0.1)
    p.add_argument("--n-steps", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--vf-coef", type=float, default=1.0)

    # VecNormalize
    p.add_argument("--norm-obs", action="store_true", help="Normalize observations")
    p.add_argument("--norm-rew", action="store_true", help="Normalize rewards")
    p.add_argument("--clip-obs", type=float, default=10.0)
    p.add_argument("--clip-rew", type=float, default=10.0)

    # Early stopping on eval stagnation
    p.add_argument("--patience-evals", type=int, default=10, help="No. of eval calls w/o improvement")
    p.add_argument("--min-delta", type=float, default=1e-6, help="Min reward improvement to reset patience")

    return p


# ---------------------------
# Main
# ---------------------------
def main(args: argparse.Namespace):
    set_global_seeds(args.seed)

    # Paths
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    models_dir = log_dir / "models"
    models_dir.mkdir(exist_ok=True, parents=True)

    # Build EnvConfig
    env_cfg = EnvConfig(
        host=args.host,
        port=args.port,
        town=args.town,
        fps=args.fps,
        max_steps=args.max_steps,
        offroad_threshold_m=args.offroad_threshold,
        spawn_index=args.spawn_index,
        warmup_ticks=args.warmup_ticks,
        seed=args.seed,
        no_rendering_mode=args.no_rendering_mode,
    )

    # Vectorized training env
    # For CARLA, start with 1 env for stability (increase only if your machine can run multi-client safely)
    train_env = DummyVecEnv([make_env(env_cfg, record_stats=True)])
    train_env = VecMonitor(train_env, filename=str(log_dir / "monitor_train.csv"))

    if args.norm_obs or args.norm_rew:
        train_env = VecNormalize(
            train_env,
            norm_obs=args.norm_obs,
            norm_reward=args.norm_rew,
            clip_obs=args.clip_obs,
            clip_reward=args.clip_rew,
            gamma=args.gamma,
        )

    # Separate eval env (no reward normalization when evaluating)
    eval_env = DummyVecEnv([make_env(env_cfg, record_stats=True)])
    eval_env = VecMonitor(eval_env, filename=str(log_dir / "monitor_eval.csv"))

    # Model
    lr = linear_schedule(args.lr) if args.lr_linear else args.lr
    policy_kwargs = dict()  # customize net arch if desired

    if args.resume:
        print(f"[INFO] Resuming from {args.resume}")
        model = PPO.load(
            args.resume,
            env=train_env,
            tensorboard_log=str(log_dir),
            device="auto",
        )
        # If VecNormalize was used previously, load stats too:
        stats_path = Path(args.resume).with_suffix(".vn.pkl")
        if stats_path.exists() and isinstance(train_env, VecNormalize):
            train_env.load_statistics(str(stats_path))
    else:
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            tensorboard_log=str(log_dir),
            seed=args.seed,
            policy_kwargs=policy_kwargs,
            verbose=1,
        )

    # Callbacks: checkpoints + eval (with early stop on no improvement)
    callbacks = []

    # Checkpoint every save_freq steps
    callbacks.append(
        CheckpointCallback(
            save_freq=args.save_freq,
            save_path=str(models_dir),
            name_prefix="ppo_carla",
            save_replay_buffer=False,
            save_vecnormalize=True,  # saves VecNormalize stats alongside
        )
    )

    # Early stop if eval doesn't improve
    stop_cb = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=args.patience_evals,
        min_epsilon=args.min_delta,
        verbose=1,
    )

    best_model_dir = log_dir / "best_model"
    best_model_dir.mkdir(exist_ok=True, parents=True)

    eval_cb = EvalCallback(
        eval_env=eval_env,
        callback_after_eval=stop_cb,
        best_model_save_path=str(best_model_dir),
        log_path=str(log_dir / "eval"),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=False,
        render=False,
        warn=True,
    )
    callbacks.append(eval_cb)

    # Train
    print(f"[INFO] Starting training for {args.total_timesteps:,} timesteps")
    print(f"[INFO] TensorBoard: {log_dir}  (run with: tensorboard --logdir {log_dir})")
    start = time.time()

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        tb_log_name=args.tb_log_name,
        progress_bar=args.progress_bar,
    )

    dur = time.time() - start
    print(f"[INFO] Training finished in {dur/60:.1f} min")

    # Save final artifacts
    final_path = models_dir / "ppo_carla_final"
    model.save(str(final_path))
    print(f"[INFO] Saved final model to {final_path}.zip")

    # Save VecNormalize stats (if used)
    if isinstance(train_env, VecNormalize):
        vn_path = str(final_path) + ".vn.pkl"
        train_env.save(vn_path)
        print(f"[INFO] Saved VecNormalize stats to {vn_path}")

    # Optional quick eval at the very end
    try:
        mean_r, std_r = evaluate(model, eval_env, n_episodes=max(3, args.eval_episodes // 2))
        print(f"[INFO] Final quick eval: mean_reward={mean_r:.2f} ± {std_r:.2f}")
    except Exception as e:
        print(f"[WARN] Final eval failed: {e}")

    # Clean up
    train_env.close()
    eval_env.close()


# ---------------------------
# Simple eval loop
# ---------------------------
def evaluate(model: PPO, vec_env, n_episodes: int = 5):
    """
    Runs n episodes on vec_env (assumed VecMonitor) and returns mean ± std episodic reward.
    """
    rewards = []
    for _ in range(n_episodes):
        obs = vec_env.reset()
        done = False
        ep_rew = 0.0
        # Gymnasium vec env returns arrays; we loop until all sub-envs done
        states_done = np.array([False] * vec_env.num_envs)
        while not states_done.all():
            action, _ = model.predict(obs, deterministic=False)
            obs, rew, dones, infos = vec_env.step(action)
            ep_rew += float(np.mean(rew))
            # for single-env DummyVecEnv this works; for multi-env, track per-env dones
            if isinstance(dones, np.ndarray):
                states_done = np.logical_or(states_done, dones)
            else:
                states_done = np.array([dones])
            # optional: break if takes too long
        rewards.append(ep_rew)
    return float(np.mean(rewards)), float(np.std(rewards))


# ---------------------------
# Entrypoint
# ---------------------------
if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    main(args)

