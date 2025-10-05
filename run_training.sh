#!/bin/bash

# Run training experiments
echo "Starting CARLA RL+CV Training Pipeline"

# Start CARLA server in background
echo "Starting CARLA server..."
cd /workspace/carla_simulator
./CarlaUE4.sh -RenderOffScreen -carla-port=2000 &
CARLA_PID=$!

# Wait for CARLA to initialize
sleep 10

# Change to project directory
cd /workspace/carla-rl-cv

# Experiment 1: Baseline (No perception, no shield)
echo "Training Baseline Model..."
python carla/train_ppo.py \
    --config town03_easy.yaml \
    --envs 4 \
    --timesteps 1000000 \
    --no-shield \
    --name baseline_no_perception

# Experiment 2: With perception but no shield
echo "Training with Perception (No Shield)..."
python carla/train_ppo.py \
    --config town03_easy.yaml \
    --envs 4 \
    --timesteps 2000000 \
    --no-shield \
    --name perception_no_shield

# Experiment 3: With perception and shield
echo "Training with Perception + Shield..."
python carla/train_ppo.py \
    --config town03_easy.yaml \
    --envs 4 \
    --timesteps 2000000 \
    --shield \
    --name perception_with_shield

# Experiment 4: Curriculum learning
echo "Training with Curriculum..."
python carla/train_ppo.py \
    --config town03_easy.yaml \
    --envs 4 \
    --timesteps 1000000 \
    --shield \
    --name curriculum_phase1

python carla/train_ppo.py \
    --config town05_intersections.yaml \
    --envs 4 \
    --timesteps 2000000 \
    --shield \
    --name curriculum_phase2

# Run evaluation
echo "Running Evaluation..."
python carla/eval.py \
    --model models/perception_with_shield/best/best_model.zip \
    --config town03_easy.yaml \
    --episodes 100 \
    --shield

# Record demo
echo "Recording Demo..."
python carla/record_demo.py \
    --model models/perception_with_shield/best/best_model.zip \
    --config town03_easy.yaml \
    --duration 180 \
    --shield

# Kill CARLA server
kill $CARLA_PID

echo "Training pipeline completed!"