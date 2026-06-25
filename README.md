# Multi-UAV Adaptive Path Planning with Deep Reinforcement Learning

Final Project — IBPI, Yuan Ze University

## Team
- Tran Nam Anh (1113543)
- Tran Phan Anh (1113542)
- Travis (1113545)

## Files
- `train_ppo.py` — PPO agent training script
- `retrain_ppo.py` — extended training (200k steps)
- `multi_uav_env.py` — custom UAV delivery environment (gym-pybullet-drones)
- `evaluate.py` — evaluation script (per-drone metrics)
- `compare_methods.py` — Direct / Reactive / A* / PPO+Classical comparison
- `fly_live2.py` — live flight visualization
- `test_cube.py`, `test_render.py` — simulation rendering tests
- `multi_uav_delivery_demo_animation_en.html` — animated demo
- `*.png` — training curves and evaluation result charts

## Requirements
gym-pybullet-drones 2.1.0, Stable-Baselines3, PyTorch, NumPy, Matplotlib
