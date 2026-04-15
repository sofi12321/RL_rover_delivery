# Rover Navigation with Soft Actor-Critic (SAC)
This project implements a 2D rover navigation environment with obstacles and trains an agent using the Soft Actor-Critic (SAC) algorithm. It includes both a custom SAC implementation from scratch and a baseline using `stable-baselines3` for comparison.

## Features

- **Custom environment** (`RoversEnv`) built with Gymnasium:
  - Differential-drive robot with continuous actions (steering, acceleration)
  - 8-ray LiDAR sensor readings
  - Randomly generated obstacles and goal positions
  - Collision detection and goal reaching logic
  - Reward shaping (progress, collision penalty, goal bonus, steering cost, etc.)
- **Custom SAC implementation** (PyTorch):
  - Gaussian policy with tanh squashing
  - Double Q‑networks with target networks
  - Automatic entropy tuning
- **Baseline SAC** using `stable-baselines3` for comparison
- **Training & evaluation scripts** with logging (CSV, TensorBoard)
- **Visualization**:
  - Environment rendering with sensor rays
  - Learning curve plots (reward and success rate)
  - Side‑by‑side video comparison of two agents
- **Pretrained models** (1M timesteps) included

## Installation

```bash
git clone https://github.com/your-repo/rovers-sac.git
cd rovers-sac
pip install -r requirements.txt
```

If you want to run the baseline, install `stable-baselines3`:

```bash
pip install stable-baselines3
```

## Project Structure

```
RL_rover_delivery
├── configs/                 # YAML config files
│   ├── default.yaml
│   └── fast_example.yaml
├── env/                      # Environment module (entities, sensors, rover_env, render)
│   ├── __init__.py
│   ├── entities.py           # Obstacle, Goal, Robot
│   ├── sensors.py            # Sensors calculations
│   ├── rover_env.py          # RoversEnv (gym.Env)
│   └── render.py             
├── utils/                     
│   ├── __init__.py
│   ├── config.py              # Config loading
│   ├── replay_buffer.py       # Replay buffer
│   └── helpers.py            
├── sac_custom/                # Custom SAC implementation
│   ├── __init__.py
│   ├── networks.py            # GaussianPolicy, QNetwork
│   └── agent.py               
├── baseline/                   # Baseline agent (wrapper for stable-baselines3)
│   ├── __init__.py
│   └── baseline_agent.py      
├── training/                   # Training scripts
│   ├── __init__.py
│   ├── train_custom.py        # Train custom SAC
│   └── train_baseline.py      # Train baseline SAC
├── evaluation/                  # Evaluation and comparison
│   ├── __init__.py
│   ├── evaluate.py            
│   └── compare.py             
├── visualization/             # Plot learning curves, create comparison videos
│   ├── __init__.py
│   ├── render_env.py          
│   ├── plot_results.py        
│   └── side_by_side.py        
├── results/                      # Output logs, models, plots, videos
│   ├── baseline_run/
│   ├── custom_run/
│   ├── plots/
│   └── videos/
├── tutorial_short.ipynb     # Jupyter notebook with full pipeline
├── requirements.txt
└── README.md
```

## Usage

### Configure the environment and training hyperparameters

Edit `configs/default.yaml` (for basic run), `configs/fast_example.yaml` (for fast check of the algorithm) or create your own. 

## Tutorial Notebook

The file `tutorial_short.ipynb` provides a step‑by‑step guide:

- Setting up the configuration
- Creating the environment and visualising a random map
- Training both custom and baseline SAC (shortened for demo)
- Loading pretrained models (1M steps) and evaluating
- Plotting learning curves and success rates
- Generating side‑by‑side comparison videos


## Results (after 1M timesteps)

| Agent    | Average Reward | Success Rate |
|----------|----------------|---------------|
| Baseline | 456.98         | 0.94          |
| Custom   | 441.78         | 0.92          |

Both agents achieve high success rates (>90%), which means they found successful policies. The baseline SAC slightly outperforms the custom implementation, but both are good. These results approve that the custom implementation is correct.


## Requirements

See `requirements.txt`. Main dependencies:
- Python 3.8+
- gymnasium
- numpy
- torch
- stable-baselines3 (for baseline)
- matplotlib
- pandas
- pyyaml

## Acknowledgements

- The custom SAC implementation follows the original paper *Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor* (Haarnoja et al., 2018).
- Baseline uses the `stable-baselines3` library.
