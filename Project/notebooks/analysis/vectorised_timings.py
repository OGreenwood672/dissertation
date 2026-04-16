
# Imports
import sys
import os
sys.path.append(os.path.abspath('../..'))

from controller.marl.main import setup
from controller.marl.config import Config

from project_paths import PROJECT_ROOT, FIGURES_DIR

from time import perf_counter

import torch


from plt_style import set_style
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import numpy as np
import math
from multiprocessing import Pool

from tqdm import tqdm

set_style()

config = Config.from_yaml(PROJECT_ROOT / "configs")
EPISODES = 40
T = 300
N = 2
A = 5
O = 73
GO = 95
NC = 1
C = 8

WORLDS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def simulate_python_simulation(world_idx):
    agent_pos = np.random.rand(N, 2)
    station_pos = np.random.rand(5, 2)
    
    for _ in range(T):
        for i in range(N):
            closest_dist = float('inf')
            closest_station = -1
            
            for j in range(5):
                dist = math.hypot(agent_pos[i][0] - station_pos[j][0], 
                                  agent_pos[i][1] - station_pos[j][1])
                if dist < closest_dist:
                    closest_dist = dist
                    closest_station = j
                    
            agent_pos[i][0] += 0.1
            agent_pos[i][1] += 0.1
            
        fake_obs = np.random.rand(N, O).astype(np.float32)
        fake_global = np.random.rand(GO).astype(np.float32)
        fake_targets = np.random.rand(N, 3).astype(np.float32)
        fake_rewards = np.random.rand(N).astype(np.float32)

    return fake_obs, fake_global, fake_targets, fake_rewards

if __name__ == "__main__":

    print(f"Using device: {device}")

    python_worlds_parallised_to_sps = []

    with Pool() as pool:
        for W in tqdm(WORLDS):
            start = perf_counter()
            for t in tqdm(range(T), leave=False):
                if W == 1:
                    results = simulate_python_simulation(0)
                else:
                    results = pool.map(simulate_python_simulation, range(W))
            runtime = perf_counter() - start
            sps = (T * W) / runtime
            python_worlds_parallised_to_sps.append([W, sps])


    rust_worlds_parallised_to_sps = []

    for W in tqdm(WORLDS):
        config.training.worlds_parallised = W
        system, config = setup(config, device) 
        sim = system['sim']
        
        dummy_actions = np.zeros((W, N), dtype=np.int32)
        dummy_comms = np.zeros((W, N, NC, C), dtype=np.float32)

        start = perf_counter()
        
        for t in range(T):
            curr_obs, curr_global_obs, curr_targets, rewards = sim.parallel_step(dummy_actions, dummy_comms)
            
        runtime = perf_counter() - start
        sps = (T * W) / runtime
        
        rust_worlds_parallised_to_sps.append([W, sps])
        
        sim.shutdown()
        del system

    
    worlds_parallised_to_sps = [[i[0], i[1], j[1]] for i, j in zip(python_worlds_parallised_to_sps, rust_worlds_parallised_to_sps)]

    df = pd.DataFrame(worlds_parallised_to_sps, columns=["worlds_parallelised", "python_steps_per_second", "rust_steps_per_second"])

    plt.figure(figsize=(8, 5))
    sns.lineplot(
        data=df,
        x="worlds_parallelised",
        y="python_steps_per_second",
        marker="o",
        label="Python"
    )
    sns.lineplot(
        data=df,
        x="worlds_parallelised",
        y="rust_steps_per_second",
        marker="o",
        label="Rust"
    )

    plt.xscale("log", base=2)
    # plt.yscale('log')
    plt.xlabel("Worlds parallelised (log 2)")
    plt.ylabel("Steps per Second (SPS)")


    plt.title("SPS vs Worlds parallelised")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    
    plt.savefig(FIGURES_DIR / "sim-comparison.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close('all')
