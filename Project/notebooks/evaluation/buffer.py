# %%
# Imports
import sys
import os
sys.path.append(os.path.abspath('../..'))


from controller.marl.config import Config

from project_paths import PROJECT_ROOT

from time import perf_counter

import torch

from plt_style import set_style
import numpy as np

from tqdm import tqdm

import gc

# %%
set_style()

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %%
config = Config.from_yaml(PROJECT_ROOT / "configs")
EPISODES = 40
T = config.training.simulation_timesteps
N = 2
A = 5
O = 73
GO = 95
NC = 1
C = 8
W = 32

# %%
WORLDS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

# %% [markdown]
# # Buffer Implementation Comparison

# %%
X = 20000

# %%
fake_obs = [np.random.rand(W, N, O).astype(np.float32) for _ in range(X)]
fake_global = [np.random.rand(W, GO).astype(np.float32) for _ in range(X)]
fake_actions = [np.random.randint(0, A, size=(W, N)).astype(np.float32) for _ in range(X)]
fake_rewards = [np.random.rand(W, N).astype(np.float32) for _ in range(X)]

# %% [markdown]
# ## Dynamic List Implementation

# %%
def dynamic_list_sim():    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    start = perf_counter()

    buf_obs, buf_global_obs, buf_actions, buf_rewards = [], [], [], []

    for t in range(X):
        buf_obs.append(torch.from_numpy(fake_obs[t]).to(device))
        buf_global_obs.append(torch.from_numpy(fake_global[t]).to(device))
        buf_actions.append(torch.from_numpy(fake_actions[t]).to(device))
        buf_rewards.append(torch.from_numpy(fake_rewards[t]).to(device))
        
    final_obs = torch.stack(buf_obs)
    final_global = torch.stack(buf_global_obs)
    final_actions = torch.stack(buf_actions)
    final_rewards = torch.stack(buf_rewards)

    torch.cuda.synchronize()
    dyn_time = perf_counter() - start
    dyn_ram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    return dyn_time, dyn_ram_mb

# %% [markdown]
# ## Preallocated Buffer

# %%
def preallocate_buffer_sim():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    start = perf_counter()

    buf_obs = torch.zeros((X, W, N, O), dtype=torch.float32, device=device)
    buf_global_obs = torch.zeros((X, W, GO), dtype=torch.float32, device=device)
    buf_actions = torch.zeros((X, W, N), dtype=torch.float32, device=device)
    buf_rewards = torch.zeros((X, W, N), dtype=torch.float32, device=device)

    for t in range(X):
        buf_obs[t] = torch.from_numpy(fake_obs[t]).to(device)
        buf_global_obs[t] = torch.from_numpy(fake_global[t]).to(device)
        buf_actions[t] = torch.from_numpy(fake_actions[t]).to(device)
        buf_rewards[t] = torch.from_numpy(fake_rewards[t]).to(device)
        
    torch.cuda.synchronize()
    pre_time = perf_counter() - start
    pre_ram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    return pre_time, pre_ram_mb

# %%
dyn_times, dyn_ram_mbs = [], []
pre_times, pre_ram_mbs = [], []

for i in tqdm(range(10)):

    dyn_time, dyn_ram_mb = dynamic_list_sim()
    dyn_times.append(dyn_time)
    dyn_ram_mbs.append(dyn_ram_mb)
    gc.collect()
    
    pre_time, pre_ram_mb = preallocate_buffer_sim()
    pre_times.append(pre_time)
    pre_ram_mbs.append(pre_ram_mb)
    gc.collect()

# %%
print("Dynamic Lists:")
print(f"  TIME: {np.mean(dyn_times):.2f} +- {np.std(dyn_times)}")
print(f"  RAM : {np.mean(dyn_ram_mbs):.2f} +- {np.std(dyn_ram_mbs)}")

print("Preallocated Lists:")
print(f"  TIME: {np.mean(pre_times):.2f} +- {np.std(pre_times)}")
print(f"  RAM : {np.mean(pre_ram_mbs):.2f} +- {np.std(pre_ram_mbs)}")

# %%
time_reduction = ((dyn_time - pre_time) / dyn_time) * 100
ram_reduction = ((dyn_ram_mb - pre_ram_mb) / dyn_ram_mb) * 100

print(f"Time Reduction: {time_reduction:.1f}%")
print(f"RAM Reduction: {ram_reduction:.1f}%")


