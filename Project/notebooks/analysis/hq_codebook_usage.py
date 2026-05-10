# %%
# Imports
import sys
import os
sys.path.append(os.path.abspath('../..'))


from controller.marl.main import setup
from controller.marl.core.config import Config
from controller.marl.runners.sim_runner import run_sim


from project_paths import PROJECT_ROOT, FIGURES_DIR


import torch
from controller.marl.core.datasets import ObsData
from torch.utils.data import DataLoader

from scipy.optimize import curve_fit

from notebooks.plt_style import set_style
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm

set_style()

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %%
config = Config.from_yaml(PROJECT_ROOT / "configs")
assert config.training.seed != -1
# assert config.comms.communication_type.value == "AIM", f"Comm type is {config.comms.communication_type}, should be aim"
# assert config.comms.autoencoder_type != "hq-vae"
CODEBOOK = 0

# %%
system, config = setup(config, device, load_agent_architecture=True)
config.training.simulation_timesteps = 30

# %%
actor = system["actor"]
actor.eval()
print("loaded")

# %%
obs_logs_file = "./temp.csv"

# %%
run_sim(system, config, device, 3, collect_obs_file=obs_logs_file)

# %%
GO = system["sim"].get_global_obs_dim()
T = config.training.simulation_timesteps
W = config.training.worlds_parallised
N = system["sim"].get_num_agents()
mask = torch.tensor(system["sim"].get_agent_external_obs_mask(0), dtype=torch.bool, device=device)
dataset = ObsData(obs_logs_file, GO, T, W, N, device)

dataloader = DataLoader(dataset, batch_size=config.aim_training.aim_batch_size)

# %%
frequencies = np.zeros((config.comms.vocab_size * config.comms.vocab_size))

for batch_obs, batch_global_obs, batch_actions, batch_targets, batch_critic_values, batch_return_values in dataloader:
    
    batch_obs = batch_obs.to(device)
        
    B = batch_obs.shape[0]
    T = batch_obs.shape[1]
    N = batch_obs.shape[2]
    O = batch_obs.shape[3]
    NC = config.comms.num_comms
    C = config.comms.communication_size
    
    actor_hidden_states = actor.init_hidden(batch_size=B)
    
    comms = torch.zeros((B, T, N, NC, C), dtype=torch.float32, device=device)
    
    action_logits, lstm_output, _ = actor(batch_obs, actor_hidden_states, comms)

    lstm_output_flat = lstm_output.contiguous().view(B * T, 1, N, -1)
    _, to_save, _ = actor.comm_protocol.get_comms_during_rollout(lstm_output_flat)
    
    codes = to_save[..., 0].flatten().cpu().numpy().astype(int), to_save[..., 1].flatten().cpu().numpy().astype(int)
    combined_codes = codes[0] * config.comms.vocab_size + codes[1]
    
    for code in combined_codes:
        frequencies[code] += 1

# %%
frequencies_normalised = frequencies / sum(frequencies)
code_indices = np.arange(len(frequencies_normalised))

plt.figure(figsize=(12, 5))
plt.bar(code_indices, frequencies_normalised, color="royalblue", alpha=0.8, width=0.9)

code_labels = list(map(lambda x: str(x) if x % config.comms.vocab_size == 0 else "", code_indices))


plt.title(f"Codebook Usage Histogram (VQ-VAE)", fontsize=14, pad=15)
plt.xlabel("Codebook Index", fontsize=12)
plt.ylabel("Frequency of Use", fontsize=12)
plt.xlim(-0.5, len(frequencies_normalised) - 0.5)
plt.xticks(code_indices, code_labels)

plt.grid(axis="y", alpha=0.3)
sns.despine()
plt.tight_layout()

# if input("Save frequency img (y/N): ") == "y":
#     plt.savefig(FIGURES_DIR / "vocab-frequency.png")
plt.show()



# %%
entropy = -np.sum(frequencies_normalised[frequencies_normalised > 0] * np.log2(frequencies_normalised[frequencies_normalised > 0]))
gini = 1 - np.sum(frequencies_normalised**2)

print("Entropy:", entropy)
print("Gini:", gini)


# %%
CODE = 12
frequencies_given = frequencies[CODE * config.comms.vocab_size:(CODE + 1) * config.comms.vocab_size]
frequencies_normalised = frequencies_given / sum(frequencies_given)
code_indices = np.arange(len(frequencies_normalised))

plt.figure(figsize=(12, 5))
plt.bar(code_indices, frequencies_normalised, color="royalblue", alpha=0.8, width=0.9)

plt.title(f"Codebook Usage Histogram (VQ-VAE)", fontsize=14, pad=15)
plt.xlabel("Codebook Index", fontsize=12)
plt.ylabel("Frequency of Use", fontsize=12)
plt.xlim(-0.5, len(frequencies_normalised) - 0.5)
plt.xticks(code_indices)

plt.grid(axis="y", alpha=0.3)
sns.despine()
plt.tight_layout()

# if input("Save frequency img (y/N): ") == "y":
#     plt.savefig(FIGURES_DIR / "vocab-frequency.png")
plt.show()


