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
CODEBOOK = 0

# %%
system, config = setup(config, device, load_agent_architecture=True)
# config.training.simulation_timesteps = 15

# %%
actor = system["actor"]
actor.eval()
print("Loaded")

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
V = config.comms.vocab_size
possible_targets = config.simulation.arena_width * config.simulation.arena_height
num_inventory_states = 4

temporal_target_matrix = np.zeros((T, V, possible_targets))
inventory_matrix = np.zeros((T, V, num_inventory_states))
transition_matrix = np.zeros((V, V))

def get_inventory_state(obs_tensor):

    collected_0 = obs_tensor[..., 13] # Holding Ingridient 1
    collected_1 = obs_tensor[..., 25] # Holding Ingridient 2
    collected_2 = obs_tensor[..., 37] # Holding Ingridient 3
    
    states = torch.zeros_like(collected_0, dtype=torch.long)
    states[collected_0 == 1.0] = 1
    states[collected_1 == 1.0] = 2
    states[collected_2 == 1.0] = 3
    
    return states.cpu().numpy()


for batch_obs, batch_global_obs, batch_actions, batch_targets, batch_critic_values, batch_return_values in dataloader:
    
    batch_obs = batch_obs.to(device)
    batch_targets = batch_targets.to(device).float()
    
    B = batch_obs.shape[0]
    NC = config.comms.num_comms
    C = config.comms.communication_size
    
    actor_hidden_states = actor.init_hidden(batch_size=B)
    comms = torch.zeros((B, T, N, NC, C), dtype=torch.float32, device=device)
    
    with torch.no_grad():
        action_logits, lstm_output, _ = actor(batch_obs, actor_hidden_states, comms)

    lstm_output_flat = lstm_output.contiguous().view(B * T, 1, N, -1)
    _, to_save, _ = actor.comm_protocol.get_comms_during_rollout(lstm_output_flat)
    
    code_indices = to_save[..., CODEBOOK].view(B, T, N).cpu().numpy().astype(int)

    disc_x = np.round(batch_targets[..., 0].cpu().numpy() * config.simulation.arena_width).astype(int)
    disc_y = np.round(batch_targets[..., 1].cpu().numpy() * config.simulation.arena_height).astype(int)
    
    target_ids = (disc_y * config.simulation.arena_width) + disc_x
    other_agent_target_ids = np.roll(target_ids, shift=1, axis=2)

    for t in range(T):
        codes_t = code_indices[:, t, :].flatten()
        targets_t = other_agent_target_ids[:, t, :].flatten()
        
        for code, target in zip(codes_t, targets_t):
            temporal_target_matrix[t, code, target] += 1
            
        inv_states_t = get_inventory_state(batch_obs[:, t, :, :]).flatten()
        for code, inv in zip(codes_t, inv_states_t):
            inventory_matrix[t, code, inv] += 1
            
        if t < T - 1:
            codes_next = code_indices[:, t + 1, :].flatten()
            for c_curr, c_next in zip(codes_t, codes_next):
                transition_matrix[c_curr, c_next] += 1

global_code_usage = temporal_target_matrix.sum(axis=(0, 2))
active_codes = np.where(global_code_usage > 100)[0]
num_active = len(active_codes)


# %%
# fig, ax2 = plt.subplots(figsize=(8, 6))

# # Do next partner token|curr token

# active_transitions = transition_matrix[np.ix_(active_codes, active_codes)]
# row_sums = active_transitions.sum(axis=1, keepdims=True)
# active_transitions_prob = np.divide(
#     active_transitions,
#     row_sums,
#     out=np.zeros_like(active_transitions, dtype=float),
#     where=row_sums != 0
# )

# im2 = ax2.imshow(active_transitions_prob, cmap='Blues', aspect='auto')
# ax2.set_title("Token Transition Probabilities (P(Next | Curr))", fontsize=14, fontweight="bold")
# ax2.set_xticks(range(num_active))
# ax2.set_yticks(range(num_active))
# ax2.set_xticklabels([str(c) for c in active_codes])
# ax2.set_yticklabels([str(c) for c in active_codes])
# ax2.set_xlabel("Next Token")
# ax2.set_ylabel("Current Token")

# fig.colorbar(im2, ax=ax2)
# plt.tight_layout()
# plt.show()

# %%
# partner_transition_matrix = np.zeros((V, V))

# partner_code_indices = np.roll(code_indices, shift=1, axis=2)

# for t in range(T - 1):
#     codes_t = code_indices[:, t, :].flatten()
#     partner_codes_next = partner_code_indices[:, t + 1, :].flatten()

#     for c_curr, c_next in zip(codes_t, partner_codes_next):
#         partner_transition_matrix[c_curr, c_next] += 1

# active_transitions = partner_transition_matrix[np.ix_(active_codes, active_codes)]

# row_sums = active_transitions.sum(axis=1, keepdims=True)
# active_transitions_prob = np.divide(
#     active_transitions,
#     row_sums,
#     out=np.zeros_like(active_transitions, dtype=float),
#     where=row_sums != 0
# )

# fig, ax = plt.subplots(figsize=(8, 6))
# im = ax.imshow(active_transitions_prob, cmap="Blues", aspect="auto", vmin=0, vmax=0.4)

# ax.set_title("P(Partner Next Token | Emitted Token)", fontsize=14, fontweight="bold")
# ax.set_xticks(range(num_active))
# ax.set_yticks(range(num_active))
# ax.set_xticklabels([str(c) for c in active_codes])
# ax.set_yticklabels([str(c) for c in active_codes])
# ax.set_xlabel("Partner Next Token")
# ax.set_ylabel("Emitted Token")

# fig.colorbar(im, ax=ax, label="Probability")
# plt.tight_layout()
# plt.show()

# %%
plt.figure(figsize=(10, 2.0))

num_active = len(active_codes)

inv_filtered = inventory_matrix.sum(axis=0)[active_codes, :]
inv_totals = inv_filtered.sum(axis=1, keepdims=True)
inv_probs = np.divide(
    inv_filtered,
    inv_totals,
    out=np.zeros_like(inv_filtered, dtype=float),
    where=inv_totals != 0
)

im3 = plt.imshow(inv_probs.T, cmap='Purples', aspect='auto')
plt.title("Token vs. Agent Inventory State", fontsize=14, fontweight="bold")
plt.xticks(range(num_active), [str(c) for c in active_codes])
plt.yticks(
    range(num_inventory_states),
    ["Empty", "Has Ingredient 1", "Has Ingredient 2", "Has Product"]
)
plt.xlabel("Token Code")
plt.ylabel("Inventory State")

plt.grid(False)

# fig.colorbar(im3)
plt.tight_layout()

# plt.savefig(FIGURES_DIR / "token-given-inventory.png", dpi=600)
# plt.savefig(FIGURES_DIR / "token-given-inventory-better.png", dpi=600)

plt.show()


