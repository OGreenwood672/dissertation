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


from notebooks.plt_style import set_style
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

set_style()

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %%
config = Config.from_yaml(PROJECT_ROOT / "configs")
assert config.training.seed != -1
# assert config.comms.communication_type.value == "AIM", f"Comm type is {config.comms.communication_type}, should be aim"
# assert config.comms.autoencoder_type != "hq-vae"

# %%
system, config = setup(config, device, load_agent_architecture=True)
# config.training.simulation_timesteps = 4

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
co_occurrence_matrix = np.zeros((config.comms.vocab_size, 5))

for batch_obs, batch_global_obs, batch_actions, batch_targets, batch_critic_values, batch_return_values in dataloader:

    batch_obs = batch_obs.to(device)
    batch_actions = batch_actions.to(device).long()

    B = batch_obs.shape[0]
    O = batch_obs.shape[3]
    NC = config.comms.num_comms
    C = config.comms.communication_size
    
    actor_hidden_states = actor.init_hidden(batch_size=B)
    
    comms = torch.zeros((B, T, N, NC, C), dtype=torch.float32, device=device)
    
    _, lstm_output, _ = actor(batch_obs, actor_hidden_states, comms)

    lstm_output_flat = lstm_output.contiguous().view(B * T, 1, N, -1)
    _, to_save, _ = actor.comm_protocol.get_comms_during_rollout(lstm_output_flat)
    
    code_indicies = to_save[..., 0].flatten().cpu().numpy().astype(int)
    actions = batch_actions.flatten().cpu().numpy().astype(int)
    
    for code, action in zip(code_indicies, actions):
        co_occurrence_matrix[code, action] += 1

# %%
code_freqs = co_occurrence_matrix.sum(axis=1)
top_codes = np.sort(np.argsort(code_freqs)[-12:])
co_occurrence_matrix = co_occurrence_matrix[top_codes]

row_sums = co_occurrence_matrix.sum(axis=1, keepdims=True)
normalised_matrix = np.divide(
    co_occurrence_matrix,
    row_sums,
    out=np.zeros_like(co_occurrence_matrix),
    where=row_sums != 0
)

actions = {
    0: "Step North",
    1: "Step South",
    2: "Step West",
    3: "Step East",
    4: "Interact"
}
action_labels = [actions[i] for i in range(5)]
code_labels = [f"{i}" for i in top_codes]

df = pd.DataFrame(normalised_matrix, index=code_labels, columns=action_labels)

# %%
plt.figure(figsize=(7, 4))
ax = sns.heatmap(
    df.T,
    cmap="Blues",
    fmt=".2f",
    cbar_kws={"label": "P(Action | Code)", "shrink": 0.2, "pad": 0.02}
)

ax.set_aspect(0.3)

plt.title("Broadcasted AIM Code vs. Action Taken", fontsize=5, pad=2)
plt.xlabel("AIM Code", fontsize=4)
plt.ylabel("Action Taken", fontsize=4)

ax.tick_params(axis="x", labelsize=4, rotation=0, pad=0.2)
ax.tick_params(axis="y", labelsize=3, rotation=0, pad=0.2)

cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=5)
cbar.ax.yaxis.label.set_size(4)

plt.tight_layout()
# plt.savefig(
#     FIGURES_DIR / "own-action-given-code.png",
#     dpi=900,
#     bbox_inches="tight",
#     pad_inches=0.02
# )
plt.show()

# %%
co_occurrence_matrix = np.zeros((config.comms.vocab_size, 5))

for batch_obs, batch_global_obs, batch_actions, batch_targets, batch_critic_values, batch_return_values in dataloader:
    
    batch_obs = batch_obs.to(device)
    batch_actions = batch_actions.to(device).long()

    B = batch_obs.shape[0]
    T = batch_obs.shape[1]
    N = batch_obs.shape[2]
    NC = config.comms.num_comms
    C = config.comms.communication_size
    
    actor_hidden_states = actor.init_hidden(batch_size=B)
    comms = torch.zeros((B, T, N, NC, C), dtype=torch.float32, device=device)
    
    action_logits, lstm_output, _ = actor(batch_obs, actor_hidden_states, comms)

    lstm_output_flat = lstm_output.contiguous().view(B * T, 1, N, -1)
    _, to_save, _ = actor.comm_protocol.get_comms_during_rollout(lstm_output_flat)
    
    codes = to_save.view(B, T, N, -1)[..., 0] 

    codes = codes[:, :-1, :] 
    actions_plus_1 = batch_actions[:, 1:, :] 
    
    other_actions_t_plus_1 = torch.roll(actions_plus_1, shifts=1, dims=2)
    
    codes_flat = codes.flatten().cpu().numpy().astype(int)
    other_actions_flat = other_actions_t_plus_1.flatten().cpu().numpy().astype(int)
    
    for code, other_action in zip(codes_flat, other_actions_flat):
        co_occurrence_matrix[code, other_action] += 1

code_freqs = co_occurrence_matrix.sum(axis=1)
top_codes = np.sort(np.argsort(code_freqs)[-12:])
co_occurrence_matrix = co_occurrence_matrix[top_codes]

row_sums = co_occurrence_matrix.sum(axis=1, keepdims=True)
normalised_matrix = np.divide(
    co_occurrence_matrix,
    row_sums,
    out=np.zeros_like(co_occurrence_matrix),
    where=row_sums != 0
)

actions_dict = {
    0: "Step North",
    1: "Step South",
    2: "Step West",
    3: "Step East",
    4: "Interact"
}

action_labels = [actions_dict[i] for i in range(5)] 
code_labels = [f"Code {i}" for i in top_codes]

df = pd.DataFrame(normalised_matrix, index=code_labels, columns=action_labels)

# %%
plt.figure(figsize=(7, 2.5))

ax = sns.heatmap(
    df.T, cmap="YlGnBu", vmin=0.0, vmax=1.0, fmt=".2f", annot_kws={"size": 10}, cbar_kws={"label": "P(Other Agent's Action | Code)", "pad": 0.02}
)

plt.title(f"P(Other Agent's Action | Emitted Code)", fontsize=11, pad=3.5)
plt.xlabel("Emitted Code (Vocabulary)", fontsize=8, fontweight='bold')
plt.ylabel("Other Agent's Action", fontsize=8, fontweight='bold')
plt.yticks(fontsize=6, rotation=0) 
plt.xticks(fontsize=6, rotation=0)

cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=5)
cbar.ax.yaxis.label.set_size(2)
cbar.set_label("P(Other Agent's Action | Code)", fontsize=7)

plt.tight_layout()
# plt.savefig(
#     FIGURES_DIR / "partners-action-given-code.png",
#     dpi=900,
#     bbox_inches="tight",
#     pad_inches=0.02
# )

plt.show()


