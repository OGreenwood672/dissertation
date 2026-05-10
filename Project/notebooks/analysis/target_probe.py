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
import torch.nn as nn
import torch.optim as optim
from controller.marl.core.datasets import ObsData
from controller.marl.core.metric_tracker import MetricTracker
from torch.utils.data import DataLoader


from notebooks.plt_style import set_style
import matplotlib.pyplot as plt
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


# %%
actor = system["actor"]
actor.eval()
print("Loaded")

# %%
obs_logs_file = "./temp.csv"

# %%
run_sim(system, config, device, 5, collect_obs_file=obs_logs_file)

# %%
GO = system["sim"].get_global_obs_dim()
T = config.training.simulation_timesteps
W = config.training.worlds_parallised
N = system["sim"].get_num_agents()
mask = torch.tensor(system["sim"].get_agent_external_obs_mask(0), dtype=torch.bool, device=device)
dataset = ObsData(obs_logs_file, GO, T, W, N, device)

train_set, validation_set, test_set = torch.utils.data.random_split(dataset, [0.75, 0.1, 0.15])

print(f"Using {len(train_set)} datapoints for training...")

training_loader, validation_loader, test_loader = (
    DataLoader(train_set, batch_size=config.aim_training.aim_batch_size, shuffle=False),
    DataLoader(validation_set, batch_size=config.aim_training.aim_batch_size, shuffle=False),
    DataLoader(test_set, batch_size=config.aim_training.aim_batch_size, shuffle=False)
)


# %%
class LSTMProbeLocation(nn.Module):
    def __init__(self, lstm_hidden_dim):
        super().__init__()
        # self.net = nn.Sequential(
        #     nn.Linear(lstm_hidden_dim, 256),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(),
        #     nn.Linear(256, 128),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(),
        #     nn.Linear(128, 2)
        # )
        self.net = nn.Sequential(
            nn.Linear(lstm_hidden_dim, 2),
        )
    def forward(self, lstm):
        return self.net(lstm)

class LSTMProbeRelativeDistance(nn.Module):
    def __init__(self, lstm_hidden_dim):
        super().__init__()
        # self.net = nn.Sequential(
        #     nn.Linear(lstm_hidden_dim, 256),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(),
        #     nn.Linear(256, 128),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(),
        #     nn.Linear(128, 2)
        # )
        self.net = nn.Sequential(
            nn.Linear(lstm_hidden_dim, 2),
        )
    def forward(self, lstm):
        return self.net(lstm)

class LSTMProbeDirection(nn.Module):
    def __init__(self, lstm_hidden_dim):
        super().__init__()
        # self.net = nn.Sequential(
        #     nn.Linear(lstm_hidden_dim, 256),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(),
        #     nn.Linear(256, 128),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(),
        #     nn.Linear(128, 2)
        # )
        self.net = nn.Sequential(
            nn.Linear(lstm_hidden_dim, 2),
        )
    def forward(self, lstm):
        return self.net(lstm)

# %%
self_probe_loc = LSTMProbeLocation(config.actor.lstm_hidden_size).to(device)
self_probe_rel = LSTMProbeRelativeDistance(config.actor.lstm_hidden_size).to(device)
self_probe_dir = LSTMProbeDirection(config.actor.lstm_hidden_size).to(device)

other_probe_loc = LSTMProbeLocation(config.actor.lstm_hidden_size).to(device)
other_probe_rel = LSTMProbeRelativeDistance(config.actor.lstm_hidden_size).to(device)
other_probe_dir = LSTMProbeDirection(config.actor.lstm_hidden_size).to(device)

optimiser_self_loc = optim.Adam(self_probe_loc.parameters(), lr=1e-3)
optimiser_self_rel = optim.Adam(self_probe_rel.parameters(), lr=1e-3)
optimiser_self_dir = optim.Adam(self_probe_dir.parameters(), lr=1e-3)

optimiser_other_loc = optim.Adam(other_probe_loc.parameters(), lr=1e-3)
optimiser_other_rel = optim.Adam(other_probe_rel.parameters(), lr=1e-3)
optimiser_other_dir = optim.Adam(other_probe_dir.parameters(), lr=1e-3)

criterion = nn.MSELoss()
tracker = MetricTracker()

# %%
epochs = 20
LAST_T = 100

for epoch in range(epochs):
    self_probe_loc.train()
    self_probe_rel.train()
    self_probe_dir.train()
    other_probe_loc.train()
    other_probe_rel.train()
    other_probe_dir.train()

    for batch_obs, _, _, batch_targets, _, _ in training_loader:
        batch_obs = batch_obs.to(device)
        batch_targets = batch_targets.to(device).float()
        
        B = batch_obs.shape[0]
        NC = config.comms.num_comms
        C = config.comms.communication_size
        
        tx0 = batch_targets[:, :, 0, 0] * config.simulation.arena_width
        ty0 = batch_targets[:, :, 0, 1] * config.simulation.arena_height
        tx1 = batch_targets[:, :, 1, 0] * config.simulation.arena_width
        ty1 = batch_targets[:, :, 1, 1] * config.simulation.arena_height

        target_loc_0 = torch.stack([tx0, ty0], dim=-1)
        target_loc_1 = torch.stack([tx1, ty1], dim=-1)

        disc_0_x = batch_obs[:, :, 0, 0] * config.simulation.arena_width
        disc_0_y = batch_obs[:, :, 0, 1] * config.simulation.arena_height

        dx0 = tx0 - disc_0_x
        dy0 = ty0 - disc_0_y
        dx1 = tx1 - disc_0_x
        dy1 = ty1 - disc_0_y
        target_rel_0 = torch.stack([dx0, dy0], dim=-1)
        target_rel_1 = torch.stack([dx1, dy1], dim=-1)

        mag0 = torch.sqrt(dx0**2 + dy0**2 + 1e-8)
        mag1 = torch.sqrt(dx1**2 + dy1**2 + 1e-8)
        ux0 = dx0 / mag0
        uy0 = dy0 / mag0
        ux1 = dx1 / mag1
        uy1 = dy1 / mag1
        target_dir_0 = torch.stack([ux0, uy0], dim=-1)
        target_dir_1 = torch.stack([ux1, uy1], dim=-1)

        actor_hidden_states = actor.init_hidden(batch_size=B)
        current_comms = torch.zeros((B, 1, N, NC, C), dtype=torch.float32, device=device)
        agent_0_lstm_states = []
        
        with torch.no_grad():
            for t in range(T):
                obs_t = batch_obs[:, t:t+1, :, :]
                _, lstm_output, actor_hidden_states = actor(obs_t, actor_hidden_states, current_comms)
                agent_0_lstm_states.append(lstm_output[:, 0, 0, :])
                comms_next, _, _ = actor.comm_protocol.get_comms_during_rollout(lstm_output)
                current_comms = comms_next.unsqueeze(1).detach() if comms_next.ndim == 4 else comms_next.detach()

        agent_0_lstm = torch.stack(agent_0_lstm_states, dim=1)

        late_cutoff = max(0, T - LAST_T)
        lstm_late = agent_0_lstm[:, late_cutoff:, :].reshape(B * (T - late_cutoff), -1)

        lbl_loc_0 = target_loc_0[:, late_cutoff:, :].reshape(-1, 2)
        lbl_loc_1 = target_loc_1[:, late_cutoff:, :].reshape(-1, 2)
        lbl_rel_0 = target_rel_0[:, late_cutoff:, :].reshape(-1, 2)
        lbl_rel_1 = target_rel_1[:, late_cutoff:, :].reshape(-1, 2)
        lbl_dir_0 = target_dir_0[:, late_cutoff:, :].reshape(-1, 2)
        lbl_dir_1 = target_dir_1[:, late_cutoff:, :].reshape(-1, 2)

        optimiser_self_loc.zero_grad()
        loss_s_loc = criterion(self_probe_loc(lstm_late), lbl_loc_0)
        loss_s_loc.backward()
        optimiser_self_loc.step()

        optimiser_other_loc.zero_grad()
        loss_o_loc = criterion(other_probe_loc(lstm_late), lbl_loc_1)
        loss_o_loc.backward()
        optimiser_other_loc.step()

        optimiser_self_rel.zero_grad()
        loss_s_rel = criterion(self_probe_rel(lstm_late), lbl_rel_0)
        loss_s_rel.backward()
        optimiser_self_rel.step()

        optimiser_other_rel.zero_grad()
        loss_o_rel = criterion(other_probe_rel(lstm_late), lbl_rel_1)
        loss_o_rel.backward()
        optimiser_other_rel.step()

        optimiser_self_dir.zero_grad()
        loss_s_dir = criterion(self_probe_dir(lstm_late), lbl_dir_0)
        loss_s_dir.backward()
        optimiser_self_dir.step()

        optimiser_other_dir.zero_grad()
        loss_o_dir = criterion(other_probe_dir(lstm_late), lbl_dir_1)
        loss_o_dir.backward()
        optimiser_other_dir.step()

    self_probe_loc.eval()
    self_probe_rel.eval()
    self_probe_dir.eval()
    other_probe_loc.eval()
    other_probe_rel.eval()
    other_probe_dir.eval()

    with torch.no_grad():
        for batch_obs, _, _, batch_targets, _, _ in validation_loader:
            batch_obs = batch_obs.to(device)
            batch_targets = batch_targets.to(device).float()
            
            B = batch_obs.shape[0]
            
            tx0 = batch_targets[:, :, 0, 0] * config.simulation.arena_width
            ty0 = batch_targets[:, :, 0, 1] * config.simulation.arena_height
            tx1 = batch_targets[:, :, 1, 0] * config.simulation.arena_width
            ty1 = batch_targets[:, :, 1, 1] * config.simulation.arena_height

            target_loc_0 = torch.stack([tx0, ty0], dim=-1)
            target_loc_1 = torch.stack([tx1, ty1], dim=-1)

            disc_0_x = batch_obs[:, :, 0, 0] * config.simulation.arena_width
            disc_0_y = batch_obs[:, :, 0, 1] * config.simulation.arena_height

            dx0 = tx0 - disc_0_x
            dy0 = ty0 - disc_0_y
            dx1 = tx1 - disc_0_x
            dy1 = ty1 - disc_0_y
            target_rel_0 = torch.stack([dx0, dy0], dim=-1)
            target_rel_1 = torch.stack([dx1, dy1], dim=-1)

            mag0 = torch.sqrt(dx0**2 + dy0**2 + 1e-8)
            mag1 = torch.sqrt(dx1**2 + dy1**2 + 1e-8)
            ux0 = dx0 / mag0
            uy0 = dy0 / mag0
            ux1 = dx1 / mag1
            uy1 = dy1 / mag1
            target_dir_0 = torch.stack([ux0, uy0], dim=-1)
            target_dir_1 = torch.stack([ux1, uy1], dim=-1)

            actor_hidden_states = actor.init_hidden(batch_size=B)
            current_comms = torch.zeros((B, 1, N, config.comms.num_comms, config.comms.communication_size), dtype=torch.float32, device=device)
            agent_0_lstm_states = []
            
            for t in range(T):
                obs_t = batch_obs[:, t:t+1, :, :]
                _, lstm_output, actor_hidden_states = actor(obs_t, actor_hidden_states, current_comms)
                agent_0_lstm_states.append(lstm_output[:, 0, 0, :])
                comms_next, _, _ = actor.comm_protocol.get_comms_during_rollout(lstm_output)
                current_comms = comms_next.unsqueeze(1) if comms_next.ndim == 4 else comms_next

            agent_0_lstm = torch.stack(agent_0_lstm_states, dim=1).reshape(B * T, -1)
            
            tracker.update("validation_self_loc", criterion(self_probe_loc(agent_0_lstm), target_loc_0.reshape(-1, 2)).item())
            tracker.update("validation_other_loc", criterion(other_probe_loc(agent_0_lstm), target_loc_1.reshape(-1, 2)).item())
            tracker.update("validation_self_rel", criterion(self_probe_rel(agent_0_lstm), target_rel_0.reshape(-1, 2)).item())
            tracker.update("validation_other_rel", criterion(other_probe_rel(agent_0_lstm), target_rel_1.reshape(-1, 2)).item())
            tracker.update("validation_self_dir", criterion(self_probe_dir(agent_0_lstm), target_dir_0.reshape(-1, 2)).item())
            tracker.update("validation_other_dir", criterion(other_probe_dir(agent_0_lstm), target_dir_1.reshape(-1, 2)).item())

    print(f"Epoch {epoch+1}/{epochs}")
    print(f"  Self  | Loc: {tracker.get_average('validation_self_loc'):.3f} | Rel: {tracker.get_average('validation_self_rel'):.3f} | Dir: {tracker.get_average('validation_self_dir'):.3f}")
    print(f"  Other | Loc: {tracker.get_average('validation_other_loc'):.3f} | Rel: {tracker.get_average('validation_other_rel'):.3f} | Dir: {tracker.get_average('validation_other_dir'):.3f}")
    tracker.reset_tracker()

# %%
self_error_loc = np.zeros(T)
other_error_loc = np.zeros(T)
self_error_rel = np.zeros(T)
other_error_rel = np.zeros(T)
self_error_dir = np.zeros(T)
other_error_dir = np.zeros(T)
counts_time = np.zeros(T)

with torch.no_grad():
    for batch_obs, _, _, batch_targets, _, _ in test_loader:
        batch_obs = batch_obs.to(device)
        batch_targets = batch_targets.to(device).float()

        B = batch_obs.shape[0]
            
        tx0 = batch_targets[:, :, 0, 0] * config.simulation.arena_width
        ty0 = batch_targets[:, :, 0, 1] * config.simulation.arena_height
        tx1 = batch_targets[:, :, 1, 0] * config.simulation.arena_width
        ty1 = batch_targets[:, :, 1, 1] * config.simulation.arena_height

        target_loc_0 = torch.stack([tx0, ty0], dim=-1)
        target_loc_1 = torch.stack([tx1, ty1], dim=-1)

        disc_0_x = batch_obs[:, :, 0, 0] * config.simulation.arena_width
        disc_0_y = batch_obs[:, :, 0, 1] * config.simulation.arena_height

        dx0 = tx0 - disc_0_x
        dy0 = ty0 - disc_0_y
        dx1 = tx1 - disc_0_x
        dy1 = ty1 - disc_0_y
        target_rel_0 = torch.stack([dx0, dy0], dim=-1)
        target_rel_1 = torch.stack([dx1, dy1], dim=-1)

        mag0 = torch.sqrt(dx0**2 + dy0**2 + 1e-8)
        mag1 = torch.sqrt(dx1**2 + dy1**2 + 1e-8)
        ux0 = dx0 / mag0
        uy0 = dy0 / mag0
        ux1 = dx1 / mag1
        uy1 = dy1 / mag1
        target_dir_0 = torch.stack([ux0, uy0], dim=-1)
        target_dir_1 = torch.stack([ux1, uy1], dim=-1)

        actor_hidden_states = actor.init_hidden(batch_size=B)
        current_comms = torch.zeros((B, 1, N, config.comms.num_comms, config.comms.communication_size), dtype=torch.float32, device=device)

        for t in range(T):
            obs_t = batch_obs[:, t:t+1, :, :]
            _, lstm_output, actor_hidden_states = actor(obs_t, actor_hidden_states, current_comms)
            agent_0_lstm = lstm_output[:, 0, 0, :]

            comms_next, _, _ = actor.comm_protocol.get_comms_during_rollout(lstm_output)
            current_comms = comms_next.unsqueeze(1) if comms_next.ndim == 4 else comms_next

            p_s_loc = self_probe_loc(agent_0_lstm)
            p_o_loc = other_probe_loc(agent_0_lstm)
            p_s_rel = self_probe_rel(agent_0_lstm)
            p_o_rel = other_probe_rel(agent_0_lstm)
            p_s_dir = self_probe_dir(agent_0_lstm)
            p_o_dir = other_probe_dir(agent_0_lstm)

            self_error_loc[t] += torch.norm(p_s_loc - target_loc_0[:, t, :], dim=1).sum().item()
            other_error_loc[t] += torch.norm(p_o_loc - target_loc_1[:, t, :], dim=1).sum().item()
            self_error_rel[t] += torch.norm(p_s_rel - target_rel_0[:, t, :], dim=1).sum().item()
            other_error_rel[t] += torch.norm(p_o_rel - target_rel_1[:, t, :], dim=1).sum().item()
            self_error_dir[t] += torch.norm(p_s_dir - target_dir_0[:, t, :], dim=1).sum().item()
            other_error_dir[t] += torch.norm(p_o_dir - target_dir_1[:, t, :], dim=1).sum().item()
            counts_time[t] += B

self_error_loc = np.divide(self_error_loc, counts_time, out=np.zeros_like(self_error_loc), where=counts_time!=0)
other_error_loc = np.divide(other_error_loc, counts_time, out=np.zeros_like(other_error_loc), where=counts_time!=0)
self_error_rel = np.divide(self_error_rel, counts_time, out=np.zeros_like(self_error_rel), where=counts_time!=0)
other_error_rel = np.divide(other_error_rel, counts_time, out=np.zeros_like(other_error_rel), where=counts_time!=0)
self_error_dir = np.divide(self_error_dir, counts_time, out=np.zeros_like(self_error_dir), where=counts_time!=0)
other_error_dir = np.divide(other_error_dir, counts_time, out=np.zeros_like(other_error_dir), where=counts_time!=0)

# %%
def smooth(arr, window=5):
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode='same')

self_error_loc = smooth(self_error_loc, window=10)
other_error_loc = smooth(other_error_loc, window=10)
self_error_rel = smooth(self_error_rel, window=10)
other_error_rel = smooth(other_error_rel, window=10)
self_error_dir = smooth(self_error_dir, window=10)
other_error_dir = smooth(other_error_dir, window=10)


# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

CUTOFF = 5

axes[0].plot(range(CUTOFF, T - CUTOFF), other_error_loc[CUTOFF:T - CUTOFF], label="Partner", linewidth=2)
axes[0].plot(range(CUTOFF, T - CUTOFF), self_error_loc[CUTOFF:T - CUTOFF], label="Self", linewidth=2)
axes[0].set_title("Absolute Location")
axes[0].set_xlabel("Simulation Step")
axes[0].set_ylabel("Error")
axes[0].grid(alpha=0.3)
axes[0].legend()

axes[1].plot(range(CUTOFF, T - CUTOFF), other_error_rel[CUTOFF:T - CUTOFF], label="Partner", linewidth=2)
axes[1].plot(range(CUTOFF, T - CUTOFF), self_error_rel[CUTOFF:T - CUTOFF], label="Self", linewidth=2)
axes[1].set_title("Relative Distance")
axes[1].set_xlabel("Simulation Step")
axes[1].grid(alpha=0.3)
axes[1].legend()

axes[2].plot(range(CUTOFF, T - CUTOFF), other_error_dir[CUTOFF:T - CUTOFF], label="Partner", linewidth=2)
axes[2].plot(range(CUTOFF, T - CUTOFF), self_error_dir[CUTOFF:T - CUTOFF], label="Self", linewidth=2)
axes[2].set_title("Unit Direction")
axes[2].set_xlabel("Simulation Step")
axes[2].grid(alpha=0.3)
axes[2].legend()

plt.tight_layout()
# plt.savefig(FIGURES_DIR / "lstm_probing.png")
# plt.savefig(FIGURES_DIR / "lstm_probing-better.png")
# plt.savefig(FIGURES_DIR / "lstm_probing_hard.png")
plt.show()



