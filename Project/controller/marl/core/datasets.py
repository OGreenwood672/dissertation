from enum import Enum
from torch.utils.data import Dataset
import numpy as np
import torch


class NudgeFeature(str, Enum):
    Actions = "actions"
    Critic = "critic"


class FilteredObsData(Dataset):

    # flat_obs []
    # flat_global_obs []
    # flat_actions [1]
    # flat_targets [3]
    # flat_c_values [1]
    # flat_returns [1]

    def __init__(self, obs_log_path, obs_count, global_obs_count, mask, device):
        self.obs = np.genfromtxt(obs_log_path, skip_header=1, delimiter=",").astype(np.float32)
        self.mask = mask
        self.device = device
        self.obs_count = obs_count
        self.global_obs_count = global_obs_count
    
    def __len__(self):
        return len(self.obs)
    
    def __getitem__(self, idx):
        
        row = self.obs[idx]

        # mask obs
        obs = torch.from_numpy(row[:- self.global_obs_count - 1 - 3 - 1 - 1]).float().to(self.device)
        filtered_obs = obs[self.mask]

        global_obs = torch.from_numpy(row[- self.global_obs_count - 1 - 3 - 1 - 1:- 1 - 3 - 1 - 1]).float().to(self.device)
        actions = torch.from_numpy(np.array([row[- 3 - 1 - 1]])).long().to(self.device)
        targets = torch.from_numpy(row[- 3 - 1 - 1:- 1 - 1]).float().to(self.device)
        critic_value = torch.from_numpy(np.array([row[- 1 - 1]])).float().to(self.device)
        return_value = torch.from_numpy(np.array([row[- 1]])).float().to(self.device)


        return filtered_obs, global_obs, actions, targets, critic_value, return_value
    

class ObsData(Dataset):
    def __init__(self, obs_log_path, global_obs_count, T, W, N, device):

        print(f"Loading CSV from {obs_log_path}...")

        raw_obs = np.genfromtxt(obs_log_path, skip_header=0, delimiter=",").astype(np.float32)
        
        self.device = device
        self.global_obs_count = global_obs_count
        
        E = raw_obs.shape[0] // (W * N) // T
        F = raw_obs.shape[1]
        
        
        restored_obs = raw_obs.reshape((E, T, W, N, F))
        episodes = restored_obs.transpose(0, 2, 1, 3, 4)
        
        self.episodes = episodes.reshape((E * W, T, N, F))
        print(f"Final dataset shape ready for Dataloader: {self.episodes.shape}")
        
    def __len__(self):
        return self.episodes.shape[0]
    
    def __getitem__(self, idx):
        episode = self.episodes[idx]
        
        GO = self.global_obs_count
        
        obs_slice = episode[..., :- GO - 6]
        global_obs_slice = episode[..., - GO - 6 : - 6]
        actions_slice = episode[..., -6]
        targets_slice = episode[..., -5 : -2]
        critic_slice = episode[..., -2]
        returns_slice = episode[..., -1]
        
        obs = torch.from_numpy(obs_slice).float().to(self.device)
        
        global_obs = torch.from_numpy(global_obs_slice).float().to(self.device)
        
        actions = torch.from_numpy(actions_slice).long().unsqueeze(-1).to(self.device)
        targets = torch.from_numpy(targets_slice).float().to(self.device)
        critic_value = torch.from_numpy(critic_slice).float().unsqueeze(-1).to(self.device)
        return_value = torch.from_numpy(returns_slice).float().unsqueeze(-1).to(self.device)
        
        return obs, global_obs, actions, targets, critic_value, return_value