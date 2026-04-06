from enum import Enum
from torch.utils.data import Dataset
import numpy as np
import torch


class NudgeFeature(str, Enum):
    Actions = "actions"
    Critic = "critic"


class ObsData(Dataset):

    # OBS, actions [5], targets [3] (x, y, is found)

    def __init__(self, obs_log_path, action_count, target_count, mask, device):
        self.obs = np.genfromtxt(obs_log_path, skip_header=1, delimiter=",").astype(np.float32)
        self.mask = mask
        self.device = device
        self.action_count = action_count
        self.target_count = target_count
    
    def __len__(self):
        return len(self.obs)
    
    def __getitem__(self, idx):
        
        row = self.obs[idx]

        # mask obs
        obs = torch.from_numpy(row[:- self.target_count - self.action_count]).float().to(self.device)
        filtered_obs = obs[self.mask]

        return filtered_obs


class ImitationLearningDataset(Dataset):

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


        return obs, global_obs, actions, targets, critic_value, return_value