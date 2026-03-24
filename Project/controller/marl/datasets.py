from enum import Enum
from torch.utils.data import Dataset
import numpy as np
import torch


class NudgeFeature(str, Enum):
    Actions = "actions"
    Critic = "critic"


class ObsData(Dataset):

    # OBS, actions [5], critic value [1], reward [1]

    def __init__(self, obs_log_path, action_count):
        self.obs = np.genfromtxt(obs_log_path, skip_header=1, delimiter=",").astype(np.float32)
        self.action_count = action_count
    
    def __len__(self):
        return len(self.obs)
    
    def __getitem__(self, idx):
        
        row = self.obs[idx]

        target_reward = torch.from_numpy(row[-1:]) * 100.0

        return torch.from_numpy(row[:-self.action_count - 1 - 1]), target_reward
