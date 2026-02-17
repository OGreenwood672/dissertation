from enum import Enum
from torch.utils.data import Dataset
import numpy as np
import torch


class NudgeFeature(str, Enum):
    Actions = "actions"
    Critic = "critic"


class ObsData(Dataset):

    # OBS, actions [5], critic value [1]

    def __init__(self, obs_log_path, action_count, nudge_feature=None):
        self.obs = np.genfromtxt(obs_log_path, skip_header=1, delimiter=",").astype(np.float32)
        self.action_count = action_count
        self.nudge_feature = nudge_feature
    
    def __len__(self):
        return len(self.obs)
    
    def __getitem__(self, idx):
        
        row = self.obs[idx]

        nudge = None
        if self.nudge_feature is NudgeFeature.Actions:
            nudge = row[-self.action_count:1]
        elif self.nudge_feature is NudgeFeature.Critic:
            nudge = row[-1:]
            nudge = (nudge - nudge.mean()) / (nudge.std() + 1e-8)

        if nudge is None:
            return torch.from_numpy(row[:-self.action_count - 1]), torch.tensor([])
        
        return torch.from_numpy(row[:-self.action_count - 1]), torch.tensor(nudge)