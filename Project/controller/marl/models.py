"""
file: models.py

Contains model defintions for:

PPO Actors (For IPPO)
IPPO Critic

VQ-VAE

"""

import torch.nn as nn


class IPPOActor(nn.Module):
    def __init__(self, obs_dim, env_action_dim, comm_action_dim=2):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        self.action_head = nn.Linear(64, env_action_dim)
        # Can add communication head later on
        

    def forward(self, x):
        features = self.body(x)
        action_logits = self.action_head(features)
        return action_logits
        


class IPPOCritic(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)