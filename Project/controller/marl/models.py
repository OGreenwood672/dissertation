"""
file: models.py

Contains model defintions for:

PPO Actors (For IPPO)
IPPO Critic

VQ-VAE

"""

import torch
import torch.nn as nn

class PPO_Actor(nn.Module):
    def __init__(self, num_agents, obs_dim, action_dim, lstm_hidden_size=64):
        super().__init__()
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.hidden_size = lstm_hidden_size
        feature_dim = 64

        # Main body
        self.body = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, feature_dim),
            nn.ReLU()
        )
        
        # Output for actions
        self.action_head = nn.Linear(lstm_hidden_size, action_dim)
        
        # Unique LSTM for each agent
        self.lstm = nn.LSTM(
            input_size=feature_dim, 
            hidden_size=lstm_hidden_size, 
            batch_first=True
        )

    def init_hidden(self, batch_size=1):
        """
        Initializes the hidden states for all agents.
        Returns a list of N (h, c) tuples.
        """
        device = self.body[0].weight.device
        length = batch_size * self.num_agents

        h = torch.zeros(1, length, self.hidden_size, device=device)
        c = torch.zeros(1, length, self.hidden_size, device=device)
        return (h, c)

    def forward(self, x, hidden_states):
        """
        Processes observations and hidden states.
        
        x: Observations. 
           Shape: [BatchSize, Timestep, NumAgents, ObsDim] = [B, T, N, O]
           OR
           Shape: [BatchSize, NumAgents, ObsDim] = [B, N, O]
        hidden_states: List of N (h, c) tuples. 
                       Each h, c has shape [BatchSize, HiddenSize]

        returns action_logits, new_hidden_states
        """
        # Add timestep for single inference
        has_time_dim = (x.ndim == 4)
        if not has_time_dim: x = x.unsqueeze(1)
        
        B, T, N, O = x.shape
        assert N == self.num_agents, "Input has wrong number of agents"
        assert O == self.obs_dim, "Input has wrong observation size"
        
        x = x.transpose(1, 2).contiguous().reshape(B * N, T, O)
        features = self.body(x)

        lstm_output, (h_out, c_out) = self.lstm(features, hidden_states)

        # [B*N, T, Hidden] -> [B, N, T, Hidden] -> [B, T, N, Hidden]
        lstm_output = lstm_output.reshape(B, N, T, -1).transpose(1, 2)
        
        action_logits = self.action_head(lstm_output)

        if not has_time_dim: action_logits = action_logits.squeeze(1)
                
        return action_logits, (h_out, c_out)


class PPO_Centralised_Critic(nn.Module):
    def __init__(self, num_agents, obs_dim):
        super().__init__()
        feature_dim = 64

        self.input_dim = obs_dim

        self.body = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, feature_dim),
            nn.ReLU()
        )
        
        # Output a value per agent
        self.value_head = nn.Linear(feature_dim, num_agents)


    def forward(self, x):
        """
        Processes observations and hidden states.
        
        x: Observations.Shape: [BatchSize, Global obs]
        hidden_states: List of (h, c) tuples.

        returns: (values, (new_hidden_state, critic_values))
        """
        assert x.shape[-1] == self.input_dim, "Input has wrong observation size"

        features = self.body(x)
        
        values = self.value_head(features)
        
        return values