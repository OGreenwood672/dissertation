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
        self.lstms = nn.ModuleList([
            nn.LSTMCell(feature_dim, lstm_hidden_size) 
            for _ in range(num_agents)
        ])

    def init_hidden(self, batch_size=1):
        """
        Initializes the hidden states for all agents.
        Returns a list of N (h, c) tuples.
        """
        device = self.body[0].weight.device
        return [
            (torch.zeros(batch_size, self.hidden_size, device=device), 
             torch.zeros(batch_size, self.hidden_size, device=device))
            for _ in range(self.num_agents)
        ]

    def forward(self, x, hidden_states):
        """
        Processes observations and hidden states.
        
        x: Observations. 
           Shape: [BatchSize, NumAgents, ObsDim] = [B, N, O]
        hidden_states: List of N (h, c) tuples. 
                       Each h, c has shape [BatchSize, HiddenSize]
        returns (action_logits, new_hidden_states)
        """
        B, N, O = x.shape
        assert N == self.num_agents, "Input has wrong number of agents"
        
        x_flat = x.reshape(B * N, O)
        features_flat = self.body(x_flat)
        
        features = features_flat.reshape(B, N, -1)
        
        new_hidden_states = []
        lstm_outputs = []
        
        for i in range(self.num_agents):
            agent_features = features[:, i, :] 
            agent_hidden = hidden_states[i]
            
            h_new, c_new = self.lstms[i](agent_features, agent_hidden)
            
            new_hidden_states.append((h_new, c_new))
            lstm_outputs.append(h_new)
        
        all_lstm_outputs = torch.stack(lstm_outputs, dim=1)
        all_lstm_outputs_flat = all_lstm_outputs.reshape(B * N, self.hidden_size)
        
        action_logits_flat = self.action_head(all_lstm_outputs_flat)
        action_logits = action_logits_flat.reshape(B, N, -1)
        
        return action_logits, new_hidden_states


class PPO_Centralised_Critic(nn.Module):
    def __init__(self, num_agents, obs_dim, lstm_hidden_size=64):
        super().__init__()
        self.num_agents = num_agents
        self.hidden_size = lstm_hidden_size
        feature_dim = 64

        self.input_dim = obs_dim * num_agents

        self.body = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, feature_dim),
            nn.ReLU()
        )
        
        self.lstm = nn.LSTMCell(feature_dim, lstm_hidden_size)

        # Output a value per agent
        self.value_head = nn.Linear(lstm_hidden_size, num_agents)


    def init_hidden(self, batch_size=1):
        """
        Initializes the hidden states for all agents.
        Returns a list of N (h, c) tuples.
        """
        device = self.body[0].weight.device
        return (
            torch.zeros(batch_size, self.hidden_size, device=device), 
            torch.zeros(batch_size, self.hidden_size, device=device)
        )

    def forward(self, x, hidden_state):
        """
        Processes observations and hidden states.
        
        x: Observations.Shape: [BatchSize, NumAgents * ObsDim]
        hidden_states: List of (h, c) tuples.

        returns: (values, (new_hidden_state, critic_values))
        """
        _, input_shape  = x.shape
        assert input_shape == self.input_dim, "Input has wrong size"

        features = self.body(x)
        
        new_hidden_state, critic_values = self.lstm(features, hidden_state)
        
        values = self.value_head(new_hidden_state)
        
        return values, (new_hidden_state, critic_values)