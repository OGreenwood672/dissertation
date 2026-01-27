"""
file: models.py

Contains model defintions for:

PPO Actors (For MAPPO)
PPO Critic

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CommunicationType


class PPO_Actor(nn.Module):
    def __init__(self, num_agents, obs_dim, action_dim, comm_dim=5, communication_type=CommunicationType.CONTINUOUS, feature_dim=256, lstm_hidden_size=64):
        super().__init__()
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.hidden_size = lstm_hidden_size
        self.comm_type = communication_type

        # Main body
        self.body = nn.Sequential(
            nn.Linear(obs_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU()
            
        )

        # Unique LSTM for each agent
        self.lstm = nn.LSTM(
            input_size=feature_dim, 
            hidden_size=lstm_hidden_size, 
            batch_first=True
        )
        
        # Output for actions
        self.action_head = nn.Linear(lstm_hidden_size, action_dim)

        # Output for continuous communication
        self.comm_head = nn.Linear(lstm_hidden_size, comm_dim)

        
        if communication_type == CommunicationType.CONTINUOUS:
            # noise to learn
            # log to maintain positive std
            self.comm_log_std = nn.Parameter(torch.zeros(1, comm_dim))

        elif communication_type == CommunicationType.DISCRETE:
            self.vq_layer = Quantiser(vocab_size=64, comm_dim=comm_dim)
        

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
        assert N == self.num_agents, f"Input has wrong number of agents: Got {N}, Expected {self.num_agents}"
        assert O == self.obs_dim, f"Input has wrong observation size: Got {O}, Expected {self.obs_dim}"
        
        x = x.transpose(1, 2).contiguous().reshape(B * N, T, O)
        features = self.body(x)

        lstm_output, (h_out, c_out) = self.lstm(features, hidden_states)

        # [B*N, T, Hidden] -> [B, N, T, Hidden] -> [B, T, N, Hidden]
        lstm_output = lstm_output.reshape(B, N, T, -1).transpose(1, 2)
        
        action_logits = self.action_head(lstm_output)

        comm_logits = self.comm_head(lstm_output)

        loss = torch.tensor(0.0, device=x.device)
        
        quantised_index = None
        codebook_loss = None
        if self.comm_type == CommunicationType.DISCRETE:
            B, T, N, C = comm_logits.shape
            flat_inputs = comm_logits.view(-1, C)
            quantised_flat, vq_loss, codebook_loss, quantised_index = self.vq_layer(flat_inputs)
            comm_logits = quantised_flat.view(B, T, N, C)
            loss += vq_loss

        if not has_time_dim:
            action_logits = action_logits.squeeze(1)
            comm_logits = comm_logits.squeeze(1)
                
        return action_logits, comm_logits, codebook_loss, quantised_index, loss, (h_out, c_out)

    def get_comm_type(self):
        return self.comm_type


class PPO_Centralised_Critic(nn.Module):
    def __init__(self, num_agents, global_obs_dim, feature_dim=256):
        super().__init__()

        self.global_obs_dim = global_obs_dim

        self.body = nn.Sequential(
            nn.Linear(self.global_obs_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
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
        assert x.shape[-1] == self.global_obs_dim, "Input has wrong observation size"

        features = self.body(x)
        
        values = self.value_head(features)
        
        return values
    

class Quantiser(nn.Module):

    def __init__(self, vocab_size, comm_dim, commitment_cost=0.25):
        super().__init__()
        self.vocab_size = vocab_size
        self.comm_dim = comm_dim
        self.commitment_cost = commitment_cost

        # The codebook
        self.embedding = nn.Embedding(vocab_size, comm_dim)

        # codebook initialisation
        self.embedding.weight.data.uniform_( -1 / vocab_size, 1 / vocab_size)

    def forward(self, x):
        
        distances = torch.cdist(x, self.embedding.weight, p=2)

        min_distance_index = distances.argmin(-1)

        quantised = self.embedding(min_distance_index)

        codebook_loss = F.mse_loss(quantised, x.detach())
        e_latent_loss = F.mse_loss(quantised.detach(), x)

        loss = codebook_loss + self.commitment_cost * e_latent_loss

        # to allow backprop, effectivly x + constant = quantised
        quantised = x + (quantised - x).detach()

        return quantised, loss, codebook_loss, min_distance_index
    
    def compute_metrics(self, quantised_indices):
        
        counts = torch.bincount(quantised_indices.flatten(), minlength=self.vocab_size).float()
        
        probs = counts / counts.sum()
        
        entropy = -torch.sum(probs * torch.log(probs + 1e-10))
        
        perplexity = torch.exp(entropy)
        
        return entropy, perplexity
    
    @staticmethod
    def get_active_codebook_usage(indices):
        return len(torch.unique(indices.flatten()))