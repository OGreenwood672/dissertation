"""
file: models.py

Contains model defintions for:

PPO Actors (For MAPPO)
PPO Critic

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import sqrt, save, load
import os

from .config import CommunicationType

# Helper function to initialise weights for PPO layers
def init_layer(layer, std=sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PPO_Actor(nn.Module):
    def __init__(
            self,
            num_agents, obs_dim, action_dim, comm_dim,
            communication_type=CommunicationType.CONTINUOUS,
            feature_dim=256, lstm_hidden_size=64, vocab_size=64,
            is_training=False, aim_codebook=None
        ):
        super().__init__()
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.hidden_size = lstm_hidden_size
        self.comm_type = communication_type

        # Main body
        self.body = nn.Sequential(
            init_layer(nn.Linear(obs_dim, feature_dim)),
            nn.ReLU(),
            init_layer(nn.Linear(feature_dim, feature_dim)),
            nn.ReLU(),
            init_layer(nn.Linear(feature_dim, feature_dim)),
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
            self.vq_layer = Quantiser(vocab_size, comm_dim=comm_dim, is_training=is_training)

        elif communication_type == CommunicationType.AIM:
            assert aim_codebook is not None
            aim_codebook = torch.tensor(aim_codebook, dtype=torch.float32)
            self.register_buffer('aim_codebook', aim_codebook)
        

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

        comm_logits = torch.tanh(self.comm_head(lstm_output))

        loss = torch.tensor(0.0, device=x.device)
        
        quantised_index = None
        codebook_loss = None
        if self.comm_type == CommunicationType.DISCRETE:
            B, T, N, C = comm_logits.shape
            flat_inputs = comm_logits.view(-1, C)
            quantised_flat, vq_loss, codebook_loss, quantised_index = self.vq_layer(flat_inputs)
            comm_output = quantised_flat.view(B, T, N, C)
            loss += vq_loss

        elif self.comm_type == CommunicationType.AIM:
            B, T, N, C = comm_logits.shape
            flat_inputs = comm_logits.view(-1, C)
            distances = torch.cdist(flat_inputs, self.aim_codebook, p=2)
            quantised_index = distances.argmin(-1)
            quantised = self.aim_codebook[quantised_index].view(B, T, N, C)
            comm_output = comm_logits + (quantised - comm_logits).detach()

        elif self.comm_type == CommunicationType.CONTINUOUS:
            comm_output = comm_logits


        if not has_time_dim:
            action_logits = action_logits.squeeze(1)
            comm_output = comm_output.squeeze(1)
                
        return action_logits, comm_output, codebook_loss, quantised_index, loss, (h_out, c_out)

    def get_comm_type(self):
        return self.comm_type


class PPO_Centralised_Critic(nn.Module):
    def __init__(self, num_agents, global_obs_dim, comm_dim, feature_dim=512):
        super().__init__()

        self.input_dim = global_obs_dim + comm_dim * num_agents

        self.body = nn.Sequential(
            nn.Linear(self.input_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
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
        assert x.shape[-1] == self.input_dim, "Input has wrong observation size"

        features = self.body(x)
        
        values = self.value_head(features)
        
        return values
    

class Quantiser(nn.Module):

    def __init__(self, vocab_size, comm_dim, commitment_cost=0.25, is_training=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.comm_dim = comm_dim
        self.commitment_cost = commitment_cost
        self.is_training = is_training

        # The codebook
        self.embedding = nn.Embedding(vocab_size, comm_dim)

        # codebook initialisation
        self.embedding.weight.data.uniform_(-0.5, 0.5)

        self.register_buffer('usage_count', torch.zeros(vocab_size))

    def forward(self, x):
        
        distances = torch.cdist(x, self.embedding.weight, p=2)

        min_distance_index = distances.argmin(-1)

        # Dead Code Revival
        if self.is_training:
            with torch.no_grad():
                unique, counts = torch.unique(min_distance_index, return_counts=True)
                self.usage_count[unique] += counts.float()

                dead_codes = torch.where(self.usage_count < 1.0)[0]
                if len(dead_codes) > 0:
                    random_inputs = x[torch.randperm(x.size(0))[:len(dead_codes)]]
                    self.embedding.weight.data[dead_codes] = random_inputs.detach()
                    self.usage_count[dead_codes] = 10.0 
                
                self.usage_count *= 0.99

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
    

class AIM(nn.Module):

    def __init__(self, obs_dim, hidden_size, latent_dim, vocab_size, commitment_cost=0.5, nudge_dim=None):
        super().__init__()

        # The codebook
        self.embedding = nn.Embedding(vocab_size, latent_dim)

        # codebook initialisation
        self.embedding.weight.data.uniform_(-0.5, 0.5)

        self.commitment_cost = commitment_cost

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, obs_dim)
        )

        # communicate what matters
        if nudge_dim is not None:
            self.nudge_head = nn.Linear(latent_dim, nudge_dim)

        self.nudge = 0
        self.recon = 0


    def forward(self, x, nudge=None):
        latent = self.encoder(x)

        # Quantise
        distances = torch.cdist(latent, self.embedding.weight, p=2)

        min_distance_index = distances.argmin(-1)

        quantised = self.embedding(min_distance_index)
        quantised = latent + (quantised - latent).detach()

        reconstructed = self.decoder(quantised)

        nudge_loss = 0.0
        nudge_contribution = 1.0
        recon_contribution = 0.5
        if nudge is not None:
            pred_actions = self.nudge_head(quantised)
            nudge_loss = F.mse_loss(pred_actions, nudge)
            self.nudge += nudge_loss.item() * nudge_contribution

        recon_loss = F.mse_loss(reconstructed, x)
        codebook_loss = F.mse_loss(quantised.detach(), latent)
        commitment_loss = F.mse_loss(quantised, latent.detach())

        self.recon += recon_loss.item()

        total_loss = recon_loss * recon_contribution + codebook_loss + commitment_loss * self.commitment_cost + nudge_loss * nudge_contribution

        return total_loss
    
    def get_codebook(self):
        return self.embedding.weight.detach().cpu().numpy()
    
    def save_codebook(self, save_folder):
        codebook = self.get_codebook()
        save(os.path.join(save_folder, "codebook.npy"), codebook)

    @staticmethod
    def load_codebook(seed):
        result_path = "./results/discrete/"

        folders = list(filter(lambda x: x.endswith(str(f"seed_{seed}")), os.listdir(result_path)))
        if len(folders) > 0:
            folder_path = result_path + folders[0]

        assert os.path.exists(folder_path)
        embedding = load(folder_path + "/codebook.npy")

        return embedding
    
    @staticmethod
    def compute_metrics(quantised_indices, vocab_size):
        counts = torch.bincount(quantised_indices.flatten(), minlength=vocab_size).float()
        
        probs = counts / counts.sum()
        
        entropy = -torch.sum(probs * torch.log(probs + 1e-10))
        
        perplexity = torch.exp(entropy)
        
        return entropy, perplexity
