from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from controller.marl.core.config import ActorHyperparameters, CommConfig
from .comms import Comms


class DiscreteComms(Comms, nn.Module):
    def __init__( self, config: CommConfig, actor_config: ActorHyperparameters, log: Callable[[str, float], None], device: torch.device, **kwargs):
        nn.Module.__init__(self)

        self.config = config
        self.device = device
        self.num_agents = kwargs["num_agents"]
        self.actor_config = actor_config

        self.num_comms = config.num_comms
        self.comm_dim = config.communication_size

        self.vocab_size = config.vocab_size

        self.comm_head = nn.Sequential(
            nn.Linear(actor_config.lstm_hidden_size, self.num_comms * self.comm_dim),
            nn.Tanh()
        )

        codebook = torch.randn(self.vocab_size, self.comm_dim)
        codebook = F.normalize(codebook, dim=-1)

        self.codebook = nn.Parameter(codebook, requires_grad=False)

        self.percieve_out_features = 74
        self.out_features = self.num_comms * self.comm_dim

        self.to(device)

    def percieve(self, x: torch.Tensor):
        return x

    def quantise(self, latent: torch.Tensor):

        flat_shape = latent.shape[:-2]
        latent_flat = latent.reshape(-1, self.num_comms, self.comm_dim)

        dists = torch.cdist(latent_flat.reshape(-1, self.comm_dim), self.codebook).reshape(-1, self.num_comms, self.vocab_size)

        indices = dists.argmin(dim=-1)
        quantised = self.codebook[indices]

        quantised = quantised.reshape(*flat_shape, self.num_comms, self.comm_dim)
        indices = indices.reshape(*flat_shape, self.num_comms)
        logits = (-dists).reshape(*flat_shape, self.num_comms, self.vocab_size)
        return quantised, indices, logits

    def get_comms_during_rollout(self, x: torch.Tensor):

        B, _, N, _ = x.shape

        proto = self.comm_head(x).reshape(B, N, self.num_comms, self.comm_dim)

        _, _, logits = self.quantise(proto)

        dist = torch.distributions.Categorical(logits=logits)
        sampled_indices = dist.sample()

        comm_output = self.codebook[sampled_indices]
        comm_log_probs = dist.log_prob(sampled_indices).sum(dim=-1)

        if self.num_comms == 1:
            sampled_indices = sampled_indices.unsqueeze(-1)

        return comm_output, sampled_indices, comm_log_probs

    def get_comms_during_update(self, x: torch.Tensor, comms: torch.Tensor):

        B, T, N, _ = x.shape

        latent = self.comm_head(x).reshape(B, T, N, self.num_comms, self.comm_dim)

        latent_flat = latent.reshape(-1, self.num_comms, self.comm_dim)
        logits = - torch.cdist(
            latent_flat.reshape(-1, self.comm_dim),
            self.codebook
        ).reshape(B, T, N, self.num_comms, self.vocab_size)

        dist = torch.distributions.Categorical(logits=logits)

        new_comm_log_probs = dist.log_prob(comms.squeeze(-1)).sum(dim=-1)
        comm_entropy = dist.entropy().sum(dim=-1)

        hard_codewords = self.codebook[comms.squeeze(-1).long()]
        
        soft_codewords = dist.probs @ self.codebook
        
        comm_output = soft_codewords + (hard_codewords - soft_codewords).detach()
        comm_output = comm_output.view(-1, self.num_comms * self.comm_dim)

        return comm_output, new_comm_log_probs, comm_entropy

    def get_loss(self, x, comm_output, all_comms, true_returns, new_critic_values, targets):
        return 0.0