from typing import Callable

import torch
import torch.nn as nn

from controller.marl.core.config import ActorHyperparameters, CommConfig

from .comms import Comms


class ContinuousComms(Comms, nn.Module):

    def __init__(self, config: CommConfig, actor_config: ActorHyperparameters, log: Callable[[str, float], None], device: torch.device, **kwargs):
        nn.Module.__init__(self)

        self.config = config
        self.device = device
        self.num_agents = kwargs["num_agents"]
        self.actor_config = actor_config

        self.comm_log_std = nn.Parameter(torch.zeros(1, 1, 1, config.num_comms, config.communication_size))
        self.comm_head = nn.Sequential(
            nn.Linear(actor_config.lstm_hidden_size, config.communication_size * config.num_comms),
            nn.Tanh()
        )

        self.percieve_out_features = 74
        self.out_features = config.communication_size * config.num_comms
        
        self.to(device)

    def percieve(self, x: torch.Tensor):
        return x
    
    def get_comms_during_rollout(self, x: torch.Tensor):
        B, _, N, _ = x.shape

        comm_logits = self.comm_head(x).reshape(B, N, self.config.num_comms, self.config.communication_size)

        comm_std = self.comm_log_std.exp().squeeze(1).expand_as(comm_logits)
        comm_dist = torch.distributions.Normal(comm_logits, comm_std)
        comm_output = comm_dist.sample()
        comm_log_probs = comm_dist.log_prob(comm_output).sum(dim=[-2, -1]) # sum as log(B) + log(C) = log(BC)
        return comm_output, comm_output, comm_log_probs


    def get_comms_during_update(self, x: torch.Tensor, comms: torch.Tensor):

        B, T, N, _ = x.shape

        comm_logits = self.comm_head(x).reshape(B, T, N, self.config.num_comms, -1)

        new_comm_std = self.comm_log_std.exp().expand_as(comm_logits)
        new_comm_dist = torch.distributions.Normal(comm_logits, new_comm_std)
        new_comm_log_probs = new_comm_dist.log_prob(comms).sum(dim=[-2, -1]).flatten()
        comm_entropy = new_comm_dist.entropy().sum(dim=[-2, -1]).flatten()

        return _, new_comm_log_probs, comm_entropy

    
    def get_loss(self, x, comm_output, all_comms, true_returns, new_critic_values, targets):
        return 0.0