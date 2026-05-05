


from pathlib import Path

import torch
from torch import nn
from typing import Callable, Tuple

from controller.marl.comms.aim_comms import AimComms
from controller.marl.comms.continuous_comms import ContinuousComms
from controller.marl.comms.discrete_comms import DiscreteComms
from controller.marl.comms.none_comms import NoneComms
from controller.marl.comms.reflective_comms import ReflectiveComms
from controller.marl.core.config import ActorHyperparameters, CommConfig, CommunicationType
from .utils import init_layer


class PPO_Actor(nn.Module):
    __constants__ = ['comm_type_str']

    def __init__(
            self,
            num_agents: int, obs_dim: int, action_dim: int, obs_external_mask: torch.Tensor,
            actor_config: ActorHyperparameters, comm_config: CommConfig,
            log: Callable[[str, float], None],
            device=None
        ):
        super().__init__()
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.actor_config = actor_config
        self.comm_config = comm_config

        self.device = device
        
        comm_protocol_class = {
            CommunicationType.AIM: AimComms,
            CommunicationType.CONTINUOUS: ContinuousComms,
            CommunicationType.DISCRETE: DiscreteComms,
            CommunicationType.NONE: NoneComms,
            CommunicationType.REFLECTIVE: ReflectiveComms,
        }
        self.comm_protocol = comm_protocol_class[comm_config.communication_type](
            comm_config,
            actor_config,
            log,
            device,
            num_agents=num_agents,
            obs_external_mask=obs_external_mask,
            obs_dim=obs_dim,
        )

        percieve_out_features = self.comm_protocol.percieve_out_features
        self.body = nn.Sequential(
            init_layer(nn.Linear(
                percieve_out_features + comm_config.communication_size * comm_config.num_comms * num_agents,
                actor_config.lstm_hidden_size
            )),
            nn.ReLU(),
            init_layer(nn.Linear(actor_config.lstm_hidden_size, actor_config.lstm_hidden_size)),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            input_size=actor_config.lstm_hidden_size,
            hidden_size=actor_config.lstm_hidden_size, 
            batch_first=True
        )
        
        # Output for actions        
        action_input_dim = actor_config.lstm_hidden_size + self.comm_protocol.out_features

        self.action_head = nn.Sequential(
            init_layer(nn.Linear(action_input_dim, actor_config.lstm_hidden_size)),
            nn.ReLU(),
            init_layer(nn.Linear(actor_config.lstm_hidden_size, action_dim))
        )

        self.register_buffer('agent_mask', ~torch.eye(num_agents, dtype=torch.bool).unsqueeze(0).unsqueeze(0))


        
    @torch.jit.export
    def init_hidden(self, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initializes the hidden states for all agents.
        Returns a tuple of (h, c).
        """
        length = batch_size * self.num_agents

        h = torch.zeros(1, length, self.actor_config.lstm_hidden_size, device=self.device)
        c = torch.zeros(1, length, self.actor_config.lstm_hidden_size, device=self.device)
        return (h, c)
    

    def forward(self, x: torch.Tensor, hidden_states: Tuple[torch.Tensor, torch.Tensor], world_comms: torch.Tensor):
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
        if not has_time_dim:
            x = x.unsqueeze(1)
            world_comms = world_comms.unsqueeze(1)
        
        B, T, N, O = x.shape
        # assert N == self.num_agents, f"Input has wrong number of agents: Got {N}, Expected {self.num_agents}"
        assert O == self.obs_dim, f"Input has wrong observation size: Got {O}, Expected {self.obs_dim}"

        NC = self.comm_config.num_comms
        C = self.comm_config.communication_size
        
        h_out, c_out = hidden_states

        x = self.comm_protocol.percieve(x)

        world_comms = world_comms.reshape(B, T, N, NC * C)
        expanded_comms = world_comms.unsqueeze(2).expand(B, T, N, N, NC * C).clone()
        # other_comms = expanded_comms[:, :, :, self.agent_mask, :].reshape(B, T, N, N - 1, -1)
        # expanded_comms = expanded_comms[self.agent_mask.expand(B, T, N, N)].view(B, T, N, N - 1, -1)
        self_mask = ~self.agent_mask
        expanded_comms.masked_fill_(self_mask.view(1, 1, N, N, 1), 0.0)

        dropout_prob = getattr(self, "comm_dropout_prob", 0.0)
        if self.training and dropout_prob > 0.0:
            dropout_mask = (torch.rand(B, T, N, N, 1, device=self.device) > dropout_prob).float()
            expanded_comms = expanded_comms * dropout_mask

        # self_comm = torch.zeros(B, T, N, 1, NC * C, device=self.device)
        # comms = torch.cat([self_comm, other_comms], dim=-2)

        flat_comms = expanded_comms.reshape(B, T, N, N * NC * C)

        x = torch.cat([x, flat_comms], dim=-1)
        x = x.transpose(1, 2).contiguous()
        x = x.reshape(B * N, T, C * NC * N + self.comm_protocol.percieve_out_features)

        x = self.body(x)
        x, (h_out, c_out) = self.lstm(x, hidden_states)

        # [B*N, T, Hidden] -> [B, N, T, Hidden] -> [B, T, N, Hidden]
        x = x.reshape(B, N, T, -1).transpose(1, 2)
        
        action_input = torch.cat([x, expanded_comms.sum(-2)], dim=-1)
        action_logits = self.action_head(action_input)
        
        if not has_time_dim:
            action_logits = action_logits.squeeze(1)
                
        return (
            action_logits,
            x, (h_out, c_out)
        )

