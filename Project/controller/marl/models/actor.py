


import torch
from torch import nn
from typing import Tuple

from controller.marl.comms.aim_comms import AimComms
from controller.marl.comms.none_comms import NoneComms
from controller.marl.config import ActorHyperparameters, CommConfig, CommunicationType
from .utils import init_layer


class PPO_Actor(nn.Module):
    __constants__ = ['comm_type_str']

    def __init__(
            self,
            num_agents, obs_dim, action_dim,
            actor_config: ActorHyperparameters, comm_config: CommConfig,
            device=None
        ):
        super().__init__()
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.actor_config = actor_config
        self.comm_config = comm_config

        # self.hidden_size = lstm_hidden_size
        # self.comm_type = communication_type
        # self.comm_type_str = communication_type.name
        # self.num_comms = num_comms
        # self.vocab_size = vocab_size
        # self.hq_levels = None
        # self.comm_dim = comm_dim
        self.device = device

        # Main body
        # if communication_type == CommunicationType.AIM:
        #     assert encoder is not None
        #     self.base_encoder = nn.Identity()
        #     self.aim_encoder = encoder
        #     # Freeze pre-trained encoder
        #     for parameter in self.aim_encoder.parameters():
        #         parameter.requires_grad = False

        # else:
        #     self.base_encoder = nn.Sequential(
        #         init_layer(nn.Linear(obs_dim, feature_dim)),
        #         nn.ReLU(),
        #         init_layer(nn.Linear(feature_dim, comm_dim * num_comms)),
        #         nn.ReLU(),
        #     )
        #     self.aim_encoder = Encoder(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "sq-vae")

        # if comm_config.communication_type == CommunicationType.CONTINUOUS:
        #     # noise to learn
        #     # log to maintain positive std
        #     self.comm_log_std = nn.Parameter(torch.zeros(1, 1, 1, num_comms, comm_dim))

        #     self.comm_head = nn.Linear(lstm_hidden_size, comm_dim * num_comms)

        # elif comm_config.communication_type == CommunicationType.DISCRETE:
        #     self.vq_layer = Quantiser(vocab_size, comm_dim=comm_dim)
        #     self.comm_head = nn.Linear(lstm_hidden_size, comm_dim * num_comms)

        if comm_config.communication_type == CommunicationType.AIM:
            self.comm_protocol = AimComms(comm_config, actor_config, num_agents, device)


        elif comm_config.communication_type == CommunicationType.NONE:
            # self.comm_head = nn.Linear(lstm_hidden_size, comm_dim * num_comms)
            self.comm_protocol = NoneComms(comm_config, actor_config, num_agents, device)

        percieve_out_features = self.comm_protocol.percieve_out_features
        self.body = nn.Sequential(
            init_layer(nn.Linear(
                # (
                #     comm_config.comm_dim *
                #     comm_config.num_comms *
                #     (1 if comm_config.communication_type == CommunicationType.NONE else (num_agents)) +
                #     (comm_config.comm_dim * comm_config.num_comms if comm_config.communication_type == CommunicationType.AIM else 0)
                # ),
                percieve_out_features + comm_config.communication_size * comm_config.num_comms * num_agents,
                actor_config.lstm_hidden_size
            )),
            nn.ReLU(),
            init_layer(nn.Linear(actor_config.lstm_hidden_size, actor_config.lstm_hidden_size)),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            input_size=actor_config.lstm_hidden_size,#comm_dim * num_comms * (1 if communication_type == CommunicationType.NONE else num_agents), 
            hidden_size=actor_config.lstm_hidden_size, 
            batch_first=True
        )
        
        # Output for actions
        # action_input_dim = actor_config.lstm_hidden_size
        # if communication_type == CommunicationType.AIM:
        #     action_input_dim += comm_dim * num_comms
        
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
        assert N == self.num_agents, f"Input has wrong number of agents: Got {N}, Expected {self.num_agents}"
        assert O == self.obs_dim, f"Input has wrong observation size: Got {O}, Expected {self.obs_dim}"

        NC = self.comm_config.num_comms
        C = self.comm_config.communication_size
        
        h_out, c_out = hidden_states
        # if self.comm_type_str == "AIM":
        # assert world_comms is not None

        x = self.comm_protocol.percieve(x)

        world_comms = world_comms.reshape(B, T, N, NC * C)
        expanded_comms = world_comms.unsqueeze(2).expand(B, T, N, N, NC * C)
        # other_comms = expanded_comms[:, :, :, self.agent_mask, :].reshape(B, T, N, N - 1, -1)
        other_comms = expanded_comms[self.agent_mask.expand(B, T, N, N)].view(B, T, N, N - 1, -1)

        self_comm = torch.zeros(B, T, N, 1, NC * C, device=self.device)

        comms = torch.cat([self_comm, other_comms], dim=-2)
        comms = comms.reshape(B, T, N, N * NC * C)

        x = torch.cat([x, comms], dim=-1)
        x = x.transpose(1, 2).contiguous()
        x = x.reshape(B * N, T, C * NC * N + self.comm_protocol.percieve_out_features)

        x = self.body(x)
        x, (h_out, c_out) = self.lstm(x, hidden_states)

        # [B*N, T, Hidden] -> [B, N, T, Hidden] -> [B, T, N, Hidden]
        x = x.reshape(B, N, T, -1).transpose(1, 2)
        
        action_input = torch.cat([x, other_comms.sum(-2)], dim=-1)
        action_logits = self.action_head(action_input)

        # elif self.comm_type_str == "NONE":
        # else:
        #     x = self.base_encoder(x)
        #     x = x.transpose(1, 2).contiguous()
        #     x = x.reshape(B * N, T, self.comm_dim * self.num_comms)
        #     x = self.body(x)
        #     x, (h_out, c_out) = self.lstm(x, hidden_states)
            
        #     # [B*N, T, Hidden] -> [B, N, T, Hidden] -> [B, T, N, Hidden]
        #     x = x.reshape(B, N, T, -1).transpose(1, 2)
        
        #     action_logits = self.action_head(x)

        if self.comm_config.communication_type == CommunicationType.NONE or self.comm_config.communication_type == CommunicationType.AIM:
            # comm_logits = torch.zeros(B, T, N, self.num_comms * self.comm_dim, device=x.device).reshape(B, T, N, self.num_comms, -1)
            comm_logits = None
        else:
            comm_logits = self.comm_head(x).reshape(B, T, N, self.num_comms, -1)
        
        if not has_time_dim:
            action_logits = action_logits.squeeze(1)
            if comm_logits is not None:
                comm_logits = comm_logits.squeeze(1)
                
        return (
            action_logits, comm_logits,
            x, (h_out, c_out)
            # output, (None, None)
        )

