


import torch
from torch import nn
from typing import Tuple, Optional

from controller.marl.config import CommunicationType
from controller.marl.models.encoders.encoder import Encoder
from controller.marl.models.quantiser import Quantiser
from .utils import init_layer


class PPO_Actor(nn.Module):
    __constants__ = ['comm_type_str']

    def __init__(
            self,
            num_agents, obs_dim, action_dim, comm_dim,
            communication_type=CommunicationType.CONTINUOUS,
            feature_dim=256, lstm_hidden_size=64, vocab_size=64,
            num_comms=1,
            encoder=None, device=None
        ):
        super().__init__()
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.hidden_size = lstm_hidden_size
        self.comm_type = communication_type
        self.comm_type_str = communication_type.name
        self.num_comms = num_comms
        self.vocab_size = vocab_size
        self.hq_levels = None
        self.comm_dim = comm_dim
        self.device = device

        # Main body
        if communication_type == CommunicationType.AIM:
            assert encoder is not None
            self.base_encoder = nn.Identity()
            self.aim_encoder = encoder
            # Freeze pre-trained encoder
            for parameter in self.aim_encoder.parameters():
                parameter.requires_grad = False

        else:
            self.base_encoder = nn.Sequential(
                init_layer(nn.Linear(obs_dim, feature_dim)),
                nn.ReLU(),
                init_layer(nn.Linear(feature_dim, comm_dim * num_comms)),
                nn.ReLU(),
            )
            self.aim_encoder = Encoder(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "sq-vae")

        self.body = nn.Sequential(
            init_layer(nn.Linear(
                (
                    comm_dim *
                    num_comms *
                    (1 if communication_type == CommunicationType.NONE else (num_agents)) +
                    (comm_dim * num_comms if communication_type == CommunicationType.AIM else 0)
                ),
                lstm_hidden_size
            )),
            nn.ReLU(),
            init_layer(nn.Linear(lstm_hidden_size, lstm_hidden_size)),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            input_size=lstm_hidden_size,#comm_dim * num_comms * (1 if communication_type == CommunicationType.NONE else num_agents), 
            hidden_size=lstm_hidden_size, 
            batch_first=True
        )
        
        # Output for actions
        action_input_dim = lstm_hidden_size
        if communication_type == CommunicationType.AIM:
             action_input_dim += comm_dim * num_comms
        self.action_head = nn.Sequential(
            init_layer(nn.Linear(action_input_dim, lstm_hidden_size)),
            nn.ReLU(),
            init_layer(nn.Linear(lstm_hidden_size, action_dim))
        )

        self.register_buffer('agent_mask', ~torch.eye(num_agents, dtype=torch.bool).unsqueeze(0).unsqueeze(0))

        self.comm_head = nn.Identity()
        self.comm_heads = nn.Identity()
        self.vq_layer = nn.Identity()
        self.predict_agents_returns = nn.Identity()
        self.predict_critic_value = nn.Identity()
        self.predict_intent_sender = nn.Identity()
        self.predict_intent_receiver = nn.Identity()
        self.predict_intent_sender_goal = nn.Identity()
        self.predict_intent_receiver_goal = nn.Identity()

        if communication_type == CommunicationType.CONTINUOUS:
            # noise to learn
            # log to maintain positive std
            self.comm_log_std = nn.Parameter(torch.zeros(1, 1, 1, num_comms, comm_dim))

            self.comm_head = nn.Linear(lstm_hidden_size, comm_dim * num_comms)

        elif communication_type == CommunicationType.DISCRETE:
            self.vq_layer = Quantiser(vocab_size, comm_dim=comm_dim)
            self.comm_head = nn.Linear(lstm_hidden_size, comm_dim * num_comms)

        elif communication_type == CommunicationType.AIM:
            aim_codebook = torch.tensor(encoder.get_codebooks(), dtype=torch.float32, device=device)
            self.register_buffer('aim_codebook', aim_codebook)

            self.hq_levels = aim_codebook.shape[0]

            # Teaching the fixed codebook
            # Sender must understand what its sending
            # Predict other agent rewards
            self.predict_agents_returns = nn.Linear(comm_dim * num_comms + lstm_hidden_size, num_agents)
            
            # Reciever must understand how it is getting effected
            # Predict critic value
            self.predict_critic_value = nn.Linear(comm_dim * num_comms + lstm_hidden_size, 1)

            self.comm_heads = nn.ModuleList([
                nn.Linear(lstm_hidden_size + comm_dim * hq_level * num_comms, num_comms * vocab_size)
                for hq_level in range(self.hq_levels)
            ])

            # Intent Predictors
            # self.predict_intent_sender = nn.Linear(comm_dim * num_comms + lstm_hidden_size, action_dim)
            # self.predict_intent_receiver = nn.Linear(comm_dim * num_comms + lstm_hidden_size, action_dim)

            self.predict_intent_sender_goal = nn.Linear(comm_dim * num_comms + lstm_hidden_size, 2)
            self.predict_intent_receiver_goal = nn.Linear(comm_dim * num_comms + lstm_hidden_size, 2)


        elif communication_type == CommunicationType.NONE:
            self.comm_head = nn.Linear(lstm_hidden_size, comm_dim * num_comms)
        
    @torch.jit.export
    def init_hidden(self, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initializes the hidden states for all agents.
        Returns a tuple of (h, c).
        """
        length = batch_size * self.num_agents

        h = torch.zeros(1, length, self.hidden_size, device=self.device)
        c = torch.zeros(1, length, self.hidden_size, device=self.device)
        return (h, c)
    

    def forward(self, x: torch.Tensor, hidden_states: Tuple[torch.Tensor, torch.Tensor], world_comms: Optional[torch.Tensor] = None):
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
            if world_comms is not None:
                world_comms = world_comms.unsqueeze(1)
        
        B, T, N, O = x.shape
        assert N == self.num_agents, f"Input has wrong number of agents: Got {N}, Expected {self.num_agents}"
        assert O == self.obs_dim, f"Input has wrong observation size: Got {O}, Expected {self.obs_dim}"
        
        h_out, c_out = hidden_states
        if self.comm_type_str == "AIM":
            assert world_comms is not None
            world_comms = world_comms.reshape(B, T, N, self.num_comms * self.comm_dim)

            _, environment_obs = self.aim_encoder(x)
            # environment_obs = self.encoder(x)

            expanded_comms = world_comms.unsqueeze(2).expand(B, T, N, N, self.num_comms * self.comm_dim)
            # other_comms = expanded_comms[:, :, :, self.agent_mask, :].reshape(B, T, N, N - 1, -1)
            other_comms = expanded_comms[self.agent_mask.expand(B, T, N, N)].view(B, T, N, N - 1, -1)

            self_comm = torch.zeros(B, T, N, 1, self.num_comms * self.comm_dim, device=x.device)

            comms = torch.cat([self_comm, other_comms], dim=-2)
            comms = comms.reshape(B, T, N, N * self.num_comms * self.comm_dim)

            x = torch.cat([environment_obs, comms], dim=-1)
            x = x.transpose(1, 2).contiguous()
            x = x.reshape(B * N, T, self.comm_dim * self.num_comms * N + self.comm_dim * self.num_comms)

            x = self.body(x)
            x, (h_out, c_out) = self.lstm(x, hidden_states)

            # [B*N, T, Hidden] -> [B, N, T, Hidden] -> [B, T, N, Hidden]
            x = x.reshape(B, N, T, -1).transpose(1, 2)
            
            action_input = torch.cat([x, other_comms.sum(-2)], dim=-1)
            action_logits = self.action_head(action_input)


        # elif self.comm_type_str == "NONE":
        else:
            features = self.base_encoder(x).transpose(1, 2).contiguous().reshape(B * N, T, self.comm_dim * self.num_comms)
            x = self.body(features)
            x, (h_out, c_out) = self.lstm(x, hidden_states)
            
            # [B*N, T, Hidden] -> [B, N, T, Hidden] -> [B, T, N, Hidden]
            x = x.reshape(B, N, T, -1).transpose(1, 2)
        
            action_logits = self.action_head(x)

        if self.comm_type_str == "NONE":
            comm_logits = torch.zeros(B, T, N, self.num_comms * self.comm_dim, device=x.device).reshape(B, T, N, self.num_comms, -1)
        elif self.comm_type_str != "AIM":
            comm_logits = self.comm_head(x).reshape(B, T, N, self.num_comms, -1)
        else:
            comm_logits = None
        
        if not has_time_dim:
            action_logits = action_logits.squeeze(1)
            if comm_logits is not None:
                comm_logits = comm_logits.squeeze(1)
                
        return (
            action_logits, comm_logits,
            x, (h_out, c_out)
            # output, (None, None)
        )

