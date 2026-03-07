


import torch
from torch import nn

from controller.marl.config import CommunicationType
from controller.marl.models.quantiser import Quantiser
from .utils import init_layer


class PPO_Actor(nn.Module):
    def __init__(
            self,
            num_agents, obs_dim, action_dim, comm_dim,
            communication_type=CommunicationType.CONTINUOUS,
            feature_dim=256, lstm_hidden_size=64, vocab_size=64,
            is_training=False, num_comms=1,
            encoder=None, device=None
        ):
        super().__init__()
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.hidden_size = lstm_hidden_size
        self.comm_type = communication_type
        self.num_comms = num_comms
        self.vocab_size = vocab_size
        self.hq_levels = None
        self.comm_dim = comm_dim
        self.is_training = is_training
        self.device = device

        # Main body
        if communication_type == CommunicationType.AIM:
            assert encoder is not None
            self.encoder = encoder
            # Freeze pre-trained encoder
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

        else:
            self.encoder = nn.Sequential(
                init_layer(nn.Linear(obs_dim, feature_dim)),
                nn.ReLU(),
                init_layer(nn.Linear(feature_dim, comm_dim * num_comms)),
                nn.ReLU(),
            )


        self.body = nn.Sequential(
            init_layer(nn.Linear(comm_dim * num_comms * (1 if communication_type == CommunicationType.NONE else num_agents), lstm_hidden_size)),
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
        self.action_head = nn.Linear(lstm_hidden_size, action_dim)

        if communication_type == CommunicationType.CONTINUOUS:
            # noise to learn
            # log to maintain positive std
            self.comm_log_std = nn.Parameter(torch.zeros(1, 1, 1, num_comms, comm_dim))

            self.comm_head = nn.Linear(lstm_hidden_size, comm_dim * num_comms)

        elif communication_type == CommunicationType.DISCRETE:
            self.vq_layer = Quantiser(vocab_size, comm_dim=comm_dim, is_training=is_training)
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

            for hq_level in range(self.hq_levels):
                setattr(self, f"comm_head_{hq_level}", nn.Linear(lstm_hidden_size + comm_dim * hq_level * num_comms, num_comms * vocab_size))

            # Intent Predictors
            self.predict_intent_sender = nn.Linear(comm_dim * num_comms + lstm_hidden_size, action_dim)
            self.predict_intent_receiver = nn.Linear(comm_dim * num_comms + lstm_hidden_size, action_dim)

        elif communication_type == CommunicationType.NONE:
            self.comm_head = nn.Linear(lstm_hidden_size, comm_dim * num_comms)
        

    def init_hidden(self, batch_size=1):
        """
        Initializes the hidden states for all agents.
        Returns a list of N (h, c) tuples.
        """
        length = batch_size * self.num_agents

        h = torch.zeros(1, length, self.hidden_size, device=self.device)
        c = torch.zeros(1, length, self.hidden_size, device=self.device)
        return (h, c)
    

    def forward(self, x, hidden_states, world_comms=None):
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
        
        if world_comms is not None:
            world_comms = world_comms.reshape(B, T, N, self.num_comms * self.comm_dim)

            _, self_comm = self.encoder(x)

            mask = torch.eye(N, dtype=torch.bool, device=x.device)
            new_comms = world_comms.unsqueeze(2).repeat(1, 1, N, 1, 1)
            new_comms[:, :, mask] = self_comm
            new_comms = new_comms.reshape(B, T, N, self.num_comms * self.comm_dim * N).transpose(1, 2).contiguous().reshape(B * N, T, self.comm_dim * self.num_comms * N)
        else:
            new_comms = self.encoder(x).transpose(1, 2).contiguous().reshape(B * N, T, self.comm_dim * self.num_comms)

        # lstm_output, (h_out, c_out) = self.lstm(new_comms, hidden_states)
        output = self.body(new_comms)
        output, (h_out, c_out) = self.lstm(output, hidden_states)


        # [B*N, T, Hidden] -> [B, N, T, Hidden] -> [B, T, N, Hidden]
        output = output.reshape(B, N, T, -1).transpose(1, 2)
        
        action_logits = self.action_head(output)

        if self.comm_type == CommunicationType.NONE:
            comm_logits = torch.zeros(B, T, N, self.comm_head.out_features, device=x.device).reshape(B, T, N, self.num_comms, -1)
        elif self.comm_type != CommunicationType.AIM:
            comm_logits = self.comm_head(output).reshape(B, T, N, self.num_comms, -1)
        else:
            comm_logits = None
        
        if not has_time_dim:
            action_logits = action_logits.squeeze(1)
            if comm_logits is not None:
                comm_logits = comm_logits.squeeze(1)
                
        return (
            action_logits, comm_logits,
            output, (h_out, c_out)
            # output, (None, None)
        )

    def get_comm_type(self):
        return self.comm_type
    
    def set_is_training(self, is_training):
        self.is_training = is_training

