

from torch import nn

from controller.marl.core.config import CommConfig, CriticHyperparameters


class PPO_Critic(nn.Module):
    def __init__(
            self,
            num_agents, global_obs_dim,
            critic_config: CriticHyperparameters, comm_config: CommConfig
        ):
        super().__init__()

        self.input_dim = global_obs_dim# + comm_config.communication_size * num_agents * comm_config.num_comms

        feature_dim = critic_config.feature_dim

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
        assert x.shape[-1] == self.input_dim, f"Input has wrong observation size, expected {self.input_dim}, got {x.shape[-1]}"

        features = self.body(x)
        
        values = self.value_head(features)
        
        return values
    
