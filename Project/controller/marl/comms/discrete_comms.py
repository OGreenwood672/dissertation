import torch
import torch.nn as nn

from controller.marl.comms.comms import Comms
from controller.marl.config import ActorHyperparameters, CommConfig
from controller.marl.models.quantiser import Quantiser

class DiscreteComms(Comms, nn.Module):

    def __init__(self, config: CommConfig, actor_config: ActorHyperparameters, device: torch.device, **kwargs):
        nn.Module.__init__(self)
        self.config = config
        self.device = device
        self.num_agents = kwargs["num_agents"]
        self.actor_config = actor_config

        self.vq_layer = Quantiser(config.vocab_size, comm_dim=config.communication_size)
        self.comm_head = nn.Linear(actor_config.lstm_hidden_size, config.communication_size * config.num_comms)

        self.vq_loss = torch.zeros(1, device=device)

        self.percieve_out_features = 74
        self.out_features = config.communication_size * config.num_comms
        
        self.to(device)

    def percieve(self, x: torch.Tensor):
        return x
      
    def get_comms_during_rollout(self, x: torch.Tensor):
        B, T, N, _ = x.shape
        
        comm_log_probs = torch.zeros((B * T, N), device=self.device)
        comm_logits = self.comm_head(x).reshape(B, N, self.config.num_comms, self.config.communication_size)

        flat_inputs = comm_logits.reshape(-1, self.config.communication_size)
        quantised_flat, self.vq_loss, _ = self.vq_layer(flat_inputs)
        comm_output = quantised_flat.view(B, N, self.config.num_comms, self.config.communication_size)
        
        return comm_output, comm_output.detach(), comm_log_probs
    
    def get_comms_during_update(self, x: torch.Tensor, comms: torch.Tensor):
        B, T, N, _ = x.shape
        
        # comm_logits = self.comm_head(x).reshape(B, T, N, self.num_comms, -1)
    
        # flat_inputs = comm_logits.view(-1, self.config.communication_size)
        # quantised_flat, vq_loss, quantised_indices = self.vq_layer(flat_inputs)

        new_comm_log_probs = torch.zeros((B * T, N), device=self.device)
        comm_entropy = torch.zeros((B * T, N), device=self.device)

        return _, new_comm_log_probs, comm_entropy
    
    
    def get_loss(self, x, comm_output, all_comms, true_returns, new_critic_values, targets):
        return self.vq_loss.item()