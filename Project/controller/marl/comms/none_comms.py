import torch
import torch.nn as nn

from controller.marl.config import ActorHyperparameters, CommConfig
from controller.marl.models.aim import AIM
from controller.marl.models.encoders.encoder import Encoder


from .comms import Comms



class NoneComms(Comms, nn.Module):

    def __init__(self, config: CommConfig, actor_config: ActorHyperparameters, num_agents: int, device: torch.device):
        nn.Module.__init__(self)


        self.config = config

        self.to(device)

        #! todo
        self.percieve_out_features = 83
        self.out_features = config.communication_size * config.num_comms

    def percieve(self, x: torch.Tensor):
        return x
    
    def get_comm_logits(self, x: torch.Tensor):
        return None
    
    def get_comm(self, x: torch.Tensor):
        return None, torch.tensor([0.0] * self.config.num_comms * self.config.communication_size, device=x.device)
        
    def get_comms_during_rollout(self, x: torch.Tensor):
        B, T, N, _ = x.shape

        comm_output = torch.zeros((B, N, self.config.num_comms, self.config.communication_size), device=x.device)
        comm_log_probs = torch.zeros((B * T, N), device=x.device)
        
        return comm_output, comm_output, comm_log_probs


    def get_comms_during_update(self, x: torch.Tensor, comms: torch.Tensor):
        B, T, N, _ = x.shape

        entropy = torch.zeros(x.shape, device=x.device)
        new_comm_log_probs = torch.zeros((B, T, N), device=x.device)
        return None, new_comm_log_probs, entropy
    
    def auxillery_losses(self, x, comm_output, all_comms, true_returns, new_critic_values, targets):
        return None, None, None