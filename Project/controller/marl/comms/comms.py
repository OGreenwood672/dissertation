from abc import ABC, abstractmethod
import torch



class Comms(ABC):

    @abstractmethod
    def __init__(self, config, actor_config, device, **kwargs):
        pass

    @abstractmethod
    def percieve(self, x: torch.Tensor):
        pass

    @abstractmethod
    def get_comms_during_rollout(self, x: torch.Tensor):
        pass

    @abstractmethod
    def get_comms_during_update(self, x: torch.Tensor, comms: torch.Tensor):
        pass

    @abstractmethod
    def get_loss(self, x, comm_output, all_comms, true_returns, new_critic_values, targets):
        pass