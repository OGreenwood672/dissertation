import numpy as np
import torch


class SimWrapper:
    def __init__(self, sim, comm_dim=5):
        self._sim = sim
        self._num_agents = sim.config.n_agents
        self._comm_dim = comm_dim

    def parallel_step(self, actions, world_comms=None):

        if world_comms is None:
            n_worlds = len(actions) 
            comm_size = self.get_world_comms_size()
            world_comms = np.zeros((n_worlds, comm_size), dtype=np.float32)

        return self._sim.parallel_step(actions, world_comms)

    def reset(self, world_id):
        return self._sim.reset(world_id)

    def get_num_agents(self):
        return self._num_agents

    def get_agent_action_count(self, world_id):
        return self._sim.get_agent_action_count(world_id)

    def get_agent_obs(self, world_id, agent_id, world_comms=None):
        if world_comms is None:
            world_comms = np.zeros(self.get_world_comms_size())
        else:
            if isinstance(world_comms, torch.Tensor):
                world_comms = world_comms.detach().cpu().numpy()
            world_comms = world_comms.flatten()

        return np.concatenate([self._sim.get_agent_obs(world_id, agent_id), world_comms])
    
    def get_world_comms_size(self):
        return self._num_agents * self._comm_dim
    
    def get_agent_obs_size(self, world_id):
        return self._sim.get_agent_obs_size(world_id) + self.get_world_comms_size()

    def get_agent_reward(self, world_id, agent_id):
        return self._sim.get_agent_reward(world_id, agent_id)
    
    def get_agents_obs(self, world_id, world_comms=None):
        return [self.get_agent_obs(world_id, agent_id, world_comms) for agent_id in range(self._num_agents)]
    
    def get_agents_reward(self, world_id):
        return [self.get_agent_reward(world_id, agent_id) for agent_id in range(self._num_agents)]

    def get_global_obs(self, world_id):
        return self._sim.get_global_obs(world_id)
    
    def get_all_global_obs(self):
        return self._sim.get_all_global_obs()

    def get_global_obs_size(self, world_id):
        return self._sim.get_global_obs_size(world_id)
