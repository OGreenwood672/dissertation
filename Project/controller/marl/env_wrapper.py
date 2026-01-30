import numpy as np
import torch


class SimWrapper:
    def __init__(self, sim, comm_dim=5):
        self._sim = sim
        self._num_agents = sim.config.n_agents
        self._comm_dim = comm_dim
        self._obs_dim = sim.get_agent_obs_size(0)

    def parallel_step(self, actions, world_comms):
        n_worlds = len(actions)

        flat_comms = world_comms.astype(np.float32).ravel()
        flat_actions = actions.ravel()

        flat_obs, flat_rewards = self._sim.parallel_step(flat_actions, flat_comms)

        obs = np.array(flat_obs, dtype=np.float32).reshape(n_worlds, self.get_num_agents(), self.get_obs_dim() + self.get_world_comms_size())
        rewards = np.array(flat_rewards, dtype=np.float32).reshape(n_worlds, self.get_num_agents())

        return obs, rewards

    def reset(self, world_id):
        return self._sim.reset(world_id)

    def get_num_agents(self):
        return self._num_agents
    
    def get_obs_dim(self):
        return self._obs_dim

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
