import numpy as np


class SimWrapper:
    def __init__(self, sim, comm_dim, num_comms):
        self._sim = sim
        self._num_agents = sim.config.n_agents
        self._comm_dim = comm_dim
        self._num_comms = num_comms
        self._obs_dim = sim.get_agent_obs_size(0)
        self._global_obs_dim = sim.get_global_obs_size(0)
        self._action_count = sim.get_agent_action_count(0)
        self._n_worlds = sim.config.n_worlds

    def parallel_step(self, actions):
        n_worlds = len(actions)

        flat_actions = actions.ravel()

        flat_obs, flat_global_obs, flat_targets, flat_rewards = self._sim.parallel_step(flat_actions)

        obs = flat_obs.reshape(n_worlds, self.get_num_agents(), self.get_obs_dim())
        global_obs = flat_global_obs.reshape(n_worlds, self.get_global_obs_dim())# + self.get_world_comms_size())
        targets = flat_targets.reshape(n_worlds, self.get_num_agents(), 3)
        rewards = flat_rewards.reshape(n_worlds, self.get_num_agents())

        return obs, global_obs, targets, rewards

    def reset(self, world_id):
        return self._sim.reset(world_id)

    def get_optimal_actions(self):
        flat_actions = self._sim.get_optimal_actions()
        return flat_actions.reshape(self._n_worlds, self._num_agents)

    def get_num_agents(self):
        return self._num_agents
    
    def get_obs_dim(self):
        return self._obs_dim
    
    def get_global_obs_dim(self):
        return self._global_obs_dim

    def get_agent_action_count(self):
        return self._action_count

    def get_agent_obs(self, world_id, agent_id):
        return self._sim.get_agent_obs(world_id, agent_id)
    
    def get_agent_external_obs_mask(self, world_id):
        return self._sim.get_agent_external_obs_mask(world_id)
    
    def get_agent_obs_mask(self, world_id):
        return self._sim.get_agent_obs_mask(world_id)

    def get_curr_targets(self):
        return self._sim.get_targets().reshape(self._n_worlds, self._num_agents, 3)
    
    def get_world_comms_size(self):
        return self._num_agents * self._comm_dim * self._num_comms
        
    def get_agent_obs_size(self, world_id):
        return self._sim.get_agent_obs_size(world_id)

    def get_agent_reward(self, world_id, agent_id):
        return self._sim.get_agent_reward(world_id, agent_id)
    
    def get_agents_obs(self, world_id):
        return [self.get_agent_obs(world_id, agent_id) for agent_id in range(self._num_agents)]
    
    def get_agents_reward(self, world_id):
        return [self.get_agent_reward(world_id, agent_id) for agent_id in range(self._num_agents)]

    def get_global_obs(self, world_id):
        return self._sim.get_global_obs(world_id)
    
    def get_all_global_obs(self, world_comms):

        num_worlds = len(world_comms)

        flat_obs = self._sim.get_all_global_obs(world_comms.ravel())

        obs = np.array(flat_obs, dtype=np.float32).reshape(num_worlds, self.get_global_obs_dim())# + self.get_world_comms_size())
        
        return obs
    

    def get_global_obs_size(self, world_id):
        return self._sim.get_global_obs_size(world_id)

    def shutdown(self):
        self._sim.shutdown()