

class SimWrapper:
    def __init__(self, sim):
        self._sim = sim
        self._num_agents = sim.config.n_agents

    def parallel_step(self, args):
        world_id, action = args
        return self._sim.step(world_id, action)

    def parallel_reset(self, world_id):
        return self._sim.reset(world_id)

    def get_num_agents(self):
        return self._num_agents

    def get_agent_action_count(self, world_id):
        return self._sim.get_agent_action_count(world_id)

    def get_agent_obs(self, world_id, agent_id):
        return self._sim.get_agent_obs(world_id, agent_id)
    
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

    def get_global_obs_size(self, world_id):
        return self._sim.get_global_obs_size(world_id)
