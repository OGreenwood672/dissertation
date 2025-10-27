



class EnvWrapper:
    def __init__(self, env):
        self.env = env
        self.num_agents = self.env.n_agents
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self):
        return self.env.reset()

    def step(self, actions):
        
        next_obs, rewards = self.env.step(actions)

        return next_obs, rewards