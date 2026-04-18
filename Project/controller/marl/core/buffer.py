import torch


class RolloutBuffer:
    """
    A buffer for storing full episodes.
    It collects `buffer_size` (number of episodes)
    """
    def __init__(self, episodes, timesteps_per_run, num_worlds_per_run, num_agents, num_obs, comm_dim, num_comms, num_global_obs, hidden_state_size, num_codebooks=None, device='cpu'):
        
        # assert(total_runs % num_worlds_per_run == 0, "total_runs must be divisible by num_worlds_per_run")

        self.device = device

        self.buffer_size = episodes * num_worlds_per_run

        self.num_worlds_per_run = num_worlds_per_run
        self.num_agents = num_agents

        self.timesteps_per_run = timesteps_per_run

        self.num_obs = num_obs
        
        # self.pos tracks how many episodes stored so far
        self.pos = 0

        self.obs = torch.zeros((self.buffer_size, timesteps_per_run, num_agents, num_obs), device=device)
        self.global_obs = torch.zeros((self.buffer_size, timesteps_per_run, num_global_obs), device=device)
        self.targets = torch.zeros((self.buffer_size, timesteps_per_run, num_agents, 3), device=device)
        self.actions = torch.zeros((self.buffer_size, timesteps_per_run, num_agents), device=device)
        self.rewards = torch.zeros((self.buffer_size, timesteps_per_run, num_agents), device=device)
        self.action_log_probs = torch.zeros((self.buffer_size, timesteps_per_run, num_agents), device=device)
        self.comm_log_probs = torch.zeros((self.buffer_size, timesteps_per_run, num_agents), device=device)
        self.c_values = torch.zeros((self.buffer_size, timesteps_per_run, num_agents), device=device)
        if num_codebooks == 0:
            self.comm = torch.zeros((self.buffer_size, timesteps_per_run, num_agents, num_comms, comm_dim), device=device)
        else:
            self.comm = torch.zeros((self.buffer_size, timesteps_per_run, num_agents, num_comms, num_codebooks), device=device)
        self.a_hidden_states = torch.zeros((self.buffer_size, timesteps_per_run, num_agents, 2, hidden_state_size), device=device)

        # Results of GAE
        self.advantages = torch.zeros((self.buffer_size, timesteps_per_run, num_agents), device=device)
        self.returns = torch.zeros((self.buffer_size, timesteps_per_run, num_agents), device=device)
    
    def add_episodes(self, episode_data):
        """
        Adds one complete episode to the buffer.
        'episode_data' is a dictionary where each key holds a list of step-data.
        episode_data must have the following keys:
            "obs" - observations from the agents
            "actions" - actions taken by the agents
            "comm" - communication vector broadcasted by actor
            "log_prob" - the log probabilities of the actions taken
            "rewards" - the rewards received by the agents
            "c_values" - the critic's values for each observation
            "a_hidden_states" - actors hidden states for each agent

            May later add "done" as a keyword to use in compute_advantages to determine
            if a simulation is finished early
        """
        
        if self.pos + self.num_worlds_per_run > self.buffer_size:
            raise ValueError("Buffer overflow! Trying to insert more episodes than buffer size.")


        def process(key, curr_store):
            data = episode_data[key].to(self.device).transpose(0, 1)            
            curr_store[self.pos : self.pos + self.num_worlds_per_run] = data
            return curr_store
        
        for key in episode_data.keys():
            setattr(self, key, process(key, getattr(self, key)))

        self.pos += self.num_worlds_per_run


    def compute_advantages(self, last_value, gamma, gae_lambda):
        """
        Computes advantages and returns using GAE.
        This function operates on the *entire* batch of data, which is
        assumed to be stored in self.batch.

        last_value: The critic's value of the *next* state after the
                    last state in the buffer. Shape: [BatchSize, NumAgents]
        gamma: The discount factor
        gae_lambda: The GAE lambda
        """
                        
        gae = 0
        
        for t in reversed(range(self.timesteps_per_run)):
            r = self.rewards[:, t]
            v = self.c_values[:, t]
            
            # Calculate the 1-step TD Error (delta)
            # delta = reward + gamma * V(s_t+1) - V(s_t)
            delta = r + gamma * last_value - v
            
            # Calculate the GAE advantage
            # gae = delta + gamma * lambda * gae
            gae = delta + gamma * gae_lambda * gae
            
            # Set the advantage for this timestep
            self.advantages[:, t, :] = gae
            
            # Set the return for this timestep (the critic target)
            # Return = Advantage + Value
            self.returns[:, t, :] = gae + v
            
            # Update last_val for the next iteration
            # The last value for step t-1 is the value at step t
            last_value = v
        
    
    def get_minibatches(self, batch_size=16):
        """
        A generator that yields shuffled minibatches from the full batch.
        Assumes self.advantages/self.returns are populated.
        """
                        
        indices = torch.randperm(self.pos).to(self.device)    

        for start in range(0, self.pos, batch_size):
            end = start + batch_size
            mb_indices = indices[start:end]

            yield {
                "obs": self.obs[mb_indices],
                "targets": self.targets[mb_indices],
                "actions": self.actions[mb_indices],
                "rewards": self.rewards[mb_indices],
                "action_log_probs": self.action_log_probs[mb_indices].view(-1),
                "comm_log_probs": self.comm_log_probs[mb_indices].view(-1),
                "c_values": self.c_values[mb_indices],
                "advantages": self.advantages[mb_indices].view(-1),
                "returns": self.returns[mb_indices],
                "a_hidden_states": self.a_hidden_states[mb_indices],
                "global_obs": self.global_obs[mb_indices],
                "comm": self.comm[mb_indices],
            }    
    
    def reset(self):
        self.pos = 0