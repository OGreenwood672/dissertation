import torch
import numpy as np


class RolloutBuffer:
    """
    A buffer for storing full episodes.
    It collects `buffer_size` (number of episodes)
    """
    def __init__(self, buffer_size, num_agents, obs_space_shape, action_space_shape, device='cpu'):
        # buffer_size is the number of episodes to store
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        self.obs_space_shape = obs_space_shape
        self.action_space_shape = action_space_shape
        self.device = device
        
        self.episodes = [None] * buffer_size
        # self.pos tracks how many episodes stored so far
        self.pos = 0

        self.batch = {}


    def add_episode(self, episode_data):
        """
        Adds one complete episode to the buffer.
        'episode_data' is a dictionary where each key holds a list of step-data.
        episode_data must have the following keys:
            "obs" - observations from the agents
            "actions" - actions taken by the agents
            "log_prob" - the log probabilities of the actions taken
            "rewards" - the rewards received by the agents
            "c_values" - the critic's values for each observation
            "a_hidden_states" - actors hidden states for each agent
            "c_hidden_states" - critics hidden states for each agent

            May later add "done" as a keyword to use in compute_advantages to determine
            if a simulation is finished early
        """
        if self.pos >= self.buffer_size:
            print("Warning: Buffer is full. Overwriting old episode.")
            self.pos = 0 # Reset position if full

        # Stack the data collected at each step into one tensor for the whole episode
        processed_episode = {}
        for key, values in episode_data.items():
            processed_episode[key] = torch.from_numpy(np.stack(values)).float().to(self.device)
            
        self.episodes[self.pos] = processed_episode
        self.pos += 1

    def flatten_episodes_to_batch(self):
        """
        Collates all episodes into a single batch.
        """
        assert self.pos == self.buffer_size, "Buffer is not yet full!"
        self.pos = 0 # Reset buffer position
        
        # Check all the episodes are the same length
        assert all([len(ep["obs"]) == len(self.episodes[0]["obs"]) for ep in self.episodes])

        keys = self.episodes[0].keys()
        self.batch = {}
        for k in keys:
            self.batch[k] = torch.stack([ep[k] for ep in self.episodes])
        
        # Clear the episodes list
        self.episodes = [None] * self.buffer_size


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
        
        rewards = self.batch["rewards"]
        values = self.batch["c_values"]
        
        B, T, N = rewards.shape
        
        advantages = torch.zeros_like(rewards, device=self.device)
        returns = torch.zeros_like(rewards, device=self.device)

        gae = 0
        next_value = last_value
        
        for t in reversed(range(T)):
            r = rewards[:, t, :]
            v = values[:, t, :]
            
            # Calculate the 1-step TD Error (delta)
            # delta = reward + gamma * V(s_t+1) - V(s_t)
            delta = r + gamma * next_value - v
            
            # Calculate the GAE advantage
            # gae = delta + gamma * lambda * gae
            gae = delta + gamma * gae_lambda * gae
            
            # Set the advantage for this timestep
            advantages[:, t, :] = gae
            
            # Set the return for this timestep (the critic target)
            # Return = Advantage + Value
            returns[:, t, :] = gae + v
            
            # Update next_value for the next iteration
            # The next value for step t-1 is the value at step t
            next_value = v
        
        self.batch["advantages"] = advantages
        self.batch["returns"] = returns

    
    def get_minibatches(self, batch_size=32):
        """
        A generator that yields shuffled minibatches from the full batch.
        Assumes self.batch and self.advantages/self.returns are populated.
        """
                
        b_obs = self.batch["obs"]
        B, T, N, _ = b_obs.shape
        total_timesteps = B * T

        flat_data = {}
        for k, v in self.batch.items():
            flat_data[k] = v.reshape(total_timesteps, N, *v.shape[3:])

        flat_advantages = self.batch["advantages"].reshape(total_timesteps, N)
        flat_returns = self.batch["returns"].reshape(total_timesteps, N)

        # Shuffle the Timesteps
        indices = torch.randperm(total_timesteps).to(self.device)
        
        for start in range(0, total_timesteps, batch_size):
            end = start + batch_size
            
            mb_indices = indices[start:end]
            
            mb = {}
            for k, v in flat_data.items():
                mb[k] = v[mb_indices]
                
            mb["advantages"] = flat_advantages[mb_indices]
            mb["returns"] = flat_returns[mb_indices]
            
            yield mb
