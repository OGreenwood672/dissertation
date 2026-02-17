from time import sleep, time
import numpy as np
import torch
from tqdm import tqdm

from .logger import CommsLogger, ObsLogger

def run_sim(
        system, config, device, episodes,
        zero_comms=False, collect_obs_file=None,
        collect_comms_folder=None
    ):
        
    sim = system['sim']
    actor = system['actor'].eval()
    critic = system['critic'].eval()


    T = config.simulation_timesteps
    W = config.worlds_parallised
    N = system['num_agents']
    C = config.communication_size

    reward_means = []

    if collect_obs_file is not None:
        OL = ObsLogger(collect_obs_file)

    if collect_comms_folder is not None:
        CL = CommsLogger(collect_comms_folder, config.vocab_size)


    for _ in tqdm(range(episodes), desc="Running Simulation"):

        for world_id in range(W):
            sim.reset(world_id)

        comm_choice = np.zeros((W, N, C), dtype=np.float32)
        curr_obs = np.array([sim.get_agents_obs(w, comm_choice[w]) for w in range(W)])
        curr_global_obs = np.array(sim.get_all_global_obs(comm_choice))            

        # Reset hidden states for the recurrent model
        actor_hidden_states = actor.init_hidden(batch_size=W)

        reward_means_per_episode = []
        
        for _ in tqdm(range(T), desc="Running Episode", leave=False):

            start = time()

            # Get action logits from the model and update memory and add time dimension for actor model
            obs_tensor = torch.from_numpy(curr_obs).float().to(device)
            with torch.no_grad():
                action_logits, comms, _, quantised_index, _, actor_hidden_states = actor(obs_tensor, actor_hidden_states)
                if collect_obs_file is not None:
                    global_state_tensor = torch.from_numpy(curr_global_obs).float().to(device)
                    critic_values = critic(global_state_tensor)


            # * -- Action Choosing --
            actions = torch.argmax(action_logits, dim=-1)

            if zero_comms:
                comms = torch.zeros_like(comms)

            # Use chosen actions in env
            action_choice = actions.cpu().numpy()
            comm_choice = comms.detach().cpu().numpy()

            if collect_obs_file is not None:
                # Curr_ob needs to be a 2d array with comms removed
                obs_logs = curr_obs[:, :, :-sim.get_world_comms_size()]
                critic_values_logs = critic_values.detach().cpu().numpy()[:, :, np.newaxis]
                
                record_logs = np.concatenate(
                    [obs_logs, action_logits.cpu().numpy(), critic_values_logs],
                    axis=-1
                ).reshape(W * N, sim.get_obs_dim() + sim.get_agent_action_count() + 1)

                OL.log(record_logs)

            if collect_comms_folder is not None:
                for world in range(W):
                    for agent in range(N):
                        index = world * N + agent
                        CL.log(int(quantised_index[index].detach().cpu().numpy()), curr_obs[world, agent].tolist())
                

            curr_obs, rewards = sim.parallel_step(action_choice, comm_choice)
            curr_global_obs = np.array(sim.get_all_global_obs(comm_choice))
                            
            reward_means_per_episode.append(np.mean(rewards))

            end = time()
            if end - start < config.timestep:
                sleep(config.timestep - (end - start))
            
        reward_means.append(np.mean(reward_means_per_episode))

    if collect_obs_file is not None:
        OL.close()
    
    return reward_means
