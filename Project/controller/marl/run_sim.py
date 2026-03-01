from time import sleep, time
import numpy as np
import torch
from tqdm import tqdm

from controller.marl.config import CommunicationType

from .logger import CommsLogger, ObsLogger

def run_sim(
        system, config, device, episodes,
        zero_comms=False, collect_obs_file=None,
        noise=None,
        collect_comms_folder=None
    ):
        
    sim = system['sim']
    actor = system['actor'].eval()
    critic = system['critic'].eval()


    T = config.simulation_timesteps
    W = config.worlds_parallised
    N = system['num_agents']
    NC = config.num_comms
    C = config.communication_size

    reward_means = []

    if collect_obs_file is not None:
        OL = ObsLogger(collect_obs_file)

    if collect_comms_folder is not None:
        CL = CommsLogger(collect_comms_folder, config.vocab_size)


    for _ in tqdm(range(episodes), desc="Running Simulation"):

        for world_id in range(W):
            sim.reset(world_id)

        comm_choice = np.zeros((W, N, NC, C), dtype=np.float32)
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
                action_logits, comm_logits, lstm_output, actor_hidden_states = actor(obs_tensor, actor_hidden_states)
                if collect_obs_file is not None:
                    global_state_tensor = torch.from_numpy(curr_global_obs).float().to(device)
                    critic_values = critic(global_state_tensor)


            # * -- Action Choosing --
            greedy_actions = torch.argmax(action_logits, dim=-1)

            # epsilon-greedy action selection
            if noise is not None and noise > 0.0:
                random_actions = torch.randint(0, action_logits.shape[-1], size=(W, N), device=device)
                mask = torch.rand((W, N), device=device) < noise
                actions = torch.where(mask, random_actions, greedy_actions)
            else:
                actions = greedy_actions

            if actor.get_comm_type() == CommunicationType.CONTINUOUS:
                # * -- Continuous Communication Sampling --
                comm_std = actor.comm_log_std.exp().squeeze(1).expand_as(comm_logits)
                comm_dist = torch.distributions.Normal(comm_logits, comm_std)
                comm_output = comm_dist.sample()

            elif actor.get_comm_type() == CommunicationType.DISCRETE:
                # * -- Discrete Communication --
                flat_inputs = comm_logits.view(-1, C)
                quantised_flat, _, quantised_index = actor.vq_layer(flat_inputs)
                comm_output = quantised_flat.view(W, N, NC, C)

            elif actor.get_comm_type() == CommunicationType.AIM:
                # * -- AIM Communication --
                codewords = torch.zeros((W, N, NC, actor.hq_levels, C), device=device)

                comm_conditions = lstm_output.squeeze(1)

                for i in range(actor.hq_levels):

                    comm_logits = getattr(actor, f"comm_head_{i}")(comm_conditions).reshape(W, N, NC, actor.vocab_size)

                    dist = torch.distributions.Categorical(logits=comm_logits)
                    quantised_index = dist.sample()

                    codewords[:, :, :, i, :] = actor.aim_codebook[i, quantised_index].view(W, N, NC, C)

                    comm_conditions = torch.cat([comm_conditions, codewords[:, :, :, i, :].reshape(W, N, NC * C)], dim=-1)

                comm_output = codewords.sum(dim=-2)

            elif actor.get_comm_type() == CommunicationType.NONE or zero_comms:
                comm_output = torch.zeros((W, N, NC, C), device=device) 

            # Use chosen actions in env
            action_choice = actions.cpu().numpy()
            comm_choice = comm_output.detach().cpu().numpy()

            if collect_obs_file is not None:
                # Curr_ob needs to be a 2d array with comms removed
                obs_logs = curr_obs[:, :, :-sim.get_world_comms_size()]
                critic_values_logs = critic_values.detach().cpu().numpy()[:, :, np.newaxis]
                
                record_logs = np.concatenate(
                    [obs_logs, action_logits.cpu().numpy(), critic_values_logs],
                    axis=-1
                ).reshape(W * N, sim.get_obs_dim() + sim.get_agent_action_count() + 1)
                # obs + actions + critic value

                OL.log(record_logs)

            if collect_comms_folder is not None:
                for world in range(W):
                    for agent in range(N):
                        index = world * N + agent
                        # todo: only currently works with discrete - only need with discrete though
                        CL.log(int(quantised_index[index].detach().cpu().numpy()), curr_obs[world, agent].tolist())
                

            curr_obs, curr_global_obs, rewards = sim.parallel_step(action_choice, comm_choice)
                            
            reward_means_per_episode.append(np.mean(rewards))

            end = time()
            if end - start < config.timestep:
                sleep(config.timestep - (end - start))
            
        reward_means.append(np.mean(reward_means_per_episode))

    if collect_obs_file is not None:
        OL.close()
    
    return reward_means
