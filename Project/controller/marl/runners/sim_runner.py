from time import sleep, time
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd

from controller.marl.core.buffer import RolloutBuffer
from controller.marl.core.config import CommunicationType, Config

# from ..core.logger import CommsLogger, ObsLogger

def run_sim(
        system, config: Config, device: torch.device, episodes,
        zero_comms=False, collect_obs_file=None,
        noise=None,
        collect_comms_folder=None, optimal=False
    ):
        
    sim = system['sim']
    actor = system['actor']
    critic = system['critic']


    T = config.training.simulation_timesteps
    W = config.training.worlds_parallised
    N = system['num_agents']
    NC = config.comms.num_comms
    C = config.comms.communication_size
    O = system['agent_obs_shape'][0]
    GO = sim.get_global_obs_dim()

    if optimal:
        comm_type = CommunicationType.NONE
    else:
        comm_type = config.comms.communication_type

    reward_means = []
    rewards = None

    # if collect_obs_file is not None:
    #     OL = ObsLogger(collect_obs_file)

    # if collect_comms_folder is not None:
    #     CL = CommsLogger(collect_comms_folder, config.comms.vocab_size)

    num_codebooks = 0
    if comm_type == CommunicationType.AIM:
        num_codebooks = config.comms.rq_levels * len(actor.comm_protocol.encoder.get_codebooks())
    elif comm_type == CommunicationType.DISCRETE:
        num_codebooks = config.comms.rq_levels

    buffer = RolloutBuffer(
        episodes=episodes,
        timesteps_per_run=T,
        num_worlds_per_run=W,
        num_agents=N,
        num_obs=system['agent_obs_shape'][0],
        num_comms=NC,
        comm_dim=C,
        num_global_obs=GO,# + N * NC * C, 
        hidden_state_size=config.actor.lstm_hidden_size,
        num_codebooks=num_codebooks,
        device=device,
    )

    final_batch_values = []

    for _ in tqdm(range(episodes), desc="Running Simulation"):

        for world_id in range(W):
            sim.reset(world_id)

        comm_choice = np.zeros((W, N, NC, C), dtype=np.float32)
        curr_obs = np.array([sim.get_agents_obs(w) for w in range(W)])
        curr_global_obs = np.array(sim.get_all_global_obs())
        curr_targets = np.array(sim.get_curr_targets())

        if not optimal:
            actor_hidden_states = actor.init_hidden(batch_size=W)

        buf_obs = torch.zeros((T, W, N, system['agent_obs_shape'][0]), dtype=torch.float32, device=device)
        buf_global_obs = torch.zeros((T, W, GO), dtype=torch.float32, device=device)
        buf_targets = torch.zeros((T, W, N, 3), dtype=torch.float32, device=device)
        buf_actions = torch.zeros((T, W, N), dtype=torch.float32, device=device)
        buf_rewards = torch.zeros((T, W, N), dtype=torch.float32, device=device)
        buf_c_values = torch.zeros((T, W, N), dtype=torch.float32, device=device)

        reward_means_per_episode = []
        
        for t in tqdm(range(T), desc="Running Episode", leave=False):

            start = time()

            obs_tensor = torch.from_numpy(curr_obs).float().to(device)
            comm_choice_tensor = torch.from_numpy(comm_choice).float().to(device)
            global_state_tensor = torch.from_numpy(curr_global_obs).float().to(device)
            target_tensor = torch.from_numpy(curr_targets).float().to(device)            

            if not optimal:
                with torch.no_grad():
                    action_logits, lstm_output, actor_hidden_states = actor(obs_tensor, actor_hidden_states, comm_choice_tensor)
                    critic_values = critic(global_state_tensor)


                # * -- Action Choosing --
                greedy_actions = torch.argmax(action_logits, dim=-1)

                comm_output, _, _ = actor.comm_protocol.get_comms_during_rollout(lstm_output)
                
            else:
                greedy_actions = torch.tensor(sim.get_optimal_actions(), device=device)
                critic_values = torch.zeros((W, N), device=device)
                comm_output = torch.zeros((W, N, NC, C), device=device)

            # epsilon-greedy action selection
            if noise is not None and noise > 0.0:
                random_actions = torch.randint(0, sim.get_agent_action_count(), (W, N), device=device)
                mask = torch.rand((W, N), device=device) < noise
                actions = torch.where(mask, random_actions, greedy_actions)
            else:
                actions = greedy_actions
            
            buf_obs[t] = obs_tensor
            buf_global_obs[t] = global_state_tensor
            buf_targets[t] = target_tensor
            buf_actions[t] = actions
            buf_c_values[t] = critic_values

            # Use chosen actions in env
            action_choice = actions.cpu().numpy()
            comm_choice = comm_output.detach().cpu().numpy()

            # if collect_obs_file is not None:
            #     # Curr_ob needs to be a 2d array with comms removed
            #     # critic_values_logs = critic_values.detach().cpu().numpy()[:, :, np.newaxis]
            #     # if rewards is None:
            #     #     reward_logs = np.zeros((W, N, 1))
            #     # else:
            #     #     reward_logs = np.array(rewards).reshape(W, N, 1)
            #     record_logs = np.concatenate(
            #         # [curr_obs, action_logits.cpu().numpy(), critic_values_logs, reward_logs],
            #         [curr_obs, curr_global_obs, action_logits.cpu().numpy(), curr_targets.cpu().numpy()],
            #         axis=-1
            #     ).reshape(W * N, O + GO + 1 + 3)
            #     # obs + actions + critic value

            #     OL.log(record_logs)

            # if collect_comms_folder is not None:
            #     for world in range(W):
            #         for agent in range(N):
            #             index = world * N + agent
            #             # todo: only currently works with discrete - only need with discrete though
            #             CL.log(int(quantised_index[index].detach().cpu().numpy()), curr_obs[world, agent].tolist())
                

            curr_obs, curr_global_obs, curr_targets, rewards = sim.parallel_step(action_choice)
            
            buf_rewards[t] = torch.tensor(rewards, dtype=torch.float32, device=device) 
            reward_means_per_episode.append(np.mean(rewards))

            end = time()
            if end - start < config.training.timestep:
                sleep(config.training.timestep - (end - start))
        
        if not optimal:
            global_state_tensor = torch.tensor(np.stack(curr_global_obs)).float().to(device)
            if global_state_tensor.ndim == 2:
                global_state_tensor = global_state_tensor.unsqueeze(1)
                        
            with torch.no_grad():
                batch_last_vals = critic(global_state_tensor)
            
            if batch_last_vals.ndim == 3:
                batch_last_vals = batch_last_vals.squeeze(1)
            final_batch_values.append(batch_last_vals)
        else:
            final_batch_values.append(torch.zeros((W, N), device=device))

        batch_data = {
            "obs": buf_obs,
            "global_obs": buf_global_obs,
            "targets": buf_targets,
            "actions": buf_actions,
            "rewards": buf_rewards,
            "c_values": buf_c_values,
        }
        buffer.add_episodes(batch_data)
            
        reward_means.append(np.mean(reward_means_per_episode))

    # with torch.no_grad():
    #     final_global_tensor = torch.from_numpy(curr_global_obs).float().to(device)
    #     last_value = critic(final_global_tensor).squeeze(-1) if not optimal else torch.zeros((W, N), device=device)
    

    if collect_obs_file is not None:

        buffer.compute_advantages(torch.cat(final_batch_values, dim=0), config.mappo.gamma, config.mappo.gae_lambda)

        print(f"Exporting Buffer to {collect_obs_file}...")
        
        flat_obs = buffer.obs.view(-1, O).cpu().numpy()
        flat_global_obs = buffer.global_obs.unsqueeze(2).expand(-1, -1, N, -1).reshape(-1, GO).cpu().numpy()# + N * NC * C
        flat_actions = buffer.actions.view(-1, 1).cpu().numpy()
        flat_targets = buffer.targets.view(-1, N, 3).reshape(-1, 3).cpu().numpy()
        flat_c_values = buffer.c_values.view(-1, 1).cpu().numpy()
        flat_returns = buffer.returns.view(-1, 1).cpu().numpy()

        record_logs = np.concatenate([
            flat_obs, 
            flat_global_obs, 
            flat_actions, 
            flat_targets, 
            flat_c_values,
            flat_returns
        ], axis=-1)

        df = pd.DataFrame(record_logs)
        df.to_csv(collect_obs_file, index=False, header=False)
        print("Done!")
    
    return np.mean(reward_means)
