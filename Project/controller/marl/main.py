from controller.marl.config import MappoConfig
from logging_utils.decorators import LoggingFunctionIdentification
from time import sleep
import environment.environment as environment
import torch
import torch.nn.functional as F
import numpy as np
import concurrent.futures

from .buffer import RolloutBuffer
from .models import PPO_Actor, PPO_Centralised_Critic as PPO_Critic
from .env_wrapper import SimWrapper



@LoggingFunctionIdentification("CONTROLLER")
def main():
    print("starting program")

    mappo_config = MappoConfig.from_yaml('./configs/mappo_settings.yaml')
    sim = SimWrapper(
        environment.Simulation(
            './configs/simulation.yaml',
            worlds_parallised=mappo_config.worlds_parallised
            )
        )
    
    num_agents = sim.get_num_agents()
    agent_obs_shape = (sim.get_agent_obs_size(0),)
    global_obs_shape = (sim.get_global_obs_size(0),)
    act_shape = (sim.get_agent_action_count(0),)

    batch_runs = mappo_config.buffer_size // mappo_config.worlds_parallised

    # Set device to GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    actor = PPO_Actor(num_agents, agent_obs_shape[0], act_shape[0], comm_dim=5, lstm_hidden_size=mappo_config.lstm_hidden_size).to(device)
    critic = PPO_Critic(num_agents, global_obs_shape[0]).to(device)

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=0.001)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=0.001)

    buffer = RolloutBuffer(
        buffer_size=mappo_config.buffer_size,
        num_agents=num_agents,
        obs_space_shape=agent_obs_shape,
        action_space_shape=act_shape,
        device=device
    )

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=mappo_config.worlds_parallised)

    for i in range(mappo_config.training_timesteps):

        final_batch_values = []
        for batch_run in range(batch_runs):
            
            current_episodes_data = [{
                "obs": [], "global_obs": [], "actions": [], "comm": [],
                "log_probs": [], "rewards": [], "c_values": [], "a_hidden_states": []
            } for _ in range(mappo_config.worlds_parallised)]
            
            executor.map(sim.parallel_reset, range(mappo_config.worlds_parallised))
            curr_obs_for_batch = [sim.get_agents_obs(world_id) for world_id in range(mappo_config.worlds_parallised)]
            curr_global_obs_for_batch = [sim.get_global_obs(world_id) for world_id in range(mappo_config.worlds_parallised)]
            
            # Reset hidden states for the recurrent model
            actor_hidden_states = actor.init_hidden(batch_size=mappo_config.worlds_parallised)
            
            for _ in range(mappo_config.simulation_timesteps):

                h = actor_hidden_states[0].reshape(mappo_config.worlds_parallised, num_agents, -1)
                c = actor_hidden_states[1].reshape(mappo_config.worlds_parallised, num_agents, -1)

                # Move hidden states to CPU for storage
                for w in range(mappo_config.worlds_parallised):
                    state_for_buffer = torch.stack([h[w], c[w]], dim=1)
                    
                    current_episodes_data[w]["a_hidden_states"].append(
                        state_for_buffer.detach().cpu().numpy()
                    )

                # Get action logits from the model and update memory and add time dimension for actor model
                obs_tensor = torch.tensor(np.stack(curr_obs_for_batch)).float().to(device)
                global_state_tensor = torch.tensor(np.stack(curr_global_obs_for_batch)).float().to(device)
                with torch.no_grad():
                    action_logits, comm_mean, actor_hidden_states = actor(obs_tensor, actor_hidden_states)
                    values = critic(global_state_tensor)

                # * -- Action Sampling --
                # Basically softmax to determine chosen action
                action_dist = torch.distributions.Categorical(logits=action_logits)
                actions = action_dist.sample()
                action_log_probs = action_dist.log_prob(actions)

                # * -- Continuous Communication Sampling --
                comm_std = actor.comm_log_std.exp().expand_as(comm_mean)
                comm_dist = torch.distributions.Normal(comm_mean, comm_std)
                comms = comm_dist.sample()
                comm_log_probs = comm_dist.log_prob(comms).sum(dim=-1) # sum as log(B) + log(C) = log(BC)

                log_probs = action_log_probs + comm_log_probs

                # Use chosen actions in env
                action_choice = actions.cpu().numpy()
                # Add world id to actions
                step_args = zip(range(mappo_config.worlds_parallised), action_choice)
                executor.map(sim.parallel_step, step_args)
                
                rewards = [sim.get_agents_reward(world_id) for world_id in range(mappo_config.worlds_parallised)]

                # Save to current episodes
                for w in range(mappo_config.worlds_parallised):
                    current_episodes_data[w]["obs"].append(curr_obs_for_batch[w])
                    current_episodes_data[w]["global_obs"].append(curr_global_obs_for_batch[w])
                    current_episodes_data[w]["actions"].append(action_choice[w])
                    current_episodes_data[w]["comm"].append(comms[w].detach().cpu().numpy())
                    current_episodes_data[w]["log_probs"].append(log_probs[w].detach().cpu().numpy())
                    current_episodes_data[w]["rewards"].append(rewards[w])
                    current_episodes_data[w]["c_values"].append(values[w].cpu())


                curr_obs_for_batch = [sim.get_agents_obs(world_id, comms[world_id]) for world_id in range(mappo_config.worlds_parallised)]
                curr_global_obs_for_batch = [sim.get_global_obs(world_id) for world_id in range(mappo_config.worlds_parallised)]

            global_state_tensor = torch.tensor(np.stack(curr_global_obs_for_batch)).float().to(device)
            if global_state_tensor.ndim == 2:
                global_state_tensor = global_state_tensor.unsqueeze(1)
                        
            with torch.no_grad():
                batch_last_vals = critic(global_state_tensor)
            
            # Remove singleton time dimension from final values
            if batch_last_vals.ndim == 3:
                batch_last_vals = batch_last_vals.squeeze(1)
            final_batch_values.append(batch_last_vals)

            # Save to buffer
            for w in range(mappo_config.worlds_parallised):
                buffer.add_episode(current_episodes_data[w])
            
            print(f"Collected batch {batch_run+1}/{batch_runs} (Total {mappo_config.worlds_parallised * (batch_run+1)} episodes)")

        print("Buffer is full. Creating batch...")

        buffer.flatten_episodes_to_batch()
        buffer.compute_advantages(torch.cat(final_batch_values, dim=0), mappo_config.gamma, mappo_config.gae_lambda)
        
        for _ in range(mappo_config.ppo_epochs):

            # Perhaps work out worlds_parallised = how large GPU mem is
            for batch in buffer.get_minibatches():
                
                # [B, T, N, Obs]
                obs = batch["obs"]

                # [B, T, N, Global Obs]
                global_obs = batch["global_obs"]
                # [B, T, N, Actions]
                actions = batch["actions"]
                # [B, T, N, Comm]
                comms = batch["comm"]

                # [B, T, N]
                old_log_probs = batch["log_probs"]
                advantages = batch["advantages"]
                returns = batch["returns"]

                h_c_init = batch["a_hidden_states"][:, 0] 
                # [1, B * N, Hidden]
                h_0 = h_c_init[:, :, 0, :].reshape(-1, mappo_config.lstm_hidden_size).unsqueeze(0)
                c_0 = h_c_init[:, :, 1, :].reshape(-1, mappo_config.lstm_hidden_size).unsqueeze(0)

                batch_a_hidden_init = (h_0.contiguous(), c_0.contiguous())
                
                # Calculate all new critic values
                flat_global_obs = global_obs.view(-1, *global_obs.shape[2:])
                new_values = critic(flat_global_obs).reshape(-1)
                
                all_action_logits, all_comm_means, _ = actor(obs, batch_a_hidden_init)

                # * -- Action Resampling --
                new_action_dist = torch.distributions.Categorical(logits=all_action_logits)
                new_action_log_probs = new_action_dist.log_prob(actions).flatten()
                action_entropy = new_action_dist.entropy().flatten()

                # * -- Continuous Communication Resampling --
                new_comm_std = actor.comm_log_std.exp().expand_as(all_comm_means)
                new_comm_dist = torch.distributions.Normal(all_comm_means, new_comm_std)
                new_comm_log_probs = new_comm_dist.log_prob(comms).sum(dim=-1).flatten()
                comm_entropy = new_comm_dist.entropy().sum(dim=-1).flatten()

                entropy = action_entropy + comm_entropy
                new_log_probs = new_action_log_probs + new_comm_log_probs                
                                
                old_log_probs = old_log_probs.flatten()
                returns = returns.flatten()
                advantages = advantages.flatten()
                actions = actions.flatten()

                # Critic Loss
                critic_loss = F.mse_loss(new_values, returns)
                
                # PPO Actor Loss
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                log_ratio = new_log_probs - old_log_probs
                ratio = log_ratio.exp()
                
                loss1 = advantages * ratio
                loss2 = advantages * torch.clamp(ratio, 1.0 - mappo_config.clip_coef, 1.0 + mappo_config.clip_coef)
                actor_loss = -torch.min(loss1, loss2).mean()
                
                entropy_loss = -entropy.mean()
                
                total_loss = actor_loss + mappo_config.vf_coef * critic_loss + mappo_config.ent_coef * entropy_loss
                
                # Update
                actor_optimizer.zero_grad()
                critic_optimizer.zero_grad()
                
                total_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                
                actor_optimizer.step()
                critic_optimizer.step()

if __name__ == "__main__":
    main()