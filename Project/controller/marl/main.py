import argparse
import torch
import torch.nn.functional as F
import numpy as np
import concurrent.futures

from logging_utils.decorators import LoggingFunctionIdentification

import environment.environment as environment
from .checkpoint_manager import CheckpointManager
from .config import MappoConfig
from .buffer import RolloutBuffer
from .models import CommunicationType, PPO_Actor, PPO_Centralised_Critic as PPO_Critic, Quantiser
from .env_wrapper import SimWrapper
from .logger import Logger


def get_args():
    parser = argparse.ArgumentParser(description="MAPPO Controller")
    parser.add_argument("--mode", type=str, default="train", help="Mode to run in", choices=["train", "eval"])
    
    return parser.parse_args()    

def setup(config, device, mode):

    cm = CheckpointManager(config, default_load_previous=mode == "eval")
    SEED = cm.get_seed()

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    sim = SimWrapper(
        environment.Simulation(
            './configs/simulation.yaml',
            worlds_parallised=config.worlds_parallised
        )
    )
    
    num_agents = sim.get_num_agents()
    agent_obs_shape = (sim.get_agent_obs_size(0),)
    global_obs_shape = (sim.get_global_obs_size(0),)
    act_shape = (sim.get_agent_action_count(0),)

    actor = PPO_Actor(
        num_agents, agent_obs_shape[0], act_shape[0], comm_dim=5,
        communication_type=config.communication_type, feature_dim=config.feature_dim,
        lstm_hidden_size=config.lstm_hidden_size
    ).to(device)
    critic = PPO_Critic(num_agents, global_obs_shape[0], config.feature_dim).to(device)

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.learning_rate)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config.learning_rate)

    start_step = 0
    if not cm.is_new_run:
        actor_state, critic_state, actor_optimiser_state, critic_optimizer_state, start_step = cm.load_checkpoint_models()

        actor.load_state_dict(actor_state)
        critic.load_state_dict(critic_state)

        if mode == "train":
            actor_optimizer.load_state_dict(actor_optimiser_state)
            critic_optimizer.load_state_dict(critic_optimizer_state)

        print(f"Loaded checkpoint from step {start_step}")

    return {
        "sim": sim,
        "actor": actor,
        "critic": critic,
        "actor_opt": actor_optimizer,
        "critic_opt": critic_optimizer,
        "start_step": start_step,
        "checkpoint_manager": cm,
        "num_agents": num_agents,
        "agent_obs_shape": (agent_obs_shape,),
        "act_shape": (act_shape,)
    }


def train(system, config, device):

    sim = system['sim']
    actor = system['actor']
    critic = system['critic']
    actor_optimizer = system['actor_opt']
    critic_optimizer = system['critic_opt']

    cm = system['checkpoint_manager']

    num_agents = system['num_agents']
    
    buffer = RolloutBuffer(
        buffer_size=config.buffer_size,
        num_agents=system['num_agents'],
        obs_space_shape=system['agent_obs_shape'],
        action_space_shape=system['act_shape'],
        device=device
    )

    batch_runs = config.buffer_size // config.worlds_parallised
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=config.worlds_parallised)

    logger = Logger(save_folder=cm.get_result_path(), start_step=system["start_step"])

    for current_training_timestep in range(system["start_step"], config.training_timesteps):

        reward_means = []

        final_batch_values = []
        for batch_run in range(batch_runs):
            
            current_episodes_data = [{
                "obs": [], "global_obs": [], "actions": [], "comm": [],
                "log_probs": [], "rewards": [], "c_values": [], "a_hidden_states": []
            } for _ in range(config.worlds_parallised)]
            
            executor.map(sim.parallel_reset, range(config.worlds_parallised))
            curr_obs_for_batch = [sim.get_agents_obs(world_id) for world_id in range(config.worlds_parallised)]
            curr_global_obs_for_batch = [sim.get_global_obs(world_id) for world_id in range(config.worlds_parallised)]
            
            # Reset hidden states for the recurrent model
            actor_hidden_states = actor.init_hidden(batch_size=config.worlds_parallised)
            
            for _ in range(config.simulation_timesteps):

                h = actor_hidden_states[0].reshape(config.worlds_parallised, num_agents, -1)
                c = actor_hidden_states[1].reshape(config.worlds_parallised, num_agents, -1)

                # Move hidden states to CPU for storage
                for w in range(config.worlds_parallised):
                    state_for_buffer = torch.stack([h[w], c[w]], dim=1)
                    
                    current_episodes_data[w]["a_hidden_states"].append(
                        state_for_buffer.detach().cpu().numpy()
                    )

                # Get action logits from the model and update memory and add time dimension for actor model
                obs_tensor = torch.tensor(np.stack(curr_obs_for_batch)).float().to(device)
                global_state_tensor = torch.tensor(np.stack(curr_global_obs_for_batch)).float().to(device)
                with torch.no_grad():
                    action_logits, comm_mean, _, _, _, actor_hidden_states = actor(obs_tensor, actor_hidden_states)
                    values = critic(global_state_tensor)

                # * -- Action Sampling --
                # Basically softmax to determine chosen action
                action_dist = torch.distributions.Categorical(logits=action_logits)
                actions = action_dist.sample()
                action_log_probs = action_dist.log_prob(actions)

                if actor.get_comm_type() == CommunicationType.CONTINUOUS:
                    # * -- Continuous Communication Sampling --
                    comm_std = actor.comm_log_std.exp().expand_as(comm_mean)
                    comm_dist = torch.distributions.Normal(comm_mean, comm_std)
                    comms = comm_dist.sample()
                    comm_log_probs = comm_dist.log_prob(comms).sum(dim=-1) # sum as log(B) + log(C) = log(BC)
                elif actor.get_comm_type() == CommunicationType.DISCRETE:
                    # * -- Discrete Communication --
                    comms = comm_mean
                    comm_log_probs = torch.zeros_like(action_log_probs)
                    
                log_probs = action_log_probs + comm_log_probs

                # Use chosen actions in env
                action_choice = actions.cpu().numpy()
                # Add world id to actions
                step_args = zip(range(config.worlds_parallised), action_choice)
                executor.map(sim.parallel_step, step_args)
                
                rewards = [sim.get_agents_reward(world_id) for world_id in range(config.worlds_parallised)]
                
                # Save to current episodes
                cpu_comms = comms.detach().cpu().numpy()
                cpu_log_probs = log_probs.detach().cpu().numpy()
                cpu_values = values.cpu().numpy()
                for w in range(config.worlds_parallised):
                    current_episodes_data[w]["obs"].append(curr_obs_for_batch[w])
                    current_episodes_data[w]["global_obs"].append(curr_global_obs_for_batch[w])
                    current_episodes_data[w]["actions"].append(action_choice[w])
                    current_episodes_data[w]["comm"].append(cpu_comms[w])
                    current_episodes_data[w]["log_probs"].append(cpu_log_probs[w])
                    current_episodes_data[w]["rewards"].append(rewards[w])
                    current_episodes_data[w]["c_values"].append(cpu_values[w])


                curr_obs_for_batch = [sim.get_agents_obs(world_id, comms[world_id]) for world_id in range(config.worlds_parallised)]
                curr_global_obs_for_batch = [sim.get_global_obs(world_id) for world_id in range(config.worlds_parallised)]

                reward_means.append(np.mean(rewards))

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

        sum_actor_loss = 0
        sum_critic_loss = 0
        sum_entropy_loss = 0
        sum_vq_loss = 0
        sum_codebook_loss = 0
        communication_entropy = 0
        communication_perplexity = 0
        communication_active_codebook_usage = 0
        update_count = 0
        
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
                
                all_action_logits, all_comm_means, codebook_loss, quantised_indices, vq_loss, _ = actor(obs, batch_a_hidden_init)

                # * -- Action Resampling --
                new_action_dist = torch.distributions.Categorical(logits=all_action_logits)
                new_action_log_probs = new_action_dist.log_prob(actions).flatten()
                action_entropy = new_action_dist.entropy().flatten()

                if actor.get_comm_type() == CommunicationType.CONTINUOUS:
                    # * -- Continuous Communication Resampling --
                    new_comm_std = actor.comm_log_std.exp().expand_as(all_comm_means)
                    new_comm_dist = torch.distributions.Normal(all_comm_means, new_comm_std)
                    new_comm_log_probs = new_comm_dist.log_prob(comms).sum(dim=-1).flatten()
                    comm_entropy = new_comm_dist.entropy().sum(dim=-1).flatten()

                elif actor.get_comm_type() == CommunicationType.DISCRETE:
                    # * -- Discrete Communication --
                    new_comm_log_probs = torch.zeros_like(new_action_log_probs)
                    comm_entropy = torch.zeros_like(action_entropy)

                    batch_entropy, batch_perplexity = actor.vq_layer.compute_metrics(quantised_indices)
                    communication_entropy += batch_entropy.item()
                    communication_perplexity += batch_perplexity.item()

                    communication_active_codebook_usage += Quantiser.get_active_codebook_usage(quantised_indices)


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
                total_loss += vq_loss

                sum_actor_loss += actor_loss.item()
                sum_critic_loss += critic_loss.item()
                sum_entropy_loss += entropy_loss.item()
                sum_vq_loss += vq_loss.item()
                if codebook_loss is not None:
                    sum_codebook_loss += codebook_loss.item()

                update_count += 1
                
                # Update
                actor_optimizer.zero_grad()
                critic_optimizer.zero_grad()
                
                total_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                
                actor_optimizer.step()
                critic_optimizer.step()

        if current_training_timestep % config.periodic_save_interval == 0:
            cm.save_checkpoint(actor, critic, actor_optimizer, critic_optimizer, current_training_timestep)
        
        logger.log(
            timestep=current_training_timestep,

            reward_mean=np.mean(reward_means),
            reward_std=np.std(reward_means),

            communication_entropy_mean=communication_entropy / update_count,
            communication_perplexity_mean=communication_perplexity / update_count,
            codebook_loss=sum_codebook_loss / update_count,
            communication_active_codebook_usage=communication_active_codebook_usage / update_count,

            actor_loss=sum_actor_loss / update_count,
            critic_loss=sum_critic_loss / update_count,
            entropy_loss=sum_entropy_loss / update_count,
            vq_loss=sum_vq_loss / update_count,
        )

    logger.close()


def evaluate(system, config, device):
    pass


if __name__ == "__main__":

    args = get_args()

    mappo_config = MappoConfig.from_yaml('./configs/mappo_settings.yaml')


    # Set device to GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    system = setup(mappo_config, device, mode=args.mode)

    if args.mode == "train":
        train(system, mappo_config, device)
    elif args.mode == "eval":
        evaluate(system, mappo_config, device)
