from time import sleep, time
import torch.nn.functional as F
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from .buffer import RolloutBuffer
from .config import CommunicationType
from .models import AIM, Quantiser
from .logger import Logger

# def get_grad_norm(params):
#     grads = [p.grad.detach() for p in params if p.grad is not None]
#     if not grads:
#         return 0.0
#     return float(torch.norm(torch.stack([torch.norm(g, 2) for g in grads]), 2).item())

def train(system, config, device):

    sim = system['sim']
    actor = system['actor']
    critic = system['critic']
    actor_optimizer = system['actor_opt']
    critic_optimizer = system['critic_opt']

    cm = system['checkpoint_manager']

    T = config.simulation_timesteps
    W = config.worlds_parallised
    N = system['num_agents']
    C = config.communication_size
    NC = config.num_comms

    A = system['act_shape'][0]

    comm_type = config.communication_type

    batch_runs = config.buffer_size // W

    buffer = RolloutBuffer(
        total_runs=config.buffer_size,
        timesteps_per_run=T,
        num_worlds_per_run=W,
        num_agents=N,
        num_obs=system['agent_obs_shape'][0],
        num_comms=NC,
        comm_dim=C,
        num_global_obs=sim.get_global_obs_size(0) + N * NC * C, 
        hidden_state_size=config.lstm_hidden_size,
        hq_levels=actor.hq_levels,
        device=device,
    )

    logger = Logger(save_folder=cm.get_result_path(), config=config, start_step=system["start_step"])

    actor = torch.jit.script(actor)
    for current_training_timestep in range(system["start_step"], config.training_timesteps):

        reward_means = []

        final_batch_values = []
        buffer.reset()

        for batch_run in range(batch_runs):

            logger.update_step(current_training_timestep, batch_run, batch_runs, "Collecting Data")

            buf_obs = torch.zeros((T, W, N, system['agent_obs_shape'][0]), dtype=torch.float32, device=device)
            buf_global_obs = torch.zeros((T, W, sim.get_global_obs_size(0) + N * NC * C), dtype=torch.float32, device=device)
            buf_targets = torch.zeros((T, W, N * 3), dtype=torch.float32, device=device)
            buf_actions = torch.zeros((T, W, N), dtype=torch.float32, device=device)
            buf_rewards = torch.zeros((T, W, N), dtype=torch.float32, device=device)
            buf_action_log_probs = torch.zeros((T, W, N), dtype=torch.float32, device=device)
            buf_comm_log_probs = torch.zeros((T, W, N), dtype=torch.float32, device=device)
            buf_c_values = torch.zeros((T, W, N), dtype=torch.float32, device=device)
            buf_hidden_states = torch.zeros((T, W, N, 2, config.lstm_hidden_size), dtype=torch.float32, device=device)

            if actor.hq_levels is not None:
                buf_comms = torch.zeros((T, W, N, NC, actor.hq_levels), dtype=torch.float32, device=device)
            else:
                buf_comms = torch.zeros((T, W, N, NC, C), dtype=torch.float32, device=device)

            for world_id in range(W):
                sim.reset(world_id)

            comm_choice = np.zeros((W, N, NC, C), dtype=np.float32)
            curr_obs = np.array([sim.get_agents_obs(w) for w in range(W)])
            curr_global_obs = np.array(sim.get_all_global_obs(comm_choice)) 

            # Reset hidden states for the recurrent model
            actor_hidden_states = actor.init_hidden(batch_size=W)
            
            for t in range(T):

                start = time()

                h = actor_hidden_states[0].reshape(W, N, -1)
                c = actor_hidden_states[1].reshape(W, N, -1)
                hidden_stacked = torch.stack([h, c], dim=2)
                buf_hidden_states[t] = hidden_stacked.detach()

                # Get action logits from the model and update memory and add time dimension for actor model
                obs_tensor = torch.from_numpy(curr_obs).float().to(device)
                global_state_tensor = torch.from_numpy(curr_global_obs).float().to(device)
                comm_choice_tensor = torch.from_numpy(comm_choice).float().to(device)
                with torch.no_grad():
                    action_logits, comm_logits, lstm_output, actor_hidden_states = actor(obs_tensor, actor_hidden_states, comm_choice_tensor if comm_type != CommunicationType.NONE else None)
                    # values = critic(global_state_tensor)

                # * -- Action Sampling --
                # Basically softmax to determine chosen action
                # action_dist = torch.distributions.Categorical(logits=action_logits)
                # actions = action_dist.sample()
                # action_log_probs = action_dist.log_prob(actions)

                action_probs = F.softmax(action_logits.float(), dim=-1)
                action_log_probs_all = F.log_softmax(action_logits.float(), dim=-1)
                actions = torch.multinomial(action_probs.view(-1, A), 1).view(W, N)
                action_log_probs = action_log_probs_all.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)

                if comm_type == CommunicationType.CONTINUOUS:
                    # * -- Continuous Communication Sampling --
                    comm_std = actor.comm_log_std.exp().squeeze(1).expand_as(comm_logits)
                    comm_dist = torch.distributions.Normal(comm_logits, comm_std)
                    comm_output = comm_dist.sample()
                    comm_log_probs = comm_dist.log_prob(comm_output).sum(dim=[-2, -1]) # sum as log(B) + log(C) = log(BC)

                    buf_comms[t] = comm_output.detach()

                elif comm_type == CommunicationType.DISCRETE:
                    # * -- Discrete Communication --
                    comm_log_probs = torch.zeros_like(action_log_probs)

                    flat_inputs = comm_logits.view(-1, C)
                    quantised_flat, vq_loss, quantised_index = actor.vq_layer(flat_inputs)
                    comm_output = quantised_flat.view(W, N, NC, C)

                    buf_comms[t] = comm_output.detach()

                elif comm_type == CommunicationType.AIM:
                    # * -- AIM Communication --

                    comm_log_probs = torch.zeros_like(action_log_probs)

                    codewords = torch.zeros((W, N, NC, actor.hq_levels, C), device=device)
                    quantised = torch.zeros((W, N, NC, actor.hq_levels), dtype=torch.long, device=device)

                    comm_conditions = lstm_output.squeeze(1)

                    for hq_level in range(actor.hq_levels):

                        comm_logits = actor.comm_heads[hq_level](comm_conditions).reshape(W, N, NC, actor.vocab_size)
                        # dist = torch.distributions.Categorical(logits=comm_logits)
                        # quantised_index = dist.sample()

                        # comm_log_probs += dist.log_prob(quantised_index).sum(dim=-1)

                        probs = F.softmax(comm_logits.float(), dim=-1)
    
                        quantised_index = torch.multinomial(probs.view(-1, actor.vocab_size), 1).view(W, N, NC)
                        
                        log_probs_all = F.log_softmax(comm_logits.float(), dim=-1)
                        level_log_probs = log_probs_all.gather(dim=-1, index=quantised_index.unsqueeze(-1)).squeeze(-1)
                        
                        comm_log_probs += level_log_probs.sum(dim=-1)

                        quantised[:, :, :, hq_level] = quantised_index

                        codewords[:, :, :, hq_level, :] = actor.aim_codebook[hq_level, quantised_index].view(W, N, NC, C)

                        comm_conditions = torch.cat([comm_conditions, codewords[:, :, :, hq_level, :].reshape(W, N, NC * C)], dim=-1)

                    comm_output = codewords.sum(dim=-2)
                    buf_comms[t] = quantised.detach()

                elif comm_type == CommunicationType.NONE:
                    comm_output = torch.zeros((W, N, NC, C), device=device) 
                    comm_log_probs = torch.zeros_like(action_log_probs)

                    buf_comms[t] = comm_output.detach()

                # Use chosen actions in env
                action_choice = actions.cpu().numpy()
                comm_choice = comm_output.detach().cpu().numpy()

                next_obs, next_global_obs, targets, rewards = sim.parallel_step(action_choice, comm_choice)

                # Save to current episodes
                buf_obs[t] = obs_tensor 
                buf_global_obs[t] = global_state_tensor
                # buf_targets[t] = torch.as_tensor(targets, dtype=torch.float32, device=device)
                # buf_rewards[t] = torch.as_tensor(rewards, dtype=torch.float32, device=device)
                buf_targets[t].copy_(torch.from_numpy(targets)) 
                buf_rewards[t].copy_(torch.from_numpy(rewards))
                buf_actions[t] = actions
                buf_action_log_probs[t] = action_log_probs.detach()
                buf_comm_log_probs[t] = comm_log_probs.detach()
                # buf_c_values[t] = values.detach().squeeze()

                curr_obs = next_obs
                curr_global_obs = next_global_obs # np.array(sim.get_all_global_obs(comm_choice))

                reward_means.append(np.mean(rewards))

                end = time()
                if end - start < config.timestep:
                    sleep(config.timestep - (end - start))

            global_state_tensor = torch.tensor(np.stack(curr_global_obs)).float().to(device)
            if global_state_tensor.ndim == 2:
                global_state_tensor = global_state_tensor.unsqueeze(1)
                        
            # with torch.no_grad():
            #     batch_last_vals = critic(global_state_tensor)
            
            # # Remove singleton time dimension from final values
            # if batch_last_vals.ndim == 3:
            #     batch_last_vals = batch_last_vals.squeeze(1)
            # final_batch_values.append(batch_last_vals)

            with torch.no_grad():
                flat_global_obs = buf_global_obs.view(-1, critic.input_dim)
                batched_values = critic(flat_global_obs).view(T, W, N)
                buf_c_values = batched_values.detach()

            global_state_tensor = torch.tensor(np.stack(curr_global_obs)).float().to(device)
            if global_state_tensor.ndim == 2:
                global_state_tensor = global_state_tensor.unsqueeze(1)
                        
            with torch.no_grad():
                batch_last_vals = critic(global_state_tensor)
            
            if batch_last_vals.ndim == 3:
                batch_last_vals = batch_last_vals.squeeze(1)
            final_batch_values.append(batch_last_vals)

            # Save to buffer
            batch_data = {
                "obs": buf_obs,
                "global_obs": buf_global_obs,
                "targets": buf_targets,
                "actions": buf_actions,
                "comm": buf_comms,
                "action_log_probs": buf_action_log_probs,
                "comm_log_probs": buf_comm_log_probs,
                "rewards": buf_rewards,
                "c_values": buf_c_values,
                "a_hidden_states": buf_hidden_states
            }
            buffer.add_episodes(batch_data)
            
        logger.update_step(current_training_timestep, batch_runs, batch_runs, "Updating models")

        buffer.compute_advantages(torch.cat(final_batch_values, dim=0), config.gamma, config.gae_lambda)

        buffer.advantages = (buffer.advantages - buffer.advantages.mean()) / (buffer.advantages.std() + 1e-8)

        sum_actor_loss = 0
        sum_critic_loss = 0
        sum_entropy_loss = 0
        sum_vq_loss = 0

        sum_returns_prediction_loss = 0
        sum_predicted_critic_value_loss = 0
        sum_predicted_intent_loss = 0

        sum_action_grad_norm = 0
        sum_lstm_grad_norm = 0
        sum_comm_grad_norm = 0

        communication_entropy = 0
        communication_perplexity = 0
        communication_active_codebook_usage = 0
        update_count = 0

        # scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')
        
        for _ in range(config.ppo_epochs):

            # Perhaps work out worlds_parallised = how large GPU mem is
            for batch in buffer.get_minibatches():
                
                # [B, T, N, Obs]
                obs = batch["obs"]
                B, _, _, _ = obs.shape

                # [B, T, N, Global Obs]
                global_obs = batch["global_obs"]
                # [B, T, N, Actions]
                actions = batch["actions"]

                # rewards = batch["rewards"]

                # [B, T, N, Comm]
                comms = batch["comm"] if comm_type != CommunicationType.NONE else None

                targets = batch["targets"].view(B, T, N, 3)

                # [B, T, N]
                old_action_log_probs = batch["action_log_probs"]
                old_comm_log_probs = batch["comm_log_probs"]

                advantages = batch["advantages"].flatten()
                returns = batch["returns"]

                h_c_init = batch["a_hidden_states"][:, 0] 
                # [1, B * N, Hidden]
                h_0 = h_c_init[:, :, 0, :].reshape(-1, config.lstm_hidden_size).unsqueeze(0)
                c_0 = h_c_init[:, :, 1, :].reshape(-1, config.lstm_hidden_size).unsqueeze(0)

                batch_a_hidden_init = (h_0.contiguous(), c_0.contiguous())
                
                # Calculate all new critic values
                flat_global_obs = global_obs.view(-1, *global_obs.shape[2:])
                new_values = critic(flat_global_obs).reshape(-1)

                if comm_type == CommunicationType.AIM:
                    embedded_comms = torch.zeros((B, T, N, NC, actor.hq_levels, C), device=device)
                    for hq_level in range(actor.hq_levels):
                        embedded_comms[:, :, :, :, hq_level, :] = actor.aim_codebook[hq_level, comms[..., hq_level].long()]
                    continuous_comms = embedded_comms.sum(dim=-2)
                else:
                    continuous_comms = comms

                shifted_comms = None
                if continuous_comms is not None:
                    shifted_comms = torch.zeros_like(continuous_comms)
                    shifted_comms[:, 1:] = continuous_comms[:, :-1]
                
                all_action_logits, comm_logits, lstm_output, _ = actor(obs, batch_a_hidden_init, shifted_comms)

                vq_loss = torch.tensor(0.0, device=device)

                # * -- Action Resampling --
                new_action_dist = torch.distributions.Categorical(logits=all_action_logits)
                new_action_log_probs = new_action_dist.log_prob(actions).flatten()
                action_entropy = new_action_dist.entropy().flatten()

                if comm_type == CommunicationType.CONTINUOUS:
                    # * -- Continuous Communication Resampling --
                    new_comm_std = actor.comm_log_std.exp().expand_as(comm_logits)
                    new_comm_dist = torch.distributions.Normal(comm_logits, new_comm_std)
                    new_comm_log_probs = new_comm_dist.log_prob(comms).sum(dim=[-2, -1]).flatten()
                    comm_entropy = new_comm_dist.entropy().sum(dim=[-2, -1]).flatten()

                elif comm_type == CommunicationType.DISCRETE:
                    # * -- Discrete Communication --
                    flat_inputs = comm_logits.view(-1, C)
                    quantised_flat, vq_loss, quantised_indices = actor.vq_layer(flat_inputs)

                    new_comm_log_probs = torch.zeros_like(new_action_log_probs)
                    comm_entropy = torch.zeros_like(action_entropy)

                    batch_entropy, batch_perplexity = actor.vq_layer.compute_metrics(quantised_indices)
                    communication_entropy += batch_entropy.item()
                    communication_perplexity += batch_perplexity.item()

                    communication_active_codebook_usage += Quantiser.get_active_codebook_usage(quantised_indices)
                
                elif comm_type == CommunicationType.AIM:
                    # * -- AIM Communication --

                    new_comm_log_probs = torch.zeros_like(new_action_log_probs)
                    comm_entropy = torch.zeros_like(action_entropy)

                    comm_conditions = lstm_output

                    codewords = torch.zeros((B, T, N, NC, actor.hq_levels, C), device=device)
                    soft_codewords = torch.zeros_like(codewords) # For STE

                    # Note: Teacher Forcing - no dist.sample - similar to chatgpt training
                    for hq_level in range(actor.hq_levels):
                        comm_logits = actor.comm_heads[hq_level](comm_conditions).reshape(B, T, N, NC, actor.vocab_size)
                        
                        dist = torch.distributions.Categorical(logits=comm_logits)
                        quantised_index = comms[..., hq_level].long()
                        
                        new_comm_log_probs += dist.log_prob(quantised_index).sum(dim=-1).flatten()
                        comm_entropy += dist.entropy().sum(dim=-1).flatten()

                        codewords[:, :, :, :, hq_level, :] = actor.aim_codebook[hq_level, quantised_index]
                        soft_codewords[:, :, :, :, hq_level, :] = dist.probs @ actor.aim_codebook[hq_level]

                        comm_conditions = torch.cat([comm_conditions, codewords[:, :, :, :, hq_level, :].reshape(B, T, N, NC * C)], dim=-1)

                    codewords = codewords.detach() + soft_codewords - soft_codewords.detach()
                    comm_output = codewords.sum(dim=-2).view(-1, NC * C)
                    
                    sender_context = torch.cat([comm_output, lstm_output.reshape(-1, config.lstm_hidden_size)], dim=-1)

                    # Predicted return values
                    predicted_agents_returns = actor.predict_agents_returns(sender_context)
                    predicted_returns_loss = F.mse_loss(predicted_agents_returns, returns.view(-1, N).repeat_interleave(N, dim=0))

                    # Predicted critic values
                    predicted_critic_values = actor.predict_critic_value(sender_context)
                    predicted_critic_loss = F.mse_loss(predicted_critic_values, new_values.detach().view(-1, 1))



                    # Intent

                    true_targets = targets[..., :2]
                    true_target_exists = targets[..., 2]

                    predicted_sender_goal = actor.predict_intent_sender_goal(sender_context)

                    flat_true_targets_exists = true_target_exists.view(-1)

                    predicted_sender_goal_loss = F.mse_loss(predicted_sender_goal, true_targets.view(-1, 2), reduction='none').mean(dim=-1)
                    predicted_sender_goal_loss = (predicted_sender_goal_loss * flat_true_targets_exists).sum() / (flat_true_targets_exists.sum() + 1e-8)

                    mask = ~torch.eye(N, dtype=torch.bool, device=actions.device)
                    expanded_comms = continuous_comms.view(B, T, N, NC * C).unsqueeze(2).expand(B, T, N, N, NC * C)
                    other_comms = expanded_comms[..., mask, :].reshape(B, T, N, N - 1, NC * C).detach().reshape(-1, NC * C)

                    lstm_expanded = lstm_output.unsqueeze(2).expand(B, T, N, N, config.lstm_hidden_size)
                    other_lstm = lstm_expanded[..., mask, :].reshape(B, T, N, N - 1, config.lstm_hidden_size).reshape(-1, config.lstm_hidden_size)

                    receiver_context = torch.cat([other_comms, other_lstm], dim=-1)

                    predicted_reciever_goal = actor.predict_intent_receiver_goal(receiver_context)

                    expanded_goals = true_targets.unsqueeze(2).expand(B, T, N, N, 2)
                    other_goals = expanded_goals[..., mask, :].reshape(-1, 2)

                    expanded_exists = true_target_exists.unsqueeze(2).expand(B, T, N, N)
                    other_exists = expanded_exists[..., mask].reshape(-1)

                    # predicted_reciever_goal_loss = F.mse_loss(predicted_reciever_goal, other_goals)

                    predicted_reciever_goal_loss = F.mse_loss(predicted_reciever_goal, other_goals, reduction='none').mean(dim=-1)
                    predicted_reciever_goal_loss = (predicted_reciever_goal_loss * other_exists).sum() / (other_exists.sum() + 1e-8)


                    # Combine both intent losses
                    intent_loss = predicted_sender_goal_loss# + predicted_reciever_goal_loss

                    batch_entropy, batch_perplexity = AIM.compute_metrics(comms, config.vocab_size)
                    communication_entropy += batch_entropy
                    communication_perplexity += batch_perplexity
                    communication_active_codebook_usage += len(torch.unique(comms.flatten()))


                elif comm_type == CommunicationType.NONE:
                    # * -- No Communication --
                    new_comm_log_probs = torch.zeros_like(new_action_log_probs)
                    comm_entropy = torch.zeros_like(action_entropy)


                entropy = action_entropy + comm_entropy
                entropy_loss = -action_entropy.mean() + -comm_entropy.mean() * 0.1
                                
                critic_loss = F.mse_loss(new_values, returns.flatten())

                log_ratio = new_action_log_probs - old_action_log_probs.flatten()
                ratio = log_ratio.exp()
                action_loss = -torch.min(
                    advantages * ratio,
                    advantages * torch.clamp(ratio, 1.0 - config.clip_coef, 1.0 + config.clip_coef)
                ).mean()

                log_ratio = new_comm_log_probs - old_comm_log_probs.flatten()
                ratio = log_ratio.exp()
                comm_loss = -torch.min(
                    advantages * ratio,
                    advantages * torch.clamp(ratio.detach(), 1.0 - config.clip_coef, 1.0 + config.clip_coef)
                ).mean()

                actor_loss = action_loss + comm_loss# * 0.2

                progress = current_training_timestep / config.training_timesteps

                if config.communication_type == CommunicationType.AIM:

                    nudge_coef = 0.001 # max(0.0005, 0.1 * (1.0 - progress)) # dynamic to allow for initial exploration

                    actor_loss += nudge_coef * (predicted_returns_loss + predicted_critic_loss + intent_loss)

                    sum_returns_prediction_loss += predicted_returns_loss.item()
                    sum_predicted_critic_value_loss += predicted_critic_loss.item()
                    sum_predicted_intent_loss += intent_loss.item()

                
                ent_coef = config.ent_coef # max(0.0001, config.ent_coef * (1.0 - progress)) # dynamic to allow for initial exploration
                
                total_loss = actor_loss + config.vf_coef * critic_loss + ent_coef * entropy_loss
                total_loss += vq_loss

                sum_actor_loss += actor_loss.item()
                sum_critic_loss += critic_loss.item()
                sum_entropy_loss += entropy_loss.item()
                sum_vq_loss += vq_loss.item()

                update_count += 1
                
                # Update
                actor_optimizer.zero_grad()
                critic_optimizer.zero_grad()
                
                total_loss.backward()

                lstm_params = [p for n, p in actor.named_parameters() if 'lstm' in n.lower()]
                comm_params = [p for n, p in actor.named_parameters() if 'comm_head' in n.lower()]
                action_params = [p for n, p in actor.named_parameters() if 'action' in n.lower() or 'actor_head' in n.lower()]
                
                sum_lstm_grad_norm += clip_grad_norm_(lstm_params, float('inf')).detach()
                sum_comm_grad_norm += clip_grad_norm_(comm_params, float('inf')).detach()
                sum_action_grad_norm += clip_grad_norm_(action_params, float('inf')).detach()
                
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                
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
            communication_active_codebook_usage=communication_active_codebook_usage / update_count,

            actor_loss=sum_actor_loss / update_count,
            critic_loss=sum_critic_loss / update_count,
            entropy_loss=sum_entropy_loss / update_count,
            vq_loss=sum_vq_loss / update_count,

            predicted_return_loss=sum_returns_prediction_loss / update_count,
            predicted_critic_value_loss=sum_predicted_critic_value_loss / update_count,
            predicted_intent_loss=sum_predicted_intent_loss / update_count,

            lstm_grad_norm=sum_lstm_grad_norm.item() / update_count,
            comm_grad_norm=sum_comm_grad_norm.item() / update_count,
            action_grad_norm=sum_action_grad_norm.item() / update_count,
            
        )

    logger.close()
