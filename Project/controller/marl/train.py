from time import sleep, time
import torch.nn.functional as F
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from .buffer import RolloutBuffer
from .config import CommunicationType, Config, GenerativeLangType
from .logger import Logging, MetricTracker, Visualiser

def train(system, config: Config, device: torch.device):

    sim = system['sim']
    actor = system['actor']
    critic = system['critic']
    actor_optimizer = system['actor_opt']
    critic_optimizer = system['critic_opt']

    cm = system['checkpoint_manager']

    T = config.training.simulation_timesteps
    W = config.training.worlds_parallised
    N = system['num_agents']
    C = config.comms.communication_size
    NC = config.comms.num_comms

    A = system['act_shape'][0]

    comm_type = config.comms.communication_type

    rollout_phases = config.training.rollout_phases

    num_codebooks = 0
    if comm_type == CommunicationType.AIM:
        num_codebooks = config.comms.rq_levels * len(actor.comm_protocol.encoder.get_codebooks())
    elif comm_type == CommunicationType.DISCRETE:
        num_codebooks = config.comms.rq_levels

    buffer = RolloutBuffer(
        episodes=rollout_phases,
        timesteps_per_run=T,
        num_worlds_per_run=W,
        num_agents=N,
        num_obs=system['agent_obs_shape'][0],
        num_comms=NC,
        comm_dim=C,
        num_global_obs=sim.get_global_obs_size(0) + N * NC * C, 
        hidden_state_size=config.actor.lstm_hidden_size,
        num_codebooks=num_codebooks,
        device=device,
    )

    logger = Logging(save_folder=cm.get_result_path(), config=config, start_step=system["start_step"], num_codebooks=num_codebooks)
    metric_tracker = system["metric_tracker"]
    visualiser = Visualiser(config)

    lstm_params = set(p for n, p in actor.named_parameters() if 'lstm' in n.lower())
    action_params = set(p for n, p in actor.named_parameters() if 'action' in n.lower() or 'actor_head' in n.lower())
    comm_params = set(p for n, p in actor.named_parameters() if 'comm' in n.lower() or 'vq' in n.lower())

    actor_optimizer.param_groups[0]['params'] = list(lstm_params) + list(action_params)
    actor_optimizer.param_groups[0]['lr'] = 5e-5

    new_comm_group = {'params': list(comm_params), 'lr': 1e-4}

    for key in actor_optimizer.param_groups[0]:
        if key not in ['params', 'lr']:
            new_comm_group[key] = actor_optimizer.param_groups[0][key]

    actor_optimizer.param_groups.append(new_comm_group)

    # actor = torch.jit.script(actor)
    for current_training_timestep in range(system["start_step"], config.training.training_timesteps):

        reward_means = []

        final_batch_values = []
        buffer.reset()

        for batch_run in range(rollout_phases):

            visualiser.update_step(current_training_timestep, batch_run, rollout_phases, "Collecting Data", metric_tracker, checkpoint=batch_run == 0 and current_training_timestep != 0)

            buf_obs = torch.zeros((T, W, N, system['agent_obs_shape'][0]), dtype=torch.float32, device=device)
            buf_global_obs = torch.zeros((T, W, sim.get_global_obs_size(0) + N * NC * C), dtype=torch.float32, device=device)
            buf_targets = torch.zeros((T, W, N, 3), dtype=torch.float32, device=device)
            buf_actions = torch.zeros((T, W, N), dtype=torch.float32, device=device)
            buf_rewards = torch.zeros((T, W, N), dtype=torch.float32, device=device)
            buf_action_log_probs = torch.zeros((T, W, N), dtype=torch.float32, device=device)
            buf_comm_log_probs = torch.zeros((T, W, N), dtype=torch.float32, device=device)
            buf_c_values = torch.zeros((T, W, N), dtype=torch.float32, device=device)
            buf_hidden_states = torch.zeros((T, W, N, 2, config.actor.lstm_hidden_size), dtype=torch.float32, device=device)

            if num_codebooks == 0:
                buf_comms = torch.zeros((T, W, N, NC, C), dtype=torch.float32, device=device)
            else:
                buf_comms = torch.zeros((T, W, N, NC, num_codebooks), dtype=torch.float32, device=device)

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
                    action_logits, lstm_output, actor_hidden_states = actor(obs_tensor, actor_hidden_states, comm_choice_tensor)
                    # values = critic(global_state_tensor)

                # * -- Action Sampling --
                action_probs = F.softmax(action_logits.float(), dim=-1)
                action_log_probs_all = F.log_softmax(action_logits.float(), dim=-1)
                actions = torch.multinomial(action_probs.view(-1, A), 1).view(W, N)
                action_log_probs = action_log_probs_all.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)

                comm_output, to_save, comm_log_probs = actor.comm_protocol.get_comms_during_rollout(lstm_output)
                buf_comms[t] = to_save

                # Use chosen actions in env
                # action_choice = actions.cpu().numpy()
                # comm_choice = comm_output.detach().cpu().numpy()

                # next_obs, next_global_obs, targets, rewards = sim.parallel_step(action_choice, comm_choice)

                action_choice = actions.cpu().numpy().astype(np.int32)
                comm_choice = comm_output.detach().cpu().numpy().astype(np.float32)

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
                if end - start < config.training.timestep:
                    sleep(config.training.timestep - (end - start))

            global_state_tensor = torch.tensor(np.stack(curr_global_obs)).float().to(device)
            if global_state_tensor.ndim == 2:
                global_state_tensor = global_state_tensor.unsqueeze(1)

            with torch.no_grad():
                flat_global_obs = buf_global_obs.view(-1, critic.input_dim)
                batched_values = critic(flat_global_obs).view(T, W, N)
                buf_c_values = batched_values.detach()
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
            
        visualiser.update_step(current_training_timestep, rollout_phases, rollout_phases, "Updating models", metric_tracker, checkpoint=False)
        metric_tracker.reset_tracker()

        buffer.compute_advantages(torch.cat(final_batch_values, dim=0), config.mappo.gamma, config.mappo.gae_lambda)

        buffer.advantages = (buffer.advantages - buffer.advantages.mean()) / (buffer.advantages.std() + 1e-8)

        # Calculate all new critic values at once
        # all_global_obs = buffer.global_obs.view(-1, buffer.global_obs.shape[-1])
        # with torch.no_grad():
        #     all_values = critic(all_global_obs)
        # buffer.c_values = all_values.view_as(buffer.c_values)

        # scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')
        
        for _ in range(config.mappo.ppo_epochs):

            # Perhaps work out worlds_parallised = how large GPU mem is
            for batch in buffer.get_minibatches(batch_size=config.training.batch_size):
                
                # [B, T, N, Obs]
                obs = batch["obs"]
                B, _, _, _ = obs.shape

                # [B, T, N, Global Obs]
                global_obs = batch["global_obs"]
                # [B, T, N, Actions]
                actions = batch["actions"]

                # rewards = batch["rewards"]

                # [B, T, N, Comm]
                comms = batch["comm"]

                targets = batch["targets"].view(B, T, N, 3)

                # [B, T, N]
                old_action_log_probs = batch["action_log_probs"]
                old_comm_log_probs = batch["comm_log_probs"]

                advantages = batch["advantages"]
                returns = batch["returns"]

                h_c_init = batch["a_hidden_states"][:, 0] 
                # [1, B * N, Hidden]
                h_0 = h_c_init[:, :, 0, :].reshape(-1, config.actor.lstm_hidden_size).unsqueeze(0)
                c_0 = h_c_init[:, :, 1, :].reshape(-1, config.actor.lstm_hidden_size).unsqueeze(0)

                batch_a_hidden_init = (h_0.contiguous(), c_0.contiguous())
                
                # Get all new critic values
                # new_values = batch["c_values"].reshape(-1)
                flat_global_obs = global_obs.view(-1, *global_obs.shape[2:])
                new_values = critic(flat_global_obs).reshape(-1)


                # if comm_type == CommunicationType.AIM:
                #     embedded_comms = torch.zeros((B, T, N, NC, config.comms.rq_levels, C), device=device)
                #     for hq_level in range(config.comms.rq_levels):
                #         embedded_comms[:, :, :, :, hq_level, :] = actor.comm_protocol.aim_codebook[hq_level, comms[..., hq_level].long()]
                #     continuous_comms = embedded_comms.sum(dim=-2)
                # else:
                #     continuous_comms = comms

                if comm_type == CommunicationType.AIM:
                    level_codewords = []
                    
                    for hq_level in range(num_codebooks):
                        cb = actor.comm_protocol.aim_codebooks[hq_level]
                        current_codewords = cb[comms[..., hq_level].long()]
                        level_codewords.append(current_codewords)
                    
                    if config.comms.autoencoder_type == GenerativeLangType.HQ_VAE:
                        continuous_comms = torch.cat(level_codewords, dim=-1)
                    else:
                        continuous_comms = torch.stack(level_codewords, dim=-2).sum(dim=-2)
                else:
                    continuous_comms = comms

                shifted_comms = None
                if continuous_comms is not None:
                    shifted_comms = torch.zeros_like(continuous_comms)
                    shifted_comms[:, 1:] = continuous_comms[:, :-1]

                all_action_logits, lstm_output, _ = actor(obs, batch_a_hidden_init, shifted_comms)


                # * -- Action Resampling --
                new_action_dist = torch.distributions.Categorical(logits=all_action_logits)
                new_action_log_probs = new_action_dist.log_prob(actions).flatten()
                action_entropy = new_action_dist.entropy().flatten()

                comm_output, new_comm_log_probs, comm_entropy = actor.comm_protocol.get_comms_during_update(lstm_output, comms)
                comm_protocol_loss = actor.comm_protocol.get_loss(
                    lstm_output, comm_output, continuous_comms,
                    returns, new_values, targets
                )

                metric_tracker.update("comm_perplexity", torch.exp(-new_comm_log_probs.mean()).item())
                metric_tracker.update("comm_entropy", comm_entropy.mean().item())

                entropy_loss = -action_entropy.mean() + -comm_entropy.mean()
                                
                critic_loss = F.mse_loss(new_values, returns.flatten())

                log_ratio = new_action_log_probs - old_action_log_probs
                ratio = log_ratio.exp()
                action_loss = -torch.min(
                    advantages * ratio,
                    advantages * torch.clamp(ratio, 1.0 - config.mappo.clip_coef, 1.0 + config.mappo.clip_coef)
                ).mean()

                log_ratio = new_comm_log_probs.flatten() - old_comm_log_probs
                ratio = log_ratio.exp()
                comm_loss = -torch.min(
                    advantages * ratio,
                    advantages * torch.clamp(ratio, 1.0 - config.mappo.clip_coef, 1.0 + config.mappo.clip_coef)
                ).mean()

                actor_loss = action_loss + comm_loss# * 0.2

                actor_loss += comm_protocol_loss
                
                ent_coef = config.mappo.ent_coef # max(0.0001, config.ent_coef * (1.0 - progress)) # dynamic to allow for initial exploration
                
                total_loss = actor_loss + config.mappo.vf_coef * critic_loss + ent_coef * entropy_loss

                metric_tracker.update("action_loss", action_loss.item())
                metric_tracker.update("comm_loss", comm_loss.item())
                metric_tracker.update("actor_loss", actor_loss.item())
                metric_tracker.update("critic_loss", critic_loss.item())
                metric_tracker.update("entropy_loss", entropy_loss.item())

                # Update
                actor_optimizer.zero_grad()
                critic_optimizer.zero_grad()
                
                total_loss.backward()

                metric_tracker.update("lstm_grad_norm", clip_grad_norm_(lstm_params, float('inf')).item())
                metric_tracker.update("comm_grad_norm", clip_grad_norm_(comm_params, float('inf')).item())
                metric_tracker.update("action_grad_norm", clip_grad_norm_(action_params, float('inf')).item())
                
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                
                actor_optimizer.step()
                critic_optimizer.step()

        if current_training_timestep % config.training.periodic_save_interval == 0:
            cm.save_checkpoint(actor, critic, actor_optimizer, critic_optimizer, current_training_timestep)
        
        metric_tracker.update("reward_mean", np.mean(reward_means))
        metric_tracker.update("reward_std", np.std(reward_means))

        logger.log(current_training_timestep, metric_tracker)

    logger.close()
    visualiser.stop()
