from time import sleep, time
import torch.nn.functional as F
import numpy as np
import torch
from torch.utils.data import DataLoader
import os

from .run_sim import run_sim

from .datasets import NudgeFeature, ObsData
from .buffer import RolloutBuffer
from .config import CommunicationType
from .models import AIM, Quantiser
from .logger import Logger


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

    batch_runs = config.buffer_size // W

    buffer = RolloutBuffer(
        total_runs=config.buffer_size,
        timesteps_per_run=T,
        num_worlds_per_run=W,
        num_agents=N,
        num_obs=system['agent_obs_shape'][0] + system["input_comms_shape"][0],
        num_comms=C,
        num_global_obs=sim.get_global_obs_size(0) + C * N, 
        hidden_state_size=config.lstm_hidden_size,
        device=device
    )

    logger = Logger(save_folder=cm.get_result_path(), config=config, start_step=system["start_step"])

    for current_training_timestep in range(system["start_step"], config.training_timesteps):

        reward_means = []

        final_batch_values = []
        buffer.reset()

        for batch_run in range(batch_runs):

            logger.update_step(current_training_timestep, batch_run, batch_runs, "Collecting Data")

            buf_obs = np.zeros((T, W, N, system['agent_obs_shape'][0] + system["input_comms_shape"][0]), dtype=np.float32)
            buf_global_obs = np.zeros((T, W, sim.get_global_obs_size(0) + C * N), dtype=np.float32)
            buf_actions = np.zeros((T, W, N), dtype=np.float32)
            buf_rewards = np.zeros((T, W, N), dtype=np.float32)
            buf_log_probs = np.zeros((T, W, N), dtype=np.float32)
            buf_c_values = np.zeros((T, W, N), dtype=np.float32)
            buf_comms = np.zeros((T, W, N, C), dtype=np.float32)
            buf_hidden_states = np.zeros((T, W, N, 2, config.lstm_hidden_size), dtype=np.float32)     

            for world_id in range(W):
                sim.reset(world_id)

            comm_choice = np.zeros((W, N, C), dtype=np.float32)
            curr_obs = np.array([sim.get_agents_obs(w, comm_choice[w]) for w in range(W)])
            curr_global_obs = np.array(sim.get_all_global_obs(comm_choice))            

            # Reset hidden states for the recurrent model
            actor_hidden_states = actor.init_hidden(batch_size=W)
            
            for t in range(T):

                start = time()

                h = actor_hidden_states[0].reshape(W, N, -1)
                c = actor_hidden_states[1].reshape(W, N, -1)
                hidden_stacked = torch.stack([h, c], dim=2)
                buf_hidden_states[t] = hidden_stacked.detach().cpu().numpy()

                # Get action logits from the model and update memory and add time dimension for actor model
                obs_tensor = torch.from_numpy(curr_obs).float().to(device)
                global_state_tensor = torch.from_numpy(curr_global_obs).float().to(device)
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
                elif actor.get_comm_type() == CommunicationType.AIM:
                    # * -- AIM Communication --
                    comms = comm_mean
                    comm_log_probs = torch.zeros_like(action_log_probs)
                    
                log_probs = action_log_probs + comm_log_probs

                # Use chosen actions in env
                action_choice = actions.cpu().numpy()
                comm_choice = comms.detach().cpu().numpy()

                next_obs, rewards = sim.parallel_step(action_choice, comm_choice)
                                
                # Save to current episodes
                buf_obs[t] = curr_obs
                buf_global_obs[t] = curr_global_obs
                buf_actions[t] = action_choice
                buf_comms[t] = comm_choice
                buf_log_probs[t] = log_probs.detach().cpu().numpy()
                buf_rewards[t] = rewards
                buf_c_values[t] = values.detach().cpu().numpy().squeeze()

                curr_obs = next_obs
                curr_global_obs = np.array(sim.get_all_global_obs(comm_choice))

                reward_means.append(np.mean(rewards))

                end = time()
                if end - start < config.timestep:
                    sleep(config.timestep - (end - start))

            global_state_tensor = torch.tensor(np.stack(curr_global_obs)).float().to(device)
            if global_state_tensor.ndim == 2:
                global_state_tensor = global_state_tensor.unsqueeze(1)
                        
            with torch.no_grad():
                batch_last_vals = critic(global_state_tensor)
            
            # Remove singleton time dimension from final values
            if batch_last_vals.ndim == 3:
                batch_last_vals = batch_last_vals.squeeze(1)
            final_batch_values.append(batch_last_vals)

            # Save to buffer
            batch_data = {
                "obs": buf_obs,
                "global_obs": buf_global_obs,
                "actions": buf_actions,
                "comm": buf_comms,
                "log_probs": buf_log_probs,
                "rewards": buf_rewards,
                "c_values": buf_c_values,
                "a_hidden_states": buf_hidden_states
            }
            buffer.add_episodes(batch_data)
            
        logger.update_step(current_training_timestep, batch_runs, batch_runs, "Updating models")

        buffer.compute_advantages(torch.cat(final_batch_values, dim=0), config.gamma, config.gae_lambda)

        sum_actor_loss = 0
        sum_critic_loss = 0
        sum_entropy_loss = 0
        sum_vq_loss = 0
        sum_codebook_loss = 0
        communication_entropy = 0
        communication_perplexity = 0
        communication_active_codebook_usage = 0
        update_count = 0
        
        for _ in range(config.ppo_epochs):

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
                h_0 = h_c_init[:, :, 0, :].reshape(-1, config.lstm_hidden_size).unsqueeze(0)
                c_0 = h_c_init[:, :, 1, :].reshape(-1, config.lstm_hidden_size).unsqueeze(0)

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
                
                elif actor.get_comm_type() == CommunicationType.AIM:
                    # * -- AIM Communication --
                    new_comm_log_probs = torch.zeros_like(new_action_log_probs)
                    comm_entropy = torch.zeros_like(action_entropy)

                    batch_entropy, batch_perplexity = AIM.compute_metrics(quantised_indices, config.vocab_size)
                    communication_entropy += batch_entropy.item()
                    communication_perplexity += batch_perplexity.item()

                entropy = action_entropy + comm_entropy * 5
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
                loss2 = advantages * torch.clamp(ratio, 1.0 - config.clip_coef, 1.0 + config.clip_coef)
                actor_loss = -torch.min(loss1, loss2).mean()
                
                entropy_loss = -entropy.mean()
                
                total_loss = actor_loss + config.vf_coef * critic_loss + config.ent_coef * entropy_loss
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


def train_language(system, config, device):

    W = config.worlds_parallised
    N = system['num_agents']
    T = config.simulation_timesteps
    OBS_DIM = system['agent_obs_shape'][0]
    COMM_DIM = config.communication_size
    VOCAB_SIZE = config.vocab_size
    ACTION_COUNT = system['act_shape'][0]

    cm = system["checkpoint_manager"]

    # TODO: Make noisy actions to cover space better

    # Load obs logs
    obs_logs_file = cm.get_result_path() + "/obs_log.csv"
    if (
        (os.path.exists(obs_logs_file) and input(f"Obs logs already exist, would you like to overwrite? (y/N) ").lower() == "y")
        or not os.path.exists(obs_logs_file)
    ):

        RUNS = 10

        rewards = run_sim(system, config, device, RUNS, collect_obs_file=obs_logs_file)

        print(f"Obs collected: {OBS_DIM * RUNS * W * N * T}")
        print(f"Baseline: {np.mean(rewards)}")

    nudge_feature = NudgeFeature.Critic
    nudge_dim = None
    if nudge_feature == NudgeFeature.Critic:
        nudge_dim = 1
    elif nudge_feature == NudgeFeature.Action:
        nudge_dim = ACTION_COUNT

    dataset = ObsData(obs_logs_file, ACTION_COUNT, nudge_feature=nudge_feature)
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [0.8, 0.1, 0.1])

    training_loader, validation_loader, test_loader = (
        DataLoader(train_set, batch_size=32, shuffle=True),
        DataLoader(val_set, batch_size=32, shuffle=False),
        DataLoader(test_set, batch_size=32, shuffle=False)
    )

    # train vq-vae
    aim = AIM(OBS_DIM, 512, COMM_DIM, VOCAB_SIZE, nudge_dim=nudge_dim).to(device)
    optimizer = torch.optim.Adam(aim.parameters(), lr=0.00005)

    EPOCHS = 12
    for epoch in range(EPOCHS):

        train_loss = 0.0
        aim.train()

        aim.nudge = 0
        aim.recon = 0

        for batch_obs, batch_nudge in training_loader:

            batch_obs = batch_obs.to(device)
            batch_nudge = batch_nudge.to(device)

            optimizer.zero_grad()
            loss = aim(batch_obs, batch_nudge)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(aim.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

        aim.nudge /= len(training_loader)
        aim.recon /= len(training_loader)
        avg_train_loss = train_loss / len(training_loader)

        print(f"Nudge Loss {aim.nudge}")
        print(f"Reocon Loss {aim.recon}")

        aim.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_obs, batch_nudge in validation_loader:

                batch_obs = batch_obs.to(device)
                batch_nudge = batch_nudge.to(device)

                loss = aim(batch_obs, batch_nudge) 
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(validation_loader)

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    aim.eval()
    test_loss = 0.0

    with torch.no_grad():
        for batch_obs, batch_nudge in test_loader:

            batch_obs = batch_obs.to(device)
            batch_nudge = batch_nudge.to(device)

            loss = aim(batch_obs, batch_nudge)
            test_loss += loss.item()

    print(f"Final Test Set Loss: {test_loss / len(test_loader):.4f}")

    # save language
    aim.save_codebook(cm.get_result_path())