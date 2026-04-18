


from datetime import datetime
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F


from controller.marl.core.config import CommunicationType, Config
from controller.marl.core.datasets import ObsData
from controller.marl.runners.sim_runner import run_sim
from controller.marl.core.logger import Logging
from controller.marl.core.metric_tracker import MetricTracker
from project_paths import RESULTS_DIR

def get_save_folder(datetime_str: str, config: Config):
    parent_folder = RESULTS_DIR / config.comms.communication_type.value / "base"
    os.makedirs(parent_folder, exist_ok=True)
    folder = parent_folder / datetime_str
    os.makedirs(folder, exist_ok=True)
    return folder


def save(folder, actor, critic, actor_optimizer, critic_optimizer, config: Config):
    state = {
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "actor_optimizer": actor_optimizer.state_dict(),
        "critic_optimizer": critic_optimizer.state_dict(),
        "step": config.aim_training.obs_runs
    }

    torch.save(state, folder / "models.pth")


def spatial_index(targets: torch.tensor, max_index: int):
    # x_bins * y_bins <= vocab_size
    x_bins = int(np.floor(np.sqrt(max_index)))
    y_bins = x_bins

    x = torch.clamp((targets[:, 0] * x_bins).long(), min=0, max=x_bins - 1)
    y = torch.clamp((targets[:, 1] * y_bins).long(), min=0, max=y_bins - 1)

    return y * x_bins + x

def imitation_learning(system, config: Config, device: torch.device):

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

    O = system['agent_obs_shape'][0]
    GO = sim.get_global_obs_dim()# + N * NC * C
    A = system['act_shape'][0]

    obs_external_mask = torch.tensor(sim.get_agent_external_obs_mask(0), dtype=torch.bool, device=device)

    obs_logs_file = RESULTS_DIR / "obs_log.csv"

    if (
        (os.path.exists(obs_logs_file) and input(f"Obs logs already exist, would you like to overwrite? (y/N) ").lower() == "y")
        or not os.path.exists(obs_logs_file)
    ):

        RUNS = config.imitation.obs_runs

        rewards = run_sim(system, config, device, RUNS, collect_obs_file=obs_logs_file, optimal=True)

        print(f"Obs collected: {RUNS * W * N * T}")
        print(f"Baseline: {np.mean(rewards)}")


    # dataset = ObsData(obs_logs_file, O, GO, obs_external_mask, device)
    dataset = ObsData(obs_logs_file, O, GO, device, T, W, N)
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [0.8, 0.1, 0.1])

    print(f"Using {len(train_set)} datapoints for training...")

    training_loader, validation_loader, test_loader = (
        DataLoader(train_set, batch_size=config.aim_training.aim_batch_size, shuffle=False),
        DataLoader(val_set, batch_size=config.aim_training.aim_batch_size, shuffle=False),
        DataLoader(test_set, batch_size=config.aim_training.aim_batch_size, shuffle=False)
    )
    
    save_folder = get_save_folder(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), config)

    num_codebooks = 0
    if config.comms.communication_type == CommunicationType.AIM:
        num_codebooks = config.comms.rq_levels * len(actor.comm_protocol.encoder.get_codebooks())
    elif config.comms.communication_type == CommunicationType.DISCRETE:
        num_codebooks = config.comms.rq_levels

    logger = Logging(str(save_folder), config, 0, num_codebooks, "imitate")
    tracker = MetricTracker()

    for epoch in range(config.imitation.epochs):

        actor.train()
        critic.train()

        for batch_obs, batch_global_obs, batch_actions, batch_targets, batch_critic_values, batch_return_values in training_loader:
    
            B, T, N, _ = batch_obs.shape

            actor_hidden_states = actor.init_hidden(batch_size=B)
            world_comms = torch.zeros((B, T, N, NC * C), device=device)

            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()

            # for t in range(T):
                
            # obs_t = batch_obs[:, t]
            # actions_t = batch_actions[:, t]
            # global_obs_t = batch_global_obs[:, t]
            # returns_t = batch_return_values[:, t]

            action_logits, lstm_output, actor_hidden_states = actor(batch_obs, actor_hidden_states, world_comms)
            
            # comm_output, _, _ = actor.comm_protocol.get_comms_during_rollout(lstm_output)
            # world_comms = comm_output.detach()

            flat_targets = batch_targets.view(-1, batch_targets.shape[-1])
            target_indices = spatial_index(flat_targets, config.comms.vocab_size)
            embedded_comms = actor.comm_protocol.aim_codebooks[0][target_indices.long()].view(B, T, N, C)

            # total_comms = embedded_comms.sum(dim=2, keepdim=True)
            # sent_comms = (total_comms - embedded_comms) / (N - 1)

            # dropout_mask = (torch.rand(B, T, N, 1, device=device) > 0.5).float()
            # sent_comms = sent_comms * dropout_mask

            world_comms = torch.zeros((B, T, N, NC, C), device=device)
            world_comms[:, :, :, 0, :] = embedded_comms 
            world_comms = world_comms.view(B, T, N, NC * C)

            actor_loss = F.cross_entropy(action_logits.reshape(-1, A), batch_actions.reshape(-1))

            flat_global_obs = batch_global_obs[:, :, 0, :].reshape(B * T, -1)
            values = critic(flat_global_obs)
            critic_loss = F.mse_loss(values.reshape(-1), batch_return_values.reshape(-1))

            tracker.update("train_actor_loss", actor_loss.item() / T)
            tracker.update("train_critic_loss", critic_loss.item() / T)
            
            loss = (actor_loss + critic_loss) / T

            loss.backward()

            # h, c = actor_hidden_states
            # actor_hidden_states = (h.detach(), c.detach())

            predicted_actions = torch.argmax(action_logits.reshape(-1, A), dim=-1)
            correct_predictions = (predicted_actions == batch_actions.reshape(-1)).sum().item()
            total_predictions = batch_actions.numel()

            tracker.update("train_accuracy", correct_predictions / total_predictions)

            actor_optimizer.step()
            critic_optimizer.step()
                    
        with torch.no_grad():
            for batch_obs, batch_global_obs, batch_actions, batch_targets, batch_critic_values, batch_return_values in validation_loader:
        
                B, T, N, _ = batch_obs.shape

                actor_hidden_states = actor.init_hidden(batch_size=B)
                world_comms = torch.zeros((B, T, N, NC * C), device=device)

                actor_optimizer.zero_grad()
                critic_optimizer.zero_grad()

                # for t in range(T):
                    
                # obs_t = batch_obs[:, t]
                # actions_t = batch_actions[:, t]
                # global_obs_t = batch_global_obs[:, t]
                # returns_t = batch_return_values[:, t]

                action_logits, lstm_output, actor_hidden_states = actor(batch_obs, actor_hidden_states, world_comms)
                
                # comm_output, _, _ = actor.comm_protocol.get_comms_during_rollout(lstm_output)
                # world_comms = comm_output.detach()

                flat_targets = batch_targets.view(-1, batch_targets.shape[-1])
                target_indices = spatial_index(flat_targets, config.comms.vocab_size)
                embedded_comms = actor.comm_protocol.aim_codebooks[0][target_indices.long()].view(B, T, N, C)

                # total_comms = embedded_comms.sum(dim=2, keepdim=True)
                # sent_comms = (total_comms - embedded_comms) / (N - 1)

                # dropout_mask = (torch.rand(B, T, N, 1, device=device) > 0.5).float()
                # sent_comms = sent_comms * dropout_mask

                world_comms = torch.zeros((B, T, N, NC, C), device=device)
                world_comms[:, :, :, 0, :] = embedded_comms 
                world_comms = world_comms.view(B, T, N, NC * C)

                actor_loss = F.cross_entropy(action_logits.reshape(-1, A), batch_actions.reshape(-1))

                flat_global_obs = batch_global_obs[:, :, 0, :].reshape(B * T, -1)
                values = critic(flat_global_obs)
                critic_loss = F.mse_loss(values.reshape(-1), batch_return_values.reshape(-1))

                tracker.update("validation_actor_loss", actor_loss.item() / T)
                tracker.update("validation_critic_loss", critic_loss.item() / T)
                
                loss = (actor_loss + critic_loss) / T

                h, c = actor_hidden_states
                actor_hidden_states = (h.detach(), c.detach())

                predicted_actions = torch.argmax(action_logits.reshape(-1, A), dim=-1)
                correct_predictions = (predicted_actions == batch_actions.reshape(-1)).sum().item()
                total_predictions = batch_actions.numel()

                tracker.update("validation_accuracy", correct_predictions / total_predictions)


        print(f"Epoch {epoch+1}/{config.imitation.epochs} | Train Accuracy: {tracker.get_average('train_accuracy'):.4f} | Val Accuracy: {tracker.get_average('validation_accuracy'):.4f}")
        logger.log(epoch, tracker)
        tracker.reset_tracker()

    test_loss = 0.0
    test_accuracy = 0.0
    
    with torch.no_grad():
        for batch_obs, batch_global_obs, batch_actions, batch_targets, batch_critic_values, batch_return_values in test_loader:

            B, T, N, _ = batch_obs.shape

            actor_hidden_states = actor.init_hidden(batch_size=B)
            world_comms = torch.zeros((B, N, NC * C), device=device)

            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()

            for t in range(T):
                
                obs_t = batch_obs[:, t]
                actions_t = batch_actions[:, t]
                global_obs_t = batch_global_obs[:, t]
                returns_t = batch_return_values[:, t]

                action_logits, lstm_output, actor_hidden_states = actor(obs_t, actor_hidden_states, world_comms)
                
                # comm_output, _, _ = actor.comm_protocol.get_comms_during_rollout(lstm_output)
                # world_comms = comm_output.detach()

                loss_actor = F.cross_entropy(action_logits.reshape(-1, A), actions_t.reshape(-1))
                values = critic(global_obs_t[:, 0, :])
                loss_critic = F.mse_loss(values.reshape(-1), returns_t.reshape(-1))
                
                loss = (loss_actor + loss_critic) / T

                h, c = actor_hidden_states
                actor_hidden_states = (h.detach(), c.detach())

                test_loss += loss.item()

                predicted_actions = torch.argmax(action_logits.reshape(-1, A), dim=-1)
                correct_predictions = (predicted_actions == actions_t.reshape(-1)).sum().item()
                total_predictions = actions_t.numel()

                test_accuracy += correct_predictions / total_predictions

            
        avg_test_loss = test_loss / len(test_loader)
        avg_test_accuracy = test_accuracy / (len(test_loader) * T)

        print(f"Final Test Set Loss: {avg_test_loss:.4f} | Test Set Accuracy: {avg_test_accuracy:.4f}")

    # Save Base
    save(save_folder, actor, critic, actor_optimizer, critic_optimizer, config)
