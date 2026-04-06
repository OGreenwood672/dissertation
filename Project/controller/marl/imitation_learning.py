


from datetime import datetime
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F



from controller.marl.config import Config
from controller.marl.datasets import ImitationLearningDataset
from controller.marl.run_sim import run_sim
from project_paths import RESULTS_DIR

def save(actor, critic, actor_optimizer, critic_optimizer, config: Config):
    state = {
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "actor_optimizer": actor_optimizer.state_dict(),
        "critic_optimizer": critic_optimizer.state_dict(),
        "step": config.aim_training.obs_runs
    }

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder = RESULTS_DIR / config.comms.communication_type.value / "base"
    os.makedirs(folder, exist_ok=True)

    torch.save(state, folder / f"{timestamp}.pth")


def spatial_index(targets: torch.tensor, max_index: int):
    # x_bins * y_bins <= vocab_size
    x_bins = int(np.floor(np.sqrt(max_index)))
    y_bins = x_bins

    x = torch.clamp((targets[:, 0] * x_bins).long(), min=0, max=x_bins - 1)
    y = torch.clamp((targets[:, 1] * y_bins).long(), min=0, max=y_bins - 1)

    return y * x_bins + x

def imitation_learning(system, config: Config, device: torch.device):
    # New actor, critic, optimisers, obs file - create here?

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
    GO = sim.get_global_obs_dim() + N * NC * C
    A = system['act_shape'][0]

    obs_external_mask = torch.tensor(sim.get_agent_external_obs_mask(0), dtype=torch.bool, device=device)

    obs_logs_file = RESULTS_DIR / "obs_log.csv"

    if (
        (os.path.exists(obs_logs_file) and input(f"Obs logs already exist, would you like to overwrite? (y/N) ").lower() == "y")
        or not os.path.exists(obs_logs_file)
    ):

        RUNS = config.aim_training.obs_runs

        rewards = run_sim(system, config, device, RUNS, collect_obs_file=obs_logs_file, optimal=True)

        print(f"Obs collected: {RUNS * W * N * T}")
        print(f"Baseline: {np.mean(rewards)}")


    dataset = ImitationLearningDataset(obs_logs_file, O, GO, obs_external_mask, device)
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [0.8, 0.1, 0.1])

    print(f"Using {len(train_set)} datapoints for training...")

    training_loader, validation_loader, test_loader = (
        DataLoader(train_set, batch_size=config.aim_training.aim_batch_size, shuffle=False),
        DataLoader(val_set, batch_size=config.aim_training.aim_batch_size, shuffle=False),
        DataLoader(test_set, batch_size=config.aim_training.aim_batch_size, shuffle=False)
    )
    
    for epoch in range(config.aim_training.ae_epochs):

        train_loss = 0.0
        actor.train()
        critic.train()
        
        for batch_obs, batch_global_obs, batch_actions, batch_targets, batch_critic_values, batch_return_values in training_loader:

            batch_obs = batch_obs.to(device)
            batch_obs = batch_obs.unsqueeze(1)

            batch_global_obs = batch_global_obs.to(device)
            batch_actions = batch_actions.to(device)
            batch_targets = batch_targets.to(device)
            batch_critic_values = batch_critic_values.to(device)
            batch_return_values = batch_return_values.to(device)

            current_B = batch_obs.shape[0]
            real_B = current_B // N

            if real_B == 0: continue

            batch_obs = batch_obs.view(real_B, N, O)
            batch_global_obs = batch_global_obs.view(real_B, N, GO)
            batch_actions = batch_actions.view(real_B, N, 1)
            batch_targets = batch_targets.view(real_B, N, 3)
            batch_return_values = batch_return_values.view(real_B, N, 1)

            actor_hidden_states = actor.init_hidden(batch_size=real_B)

            flat_targets = batch_targets.view(-1, batch_targets.shape[-1])
            target_indices = spatial_index(flat_targets, config.comms.vocab_size)
            embedded_comms = actor.comm_protocol.aim_codebook[0, target_indices.long()].view(real_B, N, C)

            sent_comms = (embedded_comms.sum(dim=1, keepdim=True) / (N - 1)) - embedded_comms

            world_comms = torch.zeros((real_B, N, NC, C), device=device)
            world_comms[:, :, 0, :] = sent_comms
            world_comms = world_comms.view(real_B, N, NC * C)

            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()
            
            action_logits, _, _ = actor(batch_obs, actor_hidden_states, world_comms)
            values = critic(batch_global_obs[:, 0, :])
            
            loss_actor = F.cross_entropy(action_logits.view(-1, A), batch_actions.view(-1).long()) 
            loss_critic = F.mse_loss(values.view(-1), batch_return_values.view(-1))
            
            loss = loss_actor + loss_critic

            loss.backward()
            actor_optimizer.step()
            critic_optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(training_loader)

        val_loss = 0.0
        
        with torch.no_grad():
            for batch_obs, batch_global_obs, batch_actions, batch_targets, batch_critic_values, batch_return_values in validation_loader:

                batch_obs = batch_obs.to(device)
                batch_obs = batch_obs.unsqueeze(1)

                batch_global_obs = batch_global_obs.to(device)
                batch_actions = batch_actions.to(device)
                batch_targets = batch_targets.to(device)
                batch_critic_values = batch_critic_values.to(device)
                batch_return_values = batch_return_values.to(device)

                current_B = batch_obs.shape[0]
                real_B = current_B // N

                if real_B == 0: continue

                batch_obs = batch_obs.view(real_B, N, O)
                batch_global_obs = batch_global_obs.view(real_B, N, GO)
                batch_actions = batch_actions.view(real_B, N, 1)
                batch_targets = batch_targets.view(real_B, N, 3)
                batch_return_values = batch_return_values.view(real_B, N, 1)

                actor_hidden_states = actor.init_hidden(batch_size=real_B)

                flat_targets = batch_targets.view(-1, batch_targets.shape[-1])
                target_indices = spatial_index(flat_targets, config.comms.vocab_size)
                embedded_comms = actor.comm_protocol.aim_codebook[0, target_indices.long()].view(real_B, N, C)

                sent_comms = (embedded_comms.sum(dim=1, keepdim=True) / (N - 1)) - embedded_comms

                world_comms = torch.zeros((real_B, N, NC, C), device=device)
                world_comms[:, :, 0, :] = sent_comms
                world_comms = world_comms.view(real_B, N, NC * C)

                actor_optimizer.zero_grad()
                critic_optimizer.zero_grad()
                
                action_logits, _, _ = actor(batch_obs, actor_hidden_states, world_comms)
                values = critic(batch_global_obs[:, 0, :])
                
                loss_actor = F.cross_entropy(action_logits.view(-1, A), batch_actions.view(-1).long()) 
                loss_critic = F.mse_loss(values.view(-1), batch_return_values.view(-1))
                
                loss = loss_actor + loss_critic

                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(validation_loader)

        print(f"Epoch {epoch+1}/{config.aim_training.ae_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    test_loss = 0.0
    
    with torch.no_grad():
        for batch_obs, batch_global_obs, batch_actions, batch_targets, batch_critic_values, batch_return_values in validation_loader:

            batch_obs = batch_obs.to(device)
            batch_obs = batch_obs.unsqueeze(1)

            batch_global_obs = batch_global_obs.to(device)
            batch_actions = batch_actions.to(device)
            batch_targets = batch_targets.to(device)
            batch_critic_values = batch_critic_values.to(device)
            batch_return_values = batch_return_values.to(device)

            current_B = batch_obs.shape[0]
            real_B = current_B // N

            if real_B == 0: continue

            batch_obs = batch_obs.view(real_B, N, O)
            batch_global_obs = batch_global_obs.view(real_B, N, GO)
            batch_actions = batch_actions.view(real_B, N, 1)
            batch_targets = batch_targets.view(real_B, N, 3)
            batch_return_values = batch_return_values.view(real_B, N, 1)

            actor_hidden_states = actor.init_hidden(batch_size=real_B)

            flat_targets = batch_targets.view(-1, batch_targets.shape[-1])
            target_indices = spatial_index(flat_targets, config.comms.vocab_size)
            embedded_comms = actor.comm_protocol.aim_codebook[0, target_indices.long()].view(real_B, N, C)

            sent_comms = (embedded_comms.sum(dim=1, keepdim=True) / (N - 1)) - embedded_comms

            world_comms = torch.zeros((real_B, N, NC, C), device=device)
            world_comms[:, :, 0, :] = sent_comms
            world_comms = world_comms.view(real_B, N, NC * C)

            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()
            
            action_logits, _, _ = actor(batch_obs, actor_hidden_states, world_comms)
            values = critic(batch_global_obs[:, 0, :])
            
            loss_actor = F.cross_entropy(action_logits.view(-1, A), batch_actions.view(-1).long()) 
            loss_critic = F.mse_loss(values.view(-1), batch_return_values.view(-1))
            
            loss = loss_actor + loss_critic

            test_loss += loss.item()
            
        avg_test_loss = test_loss / len(test_loader)

        print(f"Final Test Set Loss: {avg_test_loss:.4f}")

    # Save Base
    save(actor, critic, actor_optimizer, critic_optimizer, config)
