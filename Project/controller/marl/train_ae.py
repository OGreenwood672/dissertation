

import os

import numpy as np
from sklearn.cluster import MiniBatchKMeans
import torch
from torch.utils.data import DataLoader


from controller.marl.run_sim import run_sim
from controller.marl.datasets import ObsData
from controller.marl.models.aim import AIM


def set_priors(model, dataloader, device='cuda'):
    print(f"Setting prior with {len(dataloader)} datapoints")
    
    model.encoder.eval()
    
    latents = []
    collected_samples = 0
    with torch.no_grad():
        for batch_obs, _ in dataloader:
            
            batch_obs = batch_obs.to(device)
            
            _, latent = model.encoder(batch_obs)

            latents.append(latent.cpu().numpy())
            collected_samples += latent.shape[0]
    
    curr_residuals = np.concatenate(latents, axis=0).reshape(-1, model.encoder.latent_dim)

    for hq_level in range(model.encoder.hq_vae):
        
        kmeans = MiniBatchKMeans(
            n_clusters=model.vocab_size, 
            batch_size=1024,
            n_init="auto",
        )
        kmeans.fit(curr_residuals)
                
        model.encoder.get_embedding(hq_level).data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=device)
        
        cluster_assignments = kmeans.predict(curr_residuals)        
        curr_residuals = curr_residuals - kmeans.cluster_centers_[cluster_assignments]
    
    model.encoder.train()


def train_language(system, config, device):

    torch.set_flush_denormal(True)

    W = config.worlds_parallised
    N = system['num_agents']
    T = config.simulation_timesteps
    OBS_DIM = system['agent_obs_shape'][0]
    COMM_DIM = config.communication_size
    VOCAB_SIZE = config.vocab_size
    NC = config.num_comms
    ACTION_COUNT = system['act_shape'][0]
    HQ_LAYERS = config.hq_layers

    obs_mask = system["sim"].get_agent_obs_mask(0)

    cm = system["checkpoint_manager"]

    # Load obs logs
    obs_logs_file = cm.get_result_path() / "obs_log.csv"
    if (
        (os.path.exists(obs_logs_file) and input(f"Obs logs already exist, would you like to overwrite? (y/N) ").lower() == "y")
        or not os.path.exists(obs_logs_file)
    ):

        RUNS = config.obs_runs

        rewards = run_sim(system, config, device, RUNS, collect_obs_file=obs_logs_file)

        print(f"Obs collected: {RUNS * W * N * T}")
        print(f"Baseline: {np.mean(rewards)}")

    print("Loading datapoints...")

    dataset = ObsData(obs_logs_file, ACTION_COUNT)
    prior_set, train_set, val_set, test_set = torch.utils.data.random_split(dataset, [0.1, 0.7, 0.1, 0.1])

    print(f"Using {len(train_set)} datapoints for training...")

    prior_loader, training_loader, validation_loader, test_loader = (
        DataLoader(prior_set, batch_size=config.aim_batch_size, shuffle=True),
        DataLoader(train_set, batch_size=config.aim_batch_size, shuffle=True),
        DataLoader(val_set, batch_size=config.aim_batch_size, shuffle=False),
        DataLoader(test_set, batch_size=config.aim_batch_size, shuffle=False)
    )

    num_training_steps = config.ae_epochs * (len(training_loader) // config.aim_batch_size)
    
    aim = AIM(
        OBS_DIM, ACTION_COUNT,
        512, COMM_DIM * NC, VOCAB_SIZE,
        commitment_cost=0.25,
        num_training_steps=num_training_steps, hq_vae=HQ_LAYERS,
        init_temperature=config.init_temperature, min_temperature=config.min_temperature,
        autoencoder_type=config.autoencoder_type, obs_mask=torch.tensor(obs_mask, dtype=torch.bool, device=device)
    ).to(device)

    set_priors(aim, prior_loader, device='cuda')

    optimizer = torch.optim.Adam(aim.parameters(), lr=config.aim_learning_rate)

    for epoch in range(config.ae_epochs):

        train_loss = 0.0
        aim.train()

        for batch_obs, batch_rewards in training_loader:

            batch_obs = batch_obs.to(device)
            batch_rewards = batch_rewards.to(device)

            optimizer.zero_grad()
            loss = aim(batch_obs, None)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(aim.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(training_loader)

        aim.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_obs, batch_rewards in validation_loader:

                batch_obs = batch_obs.to(device)
                batch_rewards = batch_rewards.to(device)

                loss = aim(batch_obs, None) 
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(validation_loader)

        print(f"Epoch {epoch+1}/{config.ae_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    aim.eval()
    test_loss = 0.0

    with torch.no_grad():
        for batch_obs, batch_rewards in test_loader:

            batch_obs = batch_obs.to(device)
            batch_rewards = batch_rewards.to(device)

            loss = aim(batch_obs, None)
            test_loss += loss.item()

        # show_loader = DataLoader(test_set, batch_size=1, shuffle=True)
        # for obs, action in show_loader:
        #     print("OBS", obs)
        #     print("Action", action)
        #     construction_loss, reward_loss, reconstructed, quantised = aim.check(obs.to(device), action.to(device))
        #     print("Reconstruction Loss", construction_loss)
        #     print("Reward Loss", reward_loss)
        #     print("Quantised", quantised)
        #     print("Reconstructed", reconstructed)
        #     print("\n")

    print(f"Final Test Set Loss: {test_loss / len(test_loader):.4f}")

    # save language
    aim.save()
