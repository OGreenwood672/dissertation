

import os

import numpy as np
from sklearn.cluster import MiniBatchKMeans
import torch
from torch.utils.data import DataLoader

import torch.nn.functional as F


from controller.marl.config import CommunicationType, Config, GenerativeLangType
from controller.marl.models.encoders.hq_vae import HQ_VAE
from controller.marl.models.encoders.sq_vae import SQ_VAE
from controller.marl.run_sim import run_sim
from controller.marl.datasets import FilteredObsData
from controller.marl.models.aim import AIM
from controller.marl.logger import Logging, MetricTracker

from project_paths import RESULTS_DIR


def fit_rq_stream(model, residuals, quantiser, latent_dim, device: torch.device):
    curr_residuals = residuals.reshape(-1, latent_dim)
    
    for rq_level in range(quantiser.rq_levels):
        
        kmeans = MiniBatchKMeans(
            n_clusters=model.comm_config.vocab_size, 
            batch_size=1024,
            n_init="auto",
        )
        
        norm_residuals = curr_residuals / (np.linalg.norm(curr_residuals, axis=-1, keepdims=True) + 1e-5)
        kmeans.fit(norm_residuals)

        centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=device)
        if isinstance(model.encoder, SQ_VAE):
            centers = F.normalize(centers, p=2.0, dim=-1)
        quantiser.get_embedding(rq_level).weight.data = centers
        cluster_assignments = kmeans.predict(norm_residuals)        
        curr_residuals = curr_residuals - kmeans.cluster_centers_[cluster_assignments]

def set_priors(model, dataloader, device='cuda'):
    print(f"Setting prior with {len(dataloader.dataset)} datapoints")
    
    model.encoder.eval()
    
    is_hq_vae = model.comm_config.autoencoder_type == GenerativeLangType.HQ_VAE
    
    if is_hq_vae:
        top_latents, bottom_latents = [], []
    else:
        latents = []

    collected_samples = 0
    with torch.no_grad():
        for batch_obs, _, _, _, _, _ in dataloader:
            batch_obs = batch_obs.to(device)
            
            if is_hq_vae:
                top, bottom = model.encoder.get_continuous_latents(batch_obs)
                top_latents.append(top.cpu().numpy())
                bottom_latents.append(bottom.cpu().numpy())
            else:
                latent = model.encoder.get_continuous_latent(batch_obs)[0]
                latents.append(latent.cpu().numpy())
                
            collected_samples += batch_obs.shape[0]

    if is_hq_vae:
        top_residuals = np.concatenate(top_latents, axis=0)
        bottom_residuals = np.concatenate(bottom_latents, axis=0)
        
        fit_rq_stream(model, top_residuals, model.encoder.latent_handler.top_quantiser, model.encoder.latent_dims[0], device)
        
        fit_rq_stream(model, bottom_residuals, model.encoder.latent_handler.bottom_quantiser, model.encoder.latent_dims[1], device)
    else:
        curr_residuals = np.concatenate(latents, axis=0)
        fit_rq_stream(model, curr_residuals, model.encoder.latent_handler, model.encoder.latent_dim, device)

    model.encoder.train()
    print("Priors set!")


def train_language(system, config: Config, device: torch.device, use_optimal: bool = False):

    torch.set_flush_denormal(True)

    W = config.training.worlds_parallised
    N = system['num_agents']
    T = config.training.simulation_timesteps
    OBS_DIM = system['agent_obs_shape'][0]
    GLOBAL_OBS_DIM = system['sim'].get_global_obs_dim() + N * config.comms.communication_size
    COMM_DIM = config.comms.communication_size
    VOCAB_SIZE = config.comms.vocab_size
    NC = config.comms.num_comms
    ACTION_COUNT = system['act_shape'][0]
    RQ_LAYERS = config.comms.rq_levels

    obs_external_mask = torch.tensor(system["sim"].get_agent_external_obs_mask(0), dtype=torch.bool, device=device)
    obs_mask = torch.tensor(system["sim"].get_agent_obs_mask(0), dtype=torch.bool, device=device)[obs_external_mask]


    OBS_DIM = obs_external_mask.sum().int().item()

    cm = system["checkpoint_manager"]


    if not use_optimal:
        # Load obs logs
        obs_logs_file = cm.get_result_path() / "obs_log.csv"
        if (
            (os.path.exists(obs_logs_file) and input(f"Obs logs already exist, would you like to overwrite? (y/N) ").lower() == "y")
            or not os.path.exists(obs_logs_file)
        ):

            RUNS = config.aim_training.obs_runs

            system["actor"].eval()
            system["critic"].eval()

            rewards = run_sim(system, config, device, RUNS, collect_obs_file=obs_logs_file)

            print(f"Obs collected: {RUNS * W * N * T}")
            print(f"Baseline: {np.mean(rewards)}")
        
    else:
        obs_logs_file = RESULTS_DIR / "obs_log.csv"
        if (
            (os.path.exists(obs_logs_file) and input(f"Obs logs already exist, would you like to overwrite? (y/N) ").lower() == "y")
            or not os.path.exists(obs_logs_file)
        ):

            RUNS = config.aim_training.obs_runs

            rewards = run_sim(system, config, device, RUNS, collect_obs_file=obs_logs_file, optimal=True, noise=config.aim_training.obs_runs_noise)

            print(f"Obs collected: {RUNS * W * N * T}")
            print(f"Baseline: {np.mean(rewards)}")

    print("Loading datapoints...")

    dataset = FilteredObsData(obs_logs_file, OBS_DIM, GLOBAL_OBS_DIM, obs_external_mask, device)
    prior_set, train_set, val_set, test_set = torch.utils.data.random_split(dataset, [0.1, 0.7, 0.1, 0.1])

    print(f"Using {len(train_set)} datapoints for training...")

    prior_loader, training_loader, validation_loader, test_loader = (
        DataLoader(prior_set, batch_size=config.aim_training.aim_batch_size, shuffle=True),
        DataLoader(train_set, batch_size=config.aim_training.aim_batch_size, shuffle=True),
        DataLoader(val_set, batch_size=config.aim_training.aim_batch_size, shuffle=False),
        DataLoader(test_set, batch_size=config.aim_training.aim_batch_size, shuffle=False)
    )

    num_training_steps = config.aim_training.ae_epochs * (len(training_loader) // config.aim_training.aim_batch_size)
    
    aim = AIM(OBS_DIM, config.comms, config.aim_training, obs_mask=obs_mask, num_training_steps=num_training_steps).to(device)

    set_priors(aim, prior_loader, device='cuda')

    optimizer = torch.optim.Adam(aim.parameters(), lr=config.aim_training.aim_learning_rate)

    num_codebooks = 0
    if config.comms.communication_type == CommunicationType.AIM:
        if aim.comm_config.autoencoder_type == GenerativeLangType.HQ_VAE:
            num_codebooks = config.comms.rq_levels * 2
        else:
            num_codebooks = config.comms.rq_levels
    elif config.comms.communication_type == CommunicationType.DISCRETE:
        num_codebooks = config.comms.rq_levels

    tracker = MetricTracker()
    logger = Logging(str(cm.get_result_path()), config, 0, num_codebooks, "language")

    for epoch in range(config.aim_training.ae_epochs):

        aim.train()

        for batch_obs, _, _, _, _, _ in training_loader:

            batch_obs = batch_obs.to(device)

            optimizer.zero_grad()
            loss = aim(batch_obs, tracker)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(aim.parameters(), max_norm=1.0)

            optimizer.step()

        train_keys = list(tracker.metrics.keys())
        for key in train_keys:
            tracker.metrics[f"train_{key}"] = tracker.metrics.pop(key)
            tracker.counts[f"train_{key}"] = tracker.counts.pop(key)

        aim.eval()
        
        with torch.no_grad():
            for batch_obs, _, _, _, _, _ in validation_loader:
                batch_obs = batch_obs.to(device)
                aim(batch_obs, tracker) 
        
        not_train_keys = list(filter(lambda x: not x.startswith("train"), tracker.metrics.keys()))
        for key in not_train_keys:
            tracker.metrics[f"validation_{key}"] = tracker.metrics.pop(key)
            tracker.counts[f"validation_{key}"] = tracker.counts.pop(key)
                

        print(f"Epoch {epoch+1}/{config.aim_training.ae_epochs}")
        train_stats = []
        train_keys = list(filter(lambda x: x.startswith("train"), tracker.metrics.keys()))
        for key in train_keys:
            train_stats.append(f"{key[len('train_'):]}: {tracker.get_average(key):.4f}")
        print("      TRAINING   " + " | ".join(train_stats))

        validation_stats = []
        validation_keys = list(filter(lambda x: x.startswith("validation"), tracker.metrics.keys()))
        for key in validation_keys:
            validation_stats.append(f"{key[len('validation_'):]}: {tracker.get_average(key):.4f}")
        print("    VALIDATION   " + " | ".join(validation_stats))



        logger.log(epoch, tracker)
        tracker.reset_tracker()

    aim.eval()
    test_loss = 0.0

    with torch.no_grad():
        for batch_obs, _, _, _, _, _ in test_loader:
            batch_obs = batch_obs.to(device)
            loss = aim(batch_obs)
            test_loss += loss.item()

    print(f"Final Test Set Loss: {test_loss / len(test_loader):.4f}")

    # save language
    aim.save()
