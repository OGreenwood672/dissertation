# %%
# Imports
import sys
import os
sys.path.append(os.path.abspath('../..'))


from controller.marl.main import setup
from controller.marl.core.config import Config
from controller.marl.runners.sim_runner import run_sim
from controller.marl.models.aim import AIM

from project_paths import PROJECT_ROOT, FIGURES_DIR


import torch
from controller.marl.core.datasets import FilteredObsData
from torch.utils.data import DataLoader


from notebooks.plt_style import set_style
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans


# %%
set_style()

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %%
config = Config.from_yaml(PROJECT_ROOT / "configs")

# %%
system, config = setup(config, device)

# %%
run_sim(system, config, device, 5, collect_obs_file="./temp.csv", optimal=True)

# %%
obs_logs_file = "./temp.csv"

GO = system["sim"].get_global_obs_dim()
mask = torch.tensor(system["sim"].get_agent_external_obs_mask(0), dtype=torch.bool, device=device)
dataset = FilteredObsData(obs_logs_file, system["act_shape"][0], GO, mask, device)

dataloader = DataLoader(dataset, batch_size=config.aim_training.aim_batch_size, shuffle=True)

# %%

obs_external_mask = torch.tensor(system["sim"].get_agent_external_obs_mask(0), dtype=torch.bool, device=device)
obs_mask = torch.tensor(system["sim"].get_agent_obs_mask(0), dtype=torch.bool, device=device)[obs_external_mask]

OBS_DIM = obs_external_mask.sum().int().item()

num_training_steps = config.aim_training.ae_epochs * len(dataloader)


aim = AIM(OBS_DIM, config.comms, config.aim_training, obs_mask=obs_mask, num_training_steps=num_training_steps).to(device)

# %%
aim.encoder.eval()
continuous_latents = []

with torch.no_grad():
    for batch in tqdm(dataloader):
        
        obs = batch[0].to(device)
        
        latent = aim.encoder.get_continuous_latent(obs)[0]
        continuous_latents.append(latent.cpu().numpy())

residuals = np.concatenate(continuous_latents, axis=0)

# %%
inertias = []
vocab_sizes = range(2, 65, 2)

flat_residuals = residuals.reshape(-1, residuals.shape[-1])
for k in tqdm(vocab_sizes):
    kmeans = MiniBatchKMeans(n_clusters=k, batch_size=1024, n_init="auto")
    kmeans.fit(flat_residuals)
    inertias.append(kmeans.inertia_)
    
plt.figure(figsize=(8, 5))

plt.plot(vocab_sizes, inertias, 'bx-')
plt.xlabel('Vocabulary Size (k)')
plt.ylabel('Inertia (Quantisation Error)')
plt.title('Elbow Method for Optimal Vocab Size')

plt.savefig(FIGURES_DIR / "elbow-curve.png", dpi=900)
plt.close()


