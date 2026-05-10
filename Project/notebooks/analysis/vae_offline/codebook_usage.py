# %%
# Imports
import sys
import os
sys.path.append(os.path.abspath('../../..'))


from controller.marl.main import setup_language_analysis
from controller.marl.core.config import Config, GenerativeLangType
from controller.marl.runners.sim_runner import run_sim


from project_paths import PROJECT_ROOT, FIGURES_DIR


import torch
from controller.marl.core.datasets import FilteredObsData
from torch.utils.data import DataLoader
import torch.nn.functional as F


from notebooks.plt_style import set_style
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

set_style()

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %%
config = Config.from_yaml(PROJECT_ROOT / "configs")

# %%
# K means 
language = "2026-04-16_15-47-58" # 75%
language = "2026-04-16_16-10-40" # 90.6%
language = "2026-04-16_16-34-01" # 84.4%

# No K means
language = "2026-04-16_17-03-57" # 81.2%
language = "2026-04-16_17-26-06" # 90.6%
language = "2026-04-16_17-48-24" # 65.6%

# 5x5 VQ-VAE
language = "2026-04-19_15-04-51"

# SQ-VAE
language = "2026-04-22_01-47-58" # 65% - old hp
language = "2026-04-22_10-59-55" # Better

language = "2026-04-23_16-52-17" # temp

config.comms.comm_folder = language
system, config = setup_language_analysis(config, device)

# %%
obs_logs_file = "./temp.csv"

# %%
run_sim(system, config, device, 1, collect_obs_file=obs_logs_file, optimal=True)

# %%
GO = system["sim"].get_global_obs_dim()
mask = torch.tensor(system["sim"].get_agent_external_obs_mask(0), dtype=torch.bool, device=device)
dataset = FilteredObsData(obs_logs_file, system["act_shape"][0], GO, mask, device)

dataloader = DataLoader(dataset, batch_size=config.aim_training.aim_batch_size, shuffle=True)

# %%
aim = system["aim"].to(device)
aim.eval()
print("loaded")

# %%
all_indices = []

with torch.no_grad():
    for batch_obs, _, _, _, _, _ in dataloader:
        batch_obs = batch_obs.to(device)
        
        is_hq_vae = config.comms.autoencoder_type == GenerativeLangType.HQ_VAE
        is_sq = config.comms.autoencoder_type == GenerativeLangType.SQ_VAE
        
        if is_hq_vae:
            latent, _ = aim.encoder.get_continuous_latents(batch_obs)
            embedding = aim.encoder.latent_handler.top_quantiser.get_embedding(0).weight
            is_sq = True
        else:
            latent = aim.encoder.get_continuous_latent(batch_obs)[0]
            embedding = aim.encoder.get_embedding(0).weight
            
        if is_sq or is_hq_vae:
            flat_input = latent.view(-1, latent.shape[-1])
            x_norm = F.normalize(flat_input, p=2.0, dim=-1)
            embed_norm = F.normalize(embedding, p=2.0, dim=-1)
            logits = torch.matmul(x_norm, embed_norm.t())
            indices = logits.argmax(dim=-1)
        else:
            distances = torch.cdist(latent, embedding)
            indices = distances.argmin(-1)
            
        all_indices.append(indices.cpu().numpy().flatten())

all_indices = np.concatenate(all_indices)
vocab_size = config.comms.vocab_size
final_embedding = embedding


# %%
plt.figure(figsize=(12, 5))
sns.histplot(all_indices, bins=range(vocab_size + 1), color="royalblue", discrete=True)

plt.title(f"Latent Codebook Usage Histogram ({config.comms.autoencoder_type.name})", fontsize=14, pad=15)
plt.xlabel("Discrete Codebook Index", fontsize=12)
plt.ylabel("Frequency of Use", fontsize=12)
plt.xlim(-0.5, vocab_size - 0.5)

unique_codes = np.unique(all_indices)
num_unique_codes = len(unique_codes)
utilization = (num_unique_codes / vocab_size) * 100

stats_text = f"Total Observations: {len(all_indices)}\nActive Codes: {num_unique_codes} / {vocab_size}\nUtilisation: {utilization:.1f}%"
plt.text(
    0.02,
    0.85,
    f"Total Observations: {len(all_indices)}\nActive Codes: {num_unique_codes} / {vocab_size}\nUtilisation: {utilization:.1f}%",
    transform=plt.gca().transAxes,
    fontsize=12, 
    bbox={
        "facecolor": "white",
        "alpha": 0.9,
        "edgecolor": "lightgray",
        "boxstyle": "round,pad=0.5"
    })

plt.grid(axis='y', alpha=0.3)
sns.despine()
plt.tight_layout()
plt.show()

# %%

codebook_tensors = final_embedding.cpu().detach().numpy()

pca = PCA(n_components=2)
codes_2d = pca.fit_transform(codebook_tensors)

plt.figure(figsize=(10, 10))

plt.scatter(codes_2d[:, 0], codes_2d[:, 1], c='lightgray', s=50, label='Unused Codes')

active_2d = codes_2d[unique_codes]
plt.scatter(active_2d[:, 0], active_2d[:, 1], c='red', s=100, label='Active Codes', edgecolors='black')

for idx, code_idx in enumerate(unique_codes):
    plt.annotate(f" {code_idx}", (active_2d[idx, 0], active_2d[idx, 1]), fontsize=10, fontweight='bold')

plt.title(f"PCA Projection of Latent Language Space (VQ-VAE)", fontsize=14)
plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
# plt.savefig(FIGURES_DIR / "pca-vq-vae", dpi=1200)
plt.show()



