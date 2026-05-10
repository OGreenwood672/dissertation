# %%
# Imports
import sys
import os
sys.path.append(os.path.abspath('../../..'))


from controller.marl.main import setup_language_analysis
from controller.marl.core.config import Config, GenerativeLangType
from controller.marl.runners.sim_runner import run_sim


from project_paths import PROJECT_ROOT


import torch
from controller.marl.core.datasets import FilteredObsData
from torch.utils.data import DataLoader
import torch.nn.functional as F


from notebooks.plt_style import set_style
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

set_style()

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %%
config = Config.from_yaml(PROJECT_ROOT / "configs")

# %%

language = "2026-04-22_12-38-35" # bad usage
language = "2026-04-22_14-29-19" # 32 not great usage
language = "2026-04-22_15-16-53" # 16

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
print("Loaded")

# %%
all_top_indices = []
all_bottom_indices = []
all_composite_indices = []

vocab_size = config.comms.vocab_size

with torch.no_grad():
    for batch_obs, _, _, _, _, _ in dataloader:
        batch_obs = batch_obs.to(device)
        
        top_latent, bottom_latent = aim.encoder.get_continuous_latents(batch_obs)
        top_embedding = aim.encoder.latent_handler.top_quantiser.get_embedding(0).weight
        bottom_embedding = aim.encoder.latent_handler.bottom_quantiser.get_embedding(0).weight
        
        x_norm_top = F.normalize(top_latent.view(-1, top_latent.shape[-1]), p=2.0, dim=-1)
        embed_norm_top = F.normalize(top_embedding, p=2.0, dim=-1)
        top_indices = torch.matmul(x_norm_top, embed_norm_top.t()).argmax(dim=-1)
        
        x_norm_bottom = F.normalize(bottom_latent.view(-1, bottom_latent.shape[-1]), p=2.0, dim=-1)
        embed_norm_bottom = F.normalize(bottom_embedding, p=2.0, dim=-1)
        bottom_indices = torch.matmul(x_norm_bottom, embed_norm_bottom.t()).argmax(dim=-1)
        
        composite_indices = (top_indices * vocab_size) + bottom_indices
        
        all_top_indices.append(top_indices.cpu().numpy().flatten())
        all_bottom_indices.append(bottom_indices.cpu().numpy().flatten())
        all_composite_indices.append(composite_indices.cpu().numpy().flatten())

top_unique = len(np.unique(np.concatenate(all_top_indices)))
bottom_unique = len(np.unique(np.concatenate(all_bottom_indices)))
composite_unique = len(np.unique(np.concatenate(all_composite_indices)))

top_usage_pct = (top_unique / vocab_size) * 100
bottom_usage_pct = (bottom_unique / vocab_size) * 100
composite_usage_pct = (composite_unique / (vocab_size * vocab_size)) * 100

print(f"Top Codebook Usage (Macro):    {top_usage_pct:.1f}% ({top_unique}/{vocab_size})")
print(f"Bottom Codebook Usage (Micro): {bottom_usage_pct:.1f}% ({bottom_unique}/{vocab_size})")
print(f"Composite Usage (Total Vocab): {composite_usage_pct:.1f}% ({composite_unique}/{(vocab_size * vocab_size)})")


