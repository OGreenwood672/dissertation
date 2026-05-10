# %%
import sys
import os
sys.path.append(os.path.abspath('../../..'))
sys.path.append(os.path.abspath('../..'))
sys.path.append(os.path.abspath('..'))

from loader import load_vae_training_log, load_specified_vae_training_log


from controller.marl.core.config import GenerativeLangType

from project_paths import FIGURES_DIR

from plt_style import set_style
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

set_style()

# %%
# VQ-VAE
TZs = [
    [ # vq-vae
        "2026-04-23_14-23-49", # 62.5%
        "2026-04-23_15-35-02", # 71.9%
        "2026-04-23_16-52-17" # 65.6%
    ],
    [ # sq-vae
        "2026-04-23_18-01-54", # 87.5%
        "2026-04-23_19-44-09", # 96.9%
        "2026-04-23_20-22-07" # 100%
    ],
    [ # vq-vae - no prior
        "2026-04-23_20-52-29", # 93.8%
        "2026-04-23_21-30-35", # 84.4%
        "2026-04-23_22-07-49" # 93.8
    ],
    [ # sq-vae - no prior
        "2026-04-23_22-36-47", # 62.5%
        "2026-04-23_23-06-17", # 75%
        "2026-04-23_23-36-39" # 84.4
    ]
]

# %%
# df = load_vae_training_log(GenerativeLangType.VQ_VAE, most_recent=3)
df_vq = load_specified_vae_training_log(TZs[0])
df_sq = load_specified_vae_training_log(TZs[1])
df_vq_no_priors = load_specified_vae_training_log(TZs[2])
df_sq_no_priors = load_specified_vae_training_log(TZs[3])

# %%
n_lang = len(TZs[0])
ROWS = 50

final_indecies = np.arange(0, n_lang * ROWS, ROWS) + ROWS - 1
for df in [df_vq, df_sq, df_vq_no_priors, df_sq_no_priors]:
    final_values = df_vq.iloc[final_indecies]["validation_reconstruction_loss"].values
    mean = final_values.mean()
    std  = final_values.std()

    print("Final values:", final_values)
    print("Mean:", mean, "Std:", std)


# %%
pairs = [
    ("train_reconstruction_loss", "validation_reconstruction_loss", "Reconstruction Loss", "recon_loss_eval.png"),
    ("train_commitment_loss_0", "validation_commitment_loss_0", "Commitment Loss", "commit_loss_eval.png"),
]

for train_key, val_key, main_label, filename in pairs:
    fig, ax = plt.subplots(figsize=(9, 5))

    sns.lineplot(
        data=df_vq,
        x="timestep",
        y=val_key,
        ax=ax,
        errorbar="ci",
        label="VQ-VAE",
        color="tab:blue",
    )

    sns.lineplot(
        data=df_vq_no_priors,
        x="timestep",
        y=val_key,
        ax=ax,
        errorbar="ci",
        label="VQ-VAE (without priors)",
        color="tab:green",
    )

    if train_key == "train_reconstruction_loss":
        sns.lineplot(
            data=df_sq,
            x="timestep",
            y=val_key,
            ax=ax,
            errorbar="ci",
            label="SQ-VAE",
            color="tab:orange",
        )

        sns.lineplot(
            data=df_sq_no_priors,
            x="timestep",
            y=val_key,
            ax=ax,
            errorbar="ci",
            label="SQ-VAE (without priors)",
            color="tab:red",
        )

    plt.title(f"Average {main_label} when training VQ-VAE vs SQ-VAE (95% CI)")
    plt.ylabel(main_label)
    plt.xlabel("Epoch")
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.tight_layout()
    # plt.savefig(FIGURES_DIR / filename, dpi=1200, bbox_inches='tight', facecolor='white')
    plt.show()
    plt.close('all')


