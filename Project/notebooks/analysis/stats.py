# %%
# Imports
import sys
import os
sys.path.append(os.path.abspath('../..'))


from controller.marl.main import setup
from controller.marl.core.config import Config
from controller.marl.runners.sim_runner import run_sim
from controller.marl.core.config import CommunicationType


from project_paths import PROJECT_ROOT, FIGURES_DIR

import torch

from notebooks.plt_style import set_style
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from tqdm import tqdm

import gc

from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd


set_style()

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %%
runs = [
    [
        ("aim", 70),
        ("aim", 70),
        ("aim", 73),
    ],
    [
        ("none", 102),
        ("none", 104),
        ("none", 106),
    ],
    [
        ("discrete", 46),
        ("discrete", 48),
        ("discrete", 49),
    ],
    # [
    #     ("no reflection", 62),
    #     ("no reflection", 75),
    #     ("no reflection", 77)
    # ],
    # [
    #     ("reflective", 7),
    #     ("reflective", 9),
    #     ("reflective", 11)
    # ]

    # [
    #     ("aim", 106),
    #     ("aim", 108),
    #     ("aim", 110),
    # ],

    # [
    #     ("reflective", 27),
    #     ("reflective", 29),
    #     ("reflective", 31)
    # ]
]

# %%
results = []

for protocol in runs:
    results.append([])
    for comms, seed in protocol:

        config = Config.from_yaml(PROJECT_ROOT / "configs")

        if comms == "no reflection":
            comms = "aim"

        config.training.seed = seed
        config.comms.communication_type = CommunicationType(comms.upper())

        try:
            system, config = setup(config, device, load_agent_architecture=True)
        except:
            print("Fail")
            continue


        actor = system["actor"]
        actor.eval()

        avg_reward_mean_per_step = run_sim(system, config, device, 8)
        results[-1].append(avg_reward_mean_per_step)

        del system
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

# %%
print(results)

# %%
df = pd.DataFrame({
    "protocol": [j[0] for i in runs for j in i],
    "score": [j for i in results for j in i]
})
groups = [g["score"].values for _, g in df.groupby("protocol")]

# %%

print("ASSUMPTIONS")
for name, g in df.groupby("protocol"):
    shapiro_stat, shapiro_p = stats.shapiro(g["score"])
    print(f"{name}: Shapiro p = {shapiro_p:.4f}")

levene_stat, levene_p = stats.levene(*groups)
print(f"Levene p = {levene_p:.4f}")

if len(runs) > 2:

    print("\nANOVA F STAT")
    f_stat, p_val = stats.f_oneway(*groups)
    print(f"ANOVA F = {f_stat:.4f}, p = {p_val:.4f}")

    if p_val < 0.05:
        print("\nTUKEY")
        tukey = pairwise_tukeyhsd(endog=df["score"], groups=df["protocol"], alpha=0.05)
        print(tukey)

else:
    print("Welch's t-test")

    g1 = df[df["protocol"] == runs[0][0][0]]["score"]
    g2 = df[df["protocol"] == runs[1][0][0]]["score"]

    t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=False)
    print(f"t = {t_stat:.4f}, p = {p_val:.4f}")

    print(f"aim mean: {g1.mean():.4f}, std: {g1.std(ddof=1):.4f}")
    print(f"reflective mean: {g2.mean():.4f}, std: {g2.std(ddof=1):.4f}")
    print(f"aim n: {len(g1)}, reflective n: {len(g2)}")
