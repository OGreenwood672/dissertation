# %%
# Imports
import sys
import os
sys.path.append(os.path.abspath('../..'))
sys.path.append(os.path.abspath('..'))

from loader import load_log_file
from plt_style import set_style
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy import stats


from project_paths import FIGURES_DIR

# %%
# CONFIG

FILES = [
    # 5x5
    # ("aim", 2, ""),
    # ("aim", 70, "2"),
    # ("aim", 73, "3"),
    # ("continuous", 15, "5x5"),
    # ("none", 102, ""),
    # ("none", 104, "2"),
    # ("none", 106, "3"),


    # ("aim", 19, "rubbish sq-vae"),
    # ("aim", 47, "hq-vae"),
    # ("aim", 67, "hq-vae2"),
    
    # ("discrete", 46, ""),
    # ("discrete", 48, "2"),
    # ("discrete", 49, "3"),

    # ("aim", 62, "with no reflection"),
    # ("aim", 75, "with no reflection 2"),
    # ("aim", 77, "with no reflection 3"),

    # ("aim", 99, "comm-better"),
    # ("aim", 104, "comm-better 2"),

    # ("reflective", 7, ""),
    # ("reflective", 9, "2")

    ("aim", 106, "",),
    ("aim", 108, "2",),

    ("reflective", 27, ""),
    ("reflective", 29, "2"),
    ("reflective", 31, "3"),

]

set_style()

# %%
# Load data
dfs = [load_log_file(comm_type, seed) for comm_type, seed, _ in FILES]

labels = {
    "aim": "AI Mother Tongue",
    "none": "No Communication",
    "continuous": "Continuous Communication",
    "discrete": "Discrete Communication",
    "reflective": "Reflective Communication"
}

df_plot = pd.concat([
    df.assign(Source=f"{labels[comm_type]} {label}") for df, (comm_type, seed, label) in zip(dfs, FILES)
])

# %%
def smoothen(df, key, factor):
    df = df.reset_index(drop=True).copy()
    df[f"{key}_smooth"] = (
        df.groupby('Source')[key]
        .transform(lambda x: x.rolling(window=factor, min_periods=1).mean())
    )
    return df

# %%
for key in df_plot.keys()[1:]:
    fig, ax = plt.subplots(figsize=(9, 5))
    try:
        df_plot = smoothen(df_plot, key, 10)
        sns.lineplot(data=df_plot, x="timestep", y=f"{key}_smooth", hue="Source", ax=ax)
    except Exception as e:
        print(f"{key} gave error: {e}")
        continue
    plt.title(f"{key} Comparison")
    plt.ylabel(f"{key}")
    plt.grid(True, alpha=0.2)
    plt.show()

# %%
def subsample(df, step=100):
    return df.groupby(["Source", "RunID"], group_keys=False).apply(
        lambda g: g.iloc[::step]
    ).reset_index(drop=True)

# %%
CI = [
    [
        ("aim", 2, ""),
        ("aim", 70, "2"),
        ("aim", 73, "3"),
    ],
    [
        ("none", 102, ""),
        ("none", 104, "2"),
        ("none", 106, "3"),
    ],
    [
        ("discrete", 46, ""),
        ("discrete", 48, "2"),
        ("discrete", 49, "3"),
    ],
    # [
    #     ("aim", 62, "with no reflection"),
    #     ("aim", 75, "with no reflection 2"),
    #     ("aim", 77, "with no reflection 3")
    # ],
    # [
    #     ("reflective", 7, ""),
    #     ("reflective", 9, "2"),
    #     ("reflective", 11, "3")
    # ]

    # [
    #     ("aim", 106, ""),
    #     ("aim", 108, "2"),
    #     ("aim", 110, "3"),
    # ],

    # [
    #     ("reflective", 27, ""),
    #     ("reflective", 29, "2"),
    #     ("reflective", 31, "3")
    # ]
]

dfs = [[load_log_file(comm_type, seed) for comm_type, seed, _  in PROTOCOL] for PROTOCOL in CI]

labels = {
    "aim": "AI Mother Tongue",
    "none": "No Communication",
    "continuous": "Continuous Communication",
    "discrete": "Discrete Communication",
    "reflective": "WhisperWise"
}

all_dfs = []
for i, group in enumerate(CI):
    if "no reflection" in group[0][2]:
        comm_type = "AI Mother Tongue without auxiliary losses"
    else:
        comm_type = labels[group[0][0]]
    for run_id, df in enumerate(dfs[i]):
        all_dfs.append(df.assign(Source=comm_type, RunID=run_id))

df_plot = pd.concat(all_dfs, ignore_index=True)

print("df_plot.shape =", df_plot.shape)
print("Unique timesteps:", df_plot["timestep"].nunique())
print("Unique Sources:", df_plot["Source"].nunique())

df_sampled = subsample(df_plot, 50)
df_smooth = smoothen(df_sampled, "reward_mean", 1)
df_smooth_perplexity = smoothen(df_sampled, "comm_perplexity", 1)

# %%
final_metrics = {}
for source in df_plot['Source'].unique():
    subset = df_plot[df_plot['Source'] == source]
    final_metrics[source] = subset['reward_mean'].tail(1000).tolist()

ai_mother_tongue = final_metrics['AI Mother Tongue']
no_comms = final_metrics['No Communication']
discrete_comms = final_metrics['Discrete Communication']
# whisperwise = final_metrics['WhisperWise']
# ai_mother_tongue_no_reflection = final_metrics['AI Mother Tongue without auxiliary losses']

# %%
# t_stat, p_val = stats.ttest_ind(ai_mother_tongue, whisperwise, equal_var=True)

# print(f"t-statistic: {t_stat}")
# print(f"p-value:  {p_val}")


# %%
f_stat, p_val = stats.f_oneway(
    ai_mother_tongue, 
    no_comms, 
    discrete_comms, 
    # ai_mother_tongue_no_reflection, 
    # whisperwise
)

print(f"F-statistic: {f_stat}")
print(f"p-value: {p_val}")


# %%

# data = pd.DataFrame({
#     'score': ai_mother_tongue + no_comms + discrete_comms + ai_mother_tongue_no_reflection + whisperwise,
#     'group': ['AI_Mother_Tongue'] * len(ai_mother_tongue) + 
#              ['No_Comms'] * len(no_comms) +
#              ['Discrete'] * len(discrete_comms) + 
#              ['AI_Mother_Tongue_without_auxiliary_losses'] * len(ai_mother_tongue_no_reflection) +
#              ['WhisperWise'] * len(whisperwise)
# })
data = pd.DataFrame({
    'score': ai_mother_tongue + no_comms + discrete_comms,
    'group': ['AI_Mother_Tongue'] * len(ai_mother_tongue) + 
             ['No_Comms'] * len(no_comms) +
             ['Discrete'] * len(discrete_comms)
})
# data = pd.DataFrame({
#     'score': ai_mother_tongue + ai_mother_tongue_no_reflection + whisperwise,
#     'group': ['AI_Mother_Tongue'] * len(ai_mother_tongue) + 
#              ['AI_Mother_Tongue_without_auxiliary_losses'] * len(ai_mother_tongue_no_reflection) +
#              ['WhisperWise'] * len(whisperwise)
# })


tukey = pairwise_tukeyhsd(endog=data['score'], groups=data['group'], alpha=0.05)
print(tukey)

# %%
plt.figure(figsize=(10, 6))

ax = sns.lineplot(
    data=df_smooth, 
    x="timestep", 
    y="reward_mean_smooth",
    hue="Source", 
    errorbar=("ci", 95)
)

plt.title("Reward Mean per Step over Training", fontsize=14)
plt.xlabel("Training Step", fontsize=12)
plt.ylabel("Reward Mean per Step", fontsize=12)
plt.legend(title="Communication Type")

# plt.savefig(FIGURES_DIR / "reward_mean_comparison.png", dpi=900, bbox_inches='tight', facecolor='white')
# plt.savefig(FIGURES_DIR / "reflection_comparison.png", dpi=900, bbox_inches='tight', facecolor='white')
# plt.savefig(FIGURES_DIR / "discrete_reflection.png", dpi=900, bbox_inches='tight', facecolor='white')
# plt.savefig(FIGURES_DIR / "challenge_performance.png", dpi=900, bbox_inches='tight', facecolor='white')

plt.tight_layout()
plt.show()

# %%
fig, ax = plt.subplots(figsize=(9, 5))

key = "comm_perplexity"

sns.lineplot(data=df_smooth_perplexity, x="timestep", y=f"{key}_smooth", hue="Source", ax=ax)

# plt.title(f"Communication Perplexity for AI Mother Tongue using SQ-VAE")
plt.title(f"Communication Perplexity for WhisperWise")
plt.ylabel("Communication Perplexity")
plt.xlabel("Training Step")
plt.grid(True, alpha=0.2)
# plt.savefig(FIGURES_DIR / "comm_perplexity_over_time.png", dpi=900, bbox_inches='tight', facecolor='white')
# plt.savefig(FIGURES_DIR / "comm_perplexity_with_reflection.png", dpi=900, bbox_inches='tight', facecolor='white')
# plt.savefig(FIGURES_DIR / "comm_perplexity-better.png", dpi=900, bbox_inches='tight', facecolor='white')
plt.show()


