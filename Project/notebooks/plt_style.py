import seaborn as sns
import matplotlib.pyplot as plt

def set_style():
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        "font.family": "serif",
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "lines.linewidth": 1.5,
        "figure.dpi": 300
    })

def save_fig(fig, name):
    fig.savefig(f"../dissertation/figures/{name}.pdf", bbox_inches='tight')