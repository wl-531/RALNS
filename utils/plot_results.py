"""论文图表生成脚本 - 3 张核心图表"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

plt.style.use("seaborn-v0_8-whitegrid")
FIGSIZE_SINGLE = (5, 3.5)
FIGSIZE_WIDE = (6, 4)
ALPHA_TARGET = 0.15

COLORS_MAIN = {
    "DG": "#1f77b4",
    "RG": "#ff7f0e",
    "Std-LNS": "#2ca02c",
    "RA-LNS": "#d62728",
}

COLORS_ABLATION = ["#c6dbef", "#6baed6", "#2171b5", "#d62728"]
MARKERS = ["o", "s", "^", "D"]

os.makedirs("figures", exist_ok=True)


def plot_main_comparison(csv_path="results_main.csv"):
    """Figure 1: 主实验 - 4 算法 CVR 对比柱状图"""
    df = pd.read_csv(csv_path)

    summary = df.groupby("algorithm").agg({
        "cvr_mean": ["mean", "std"],
        "O1_mean": ["mean", "std"],
        "time_mean_ms": "mean",
    }).round(4)

    algorithms = ["DG", "RG", "Std-LNS", "RA-LNS"]

    cvr_means = [summary.loc[a, ("cvr_mean", "mean")] for a in algorithms]
    cvr_stds = [summary.loc[a, ("cvr_mean", "std")] for a in algorithms]
    colors = [COLORS_MAIN[a] for a in algorithms]

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    x = np.arange(len(algorithms))
    width = 0.6

    bars = ax.bar(x, cvr_means, width, yerr=cvr_stds, capsize=4,
                  color=colors, edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Algorithm", fontsize=11)
    ax.set_ylabel("System CVR", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, fontsize=10)
    ax.set_ylim(0, max(cvr_means) * 1.4)
    ax.legend(fontsize=9, loc="upper right")

    for i, (m, s) in enumerate(zip(cvr_means, cvr_stds)):
        ax.text(i, m + s + 0.003, f"{m:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig("figures/fig1_cvr_comparison.pdf", bbox_inches="tight", dpi=300)
    plt.savefig("figures/fig1_cvr_comparison.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("Generated: figures/fig1_cvr_comparison.pdf")

    print("\n===== Table 1: Main Results =====")
    for a in algorithms:
        cvr = summary.loc[a, ("cvr_mean", "mean")]
        o1 = summary.loc[a, ("O1_mean", "mean")]
        time_ms = summary.loc[a, ("time_mean_ms", "mean")]
        print(f"{a}: CVR={cvr:.4f}, Makespan={o1:.1f}, Time={time_ms:.2f}ms")


def plot_backlog_evolution(csv_path="results_backlog.csv"):
    """Figure 2: System Backlog 演化折线图"""
    df = pd.read_csv(csv_path, index_col=0)

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    algorithms = ["DG", "RG", "Std-LNS", "RA-LNS"]
    periods = df.index.values

    for i, algo in enumerate(algorithms):
        ax.plot(periods, df[algo].values, marker=MARKERS[i], markevery=10,
                color=COLORS_MAIN[algo], label=algo, linewidth=1.5, markersize=5)

    ax.set_xlabel("Decision Period (t)", fontsize=11)
    ax.set_ylabel("System Backlog", fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)

    plt.tight_layout()
    plt.savefig("figures/fig2_backlog_evolution.pdf", bbox_inches="tight", dpi=300)
    plt.savefig("figures/fig2_backlog_evolution.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("Generated: figures/fig2_backlog_evolution.pdf")


def plot_ablation(csv_path="results_ablation.csv"):
    """Figure 3: 消融实验柱状图"""
    df = pd.read_csv(csv_path)

    summary = df.groupby("algorithm")["cvr_mean"].agg(["mean", "std"])
    variants = ["Construction-Only", "Micro-Only", "Random-Destroy", "RA-LNS"]

    means = [summary.loc[v, "mean"] for v in variants]
    stds = [summary.loc[v, "std"] for v in variants]

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    x = np.arange(len(variants))
    width = 0.6

    bars = ax.bar(x, means, width, yerr=stds, capsize=4,
                  color=COLORS_ABLATION, edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Algorithm Variant", fontsize=11)
    ax.set_ylabel("System CVR", fontsize=11)
    ax.set_xticks(x)
    labels = ["Construct\n-Only", "Micro\n-Only", "Random\n-Destroy", "RA-LNS\n(Full)"]
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, max(means) * 1.4)
    ax.legend(fontsize=9)

    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.003, f"{m:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig("figures/fig3_ablation.pdf", bbox_inches="tight", dpi=300)
    plt.savefig("figures/fig3_ablation.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("Generated: figures/fig3_ablation.pdf")


def plot_boxplot_comparison(csv_path="results_main.csv"):
    """(可选) CVR 分布箱线图"""
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    algorithms = ["DG", "RG", "Std-LNS", "RA-LNS"]
    data = [df[df["algorithm"] == a]["cvr_mean"].values for a in algorithms]
    colors = [COLORS_MAIN[a] for a in algorithms]

    bp = ax.boxplot(data, tick_labels=algorithms, patch_artist=True)

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axhline(y=ALPHA_TARGET, color="red", linestyle="--", linewidth=1,
               label=f"Target CVR = {ALPHA_TARGET}")
    ax.set_xlabel("Algorithm", fontsize=11)
    ax.set_ylabel("System CVR", fontsize=11)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("figures/fig_boxplot.pdf", bbox_inches="tight", dpi=300)
    plt.savefig("figures/fig_boxplot.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("Generated: figures/fig_boxplot.pdf")


def generate_all_figures():
    """生成所有图表"""
    print("=" * 50)
    print("Generating Paper Figures")
    print("=" * 50)

    files = {
        "main": "results_main.csv",
        "backlog": "results_backlog.csv",
        "ablation": "results_ablation.csv",
    }

    for name, path in files.items():
        if os.path.exists(path):
            print(f"Found: {path}")
        else:
            print(f"Missing: {path}")

    print("-" * 50)

    if os.path.exists(files["main"]):
        plot_main_comparison(files["main"])
        plot_boxplot_comparison(files["main"])

    if os.path.exists(files["backlog"]):
        plot_backlog_evolution(files["backlog"])

    if os.path.exists(files["ablation"]):
        plot_ablation(files["ablation"])

    print("=" * 50)
    print("Done! Check figures/ directory")
    print("=" * 50)


if __name__ == "__main__":
    generate_all_figures()
