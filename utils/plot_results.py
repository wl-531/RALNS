"""论文图表生成脚本 - 核心图表 + 扩展图表"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

from config import T_MAX
T_MAX_MS = T_MAX * 1000 if T_MAX < 10 else T_MAX

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

plt.style.use("seaborn-v0_8-whitegrid")
FIGSIZE_SINGLE = (5, 3.5)
FIGSIZE_WIDE = (6, 4)
ALPHA_TARGET = 0.15

COLORS_MAIN = {
    "DG": "#1f77b4",
    "RG": "#ff7f0e",
    "EPD-FF": "#9467bd",
    "Std-LNS": "#2ca02c",
    "RA-LNS": "#d62728",
}

COLORS_ABLATION = ["#c6dbef", "#6baed6", "#2171b5", "#d62728"]
MARKERS = ["o", "s", "D", "p", "h"]  # 圆、方、菱、五边形、六边形

os.makedirs("figures", exist_ok=True)


def plot_main_comparison(csv_path="results_main.csv"):
    """Figure 1: 主实验 - 4 算法 CVR 对比柱状图"""
    df = pd.read_csv(csv_path)

    summary = df.groupby("algorithm").agg({
        "cvr_mean": ["mean", "std"],
        "O1_mean": ["mean", "std"],
        "time_mean_ms": "mean",
    }).round(4)

    algorithms = ["DG", "RG", "EPD-FF", "Std-LNS", "RA-LNS"]

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
    ax.set_ylim(0, max(cvr_means) * 1.4 + 0.025)
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

    algorithms = ["DG", "RG", "EPD-FF", "Std-LNS", "RA-LNS"]
    periods = df.index.values

    for i, algo in enumerate(algorithms):
        ax.plot(periods, df[algo].values,
                color=COLORS_MAIN[algo], label=algo, linewidth=1.5)

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
    ax.set_ylim(0, max(means) * 1.4 + 0.04)
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

    algorithms = ["DG", "RG", "EPD-FF", "Std-LNS", "RA-LNS"]
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


def print_efficiency_table(csv_path="results_main.csv"):
    """Table 2: Efficiency Metrics（生成 LaTeX 表格）"""
    df = pd.read_csv(csv_path)

    algorithms = ["DG", "RG", "EPD-FF", "Std-LNS", "RA-LNS"]

    agg_dict = {"cvr_mean": "mean", "O1_mean": "mean", "time_mean_ms": "mean"}
    # 兼容：有些列可能不存在
    for col in ["exp_makespan_mean", "robust_util_total_mean", "U_max_mean"]:
        if col in df.columns:
            agg_dict[col] = "mean"

    summary = df.groupby("algorithm").agg(agg_dict).round(4)
    dg_o1 = summary.loc["DG", "O1_mean"]

    has_exp_ms = "exp_makespan_mean" in summary.columns
    has_rob_util = "robust_util_total_mean" in summary.columns
    has_umax = "U_max_mean" in summary.columns

    print("\n" + "="*70)
    print("Table 2: Efficiency Metrics")
    print("="*70)
    header = f"{'Algorithm':<10} {'CVR':>8} {'O1':>10} {'O1/DG':>8}"
    if has_exp_ms:
        header += f" {'ExpMS':>10}"
    if has_rob_util:
        header += f" {'RobUtil':>8}"
    elif has_umax:
        header += f" {'U_max':>8}"
    header += f" {'Time(ms)':>10}"
    print(header)
    print("-"*70)
    for a in algorithms:
        row = summary.loc[a]
        o1_ratio = row['O1_mean'] / dg_o1
        line = f"{a:<10} {row['cvr_mean']:>8.3f} {row['O1_mean']:>10.1f} {o1_ratio:>8.2f}"
        if has_exp_ms:
            line += f" {row['exp_makespan_mean']:>10.1f}"
        if has_rob_util:
            line += f" {row['robust_util_total_mean']:>8.3f}"
        elif has_umax:
            line += f" {row['U_max_mean']:>8.3f}"
        line += f" {row['time_mean_ms']:>10.2f}"
        print(line)
    print("="*70)


def print_runtime_table(csv_path="results_main.csv"):
    """Table 3: Runtime Distribution"""
    df = pd.read_csv(csv_path)

    algorithms = ["DG", "RG", "EPD-FF", "Std-LNS", "RA-LNS"]

    # 只聚合存在的列
    time_cols = ["time_mean_ms", "time_std_ms", "time_p50_ms", "time_p99_ms",
                 "time_max_ms", "timeout_rate"]
    agg_dict = {c: "mean" for c in time_cols if c in df.columns}

    summary = df.groupby("algorithm").agg(agg_dict).round(3)

    print("\n" + "="*85)
    print(f"Table 3: Runtime Distribution (Budget = {T_MAX_MS:.0f} ms)")
    print("="*85)

    # 动态构建表头和行
    available = [c for c in time_cols if c in summary.columns]
    col_labels = {"time_mean_ms": "Mean", "time_std_ms": "Std", "time_p50_ms": "p50",
                  "time_p99_ms": "p99", "time_max_ms": "Max", "timeout_rate": "Timeout"}
    header = f"{'Algorithm':<10}" + "".join(f" {col_labels[c]:>8}" for c in available)
    print(header)
    print("-"*85)
    for a in algorithms:
        row = summary.loc[a]
        line = f"{a:<10}"
        for c in available:
            if c == "timeout_rate":
                line += f" {row[c]*100:>7.1f}%"
            else:
                line += f" {row[c]:>8.2f}"
        print(line)
    print("="*85)


def plot_alpha_pareto(csv_path="results_alpha_sweep.csv"):
    """Figure 4: α-sweep Pareto 曲线"""
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(5, 4))

    # RA-LNS 曲线（随 α 变化）
    ralns = df[df["algorithm"] == "RA-LNS"].groupby("alpha")[["cvr_mean", "robust_load_ratio_mean"]].mean()
    ax.plot(ralns["robust_load_ratio_mean"], ralns["cvr_mean"], 'o-',
            color=COLORS_MAIN.get("RA-LNS", "#2ecc71"), markersize=8, linewidth=2, label="RA-LNS")

    # 标注 α 值
    for alpha, row in ralns.iterrows():
        ax.annotate(f"α={alpha}", (row["robust_load_ratio_mean"], row["cvr_mean"]),
                   textcoords="offset points", xytext=(5, 3), fontsize=8)

    # DG 参考点（静态，不受 α 影响）
    dg = df[df["algorithm"] == "DG"]
    dg_util = dg["robust_load_ratio_mean"].mean()
    dg_cvr = dg["cvr_mean"].mean()
    ax.scatter([dg_util], [dg_cvr], marker='o', s=60, color='gray',
               zorder=5, label="DG (baseline)")

    ax.set_xlabel("Robust Load Ratio", fontsize=11)
    ax.set_ylabel("System CVR", fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/fig4_alpha_tradeoff.pdf", bbox_inches="tight", dpi=300)
    plt.savefig("figures/fig4_alpha_tradeoff.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("Generated: figures/fig4_alpha_tradeoff.pdf")


def plot_scalability(csv_path="results_scalability.csv"):
    """Figure 5: 可扩展性实验"""
    df = pd.read_csv(csv_path)

    algorithms = ["DG", "RG", "EPD-FF", "Std-LNS", "RA-LNS"]

    # fig5(a): CVR vs M
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for algo in algorithms:
        data = df[df["algorithm"] == algo].groupby("M")["cvr_mean"].mean()
        ax.plot(data.index, data.values, 'o-', label=algo,
                color=COLORS_MAIN.get(algo, "gray"), markersize=6, linewidth=1.5)

    ax.set_xlabel("Number of Servers (M)", fontsize=11)
    ax.set_ylabel("System CVR", fontsize=11)
    ax.set_title("(a) CVR vs. Cluster Size", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xticks([5, 10, 20, 50])
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + 0.2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/fig5a_scalability_cvr.pdf", bbox_inches="tight", dpi=300)
    plt.savefig("figures/fig5a_scalability_cvr.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("Generated: figures/fig5a_scalability_cvr.pdf")

    # fig5(b): Runtime vs M
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for algo in algorithms:
        data = df[df["algorithm"] == algo].groupby("M")["time_mean_ms"].mean()
        ax.plot(data.index, data.values, 'o-', label=algo,
                color=COLORS_MAIN.get(algo, "gray"), markersize=6, linewidth=1.5)

    ax.set_xlabel("Number of Servers (M)", fontsize=11)
    ax.set_ylabel("Mean Runtime (ms)", fontsize=11)
    ax.set_title("(b) Runtime vs. Cluster Size", fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xticks([5, 10, 20, 50])
    ax.set_ylim(0, 25)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/fig5b_scalability_runtime.pdf", bbox_inches="tight", dpi=300)
    plt.savefig("figures/fig5b_scalability_runtime.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("Generated: figures/fig5b_scalability_runtime.pdf")

    print("\n===== Scalability Summary =====")
    print("\nCVR by (M, Algorithm):")
    print(df.groupby(['M', 'algorithm'])['cvr_mean'].mean().unstack().round(4))
    print("\nRuntime (ms) by (M, Algorithm):")
    print(df.groupby(['M', 'algorithm'])['time_mean_ms'].mean().unstack().round(2))
    print("\nTimeout Rate by (M, Algorithm):")
    print(df.groupby(['M', 'algorithm'])['timeout_rate'].mean().unstack().round(4))


def generate_all_figures():
    """生成所有图表"""
    print("=" * 60)
    print("Generating Paper Figures (Tier-0 + Tier-1)")
    print("=" * 60)

    files = {
        "main": "results_main.csv",
        "backlog": "results_backlog.csv",
        "ablation": "results_ablation.csv",
        "alpha_sweep": "results_alpha_sweep.csv",
        "scalability": "results_scalability.csv",
    }

    for name, path in files.items():
        status = "OK" if os.path.exists(path) else "MISSING"
        print(f"  [{status}] {path}")

    print("-" * 60)

    # 现有图表
    if os.path.exists(files["main"]):
        plot_main_comparison(files["main"])
        plot_boxplot_comparison(files["main"])
        print_efficiency_table(files["main"])
        print_runtime_table(files["main"])

    if os.path.exists(files["backlog"]):
        plot_backlog_evolution(files["backlog"])

    if os.path.exists(files["ablation"]):
        plot_ablation(files["ablation"])

    # 新增图表
    if os.path.exists(files["alpha_sweep"]):
        plot_alpha_pareto(files["alpha_sweep"])

    if os.path.exists(files["scalability"]):
        plot_scalability(files["scalability"])

    print("=" * 60)
    print("Done! Check figures/ directory")
    print("=" * 60)


if __name__ == "__main__":
    generate_all_figures()
