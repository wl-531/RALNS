"""论文图表生成脚本"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42  # 嵌入字体
matplotlib.rcParams['ps.fonttype'] = 42

# 论文风格设置
plt.style.use('seaborn-v0_8-whitegrid')
FIGSIZE_SINGLE = (4, 3)      # 单栏图
FIGSIZE_DOUBLE = (8, 3)      # 双栏图
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # 蓝橙绿红
MARKERS = ['o', 's', '^', 'D']
ALPHA_TARGET = 0.15  # CVR 目标线

import os
os.makedirs('figures', exist_ok=True)


def plot_main_comparison(csv_path='results_main.csv'):
    """图1: 主实验 - 4算法 CVR/U_max 对比柱状图"""
    df = pd.read_csv(csv_path)

    # 按算法聚合
    summary = df.groupby('algorithm').agg({
        'cvr_mean': ['mean', 'std'],
        'per_server_vr_mean': ['mean', 'std'],
        'U_max_mean': ['mean', 'std'],
        'time_mean_ms': 'mean',
        'feasible_rate': 'mean'
    }).round(4)

    algorithms = ['DG', 'RG', 'Micro', 'RA-LNS']

    # 确保顺序
    cvr_means = [summary.loc[a, ('cvr_mean', 'mean')] for a in algorithms]
    cvr_stds = [summary.loc[a, ('cvr_mean', 'std')] for a in algorithms]
    umax_means = [summary.loc[a, ('U_max_mean', 'mean')] for a in algorithms]
    umax_stds = [summary.loc[a, ('U_max_mean', 'std')] for a in algorithms]

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_DOUBLE)

    x = np.arange(len(algorithms))
    width = 0.6

    # (a) System CVR
    ax = axes[0]
    bars = ax.bar(x, cvr_means, width, yerr=cvr_stds, capsize=3,
                  color=COLORS[:4], edgecolor='black', linewidth=0.5)
    ax.axhline(y=ALPHA_TARGET, color='red', linestyle='--', linewidth=1.5, label=f'Target α={ALPHA_TARGET}')
    ax.set_xlabel('Algorithm', fontsize=10)
    ax.set_ylabel('System CVR', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, fontsize=9)
    ax.set_ylim(0, max(cvr_means) * 1.3)
    ax.legend(fontsize=8, loc='upper right')
    ax.set_title('(a) Constraint Violation Rate', fontsize=10)

    # (b) U_max
    ax = axes[1]
    bars = ax.bar(x, umax_means, width, yerr=umax_stds, capsize=3,
                  color=COLORS[:4], edgecolor='black', linewidth=0.5)
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, label='Capacity Limit')
    ax.axhline(y=0.9, color='gray', linestyle=':', linewidth=1, label='Target ρ=0.90')
    ax.set_xlabel('Algorithm', fontsize=10)
    ax.set_ylabel('Max Robust Utilization (U_max)', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, fontsize=9)
    ax.set_ylim(0.7, 1.2)
    ax.legend(fontsize=8, loc='upper right')
    ax.set_title('(b) Robust Utilization', fontsize=10)

    plt.tight_layout()
    plt.savefig('figures/fig1_main_comparison.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('figures/fig1_main_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Generated: figures/fig1_main_comparison.pdf")

    # 打印 LaTeX 表格
    print("\n===== LaTeX Table =====")
    print("Algorithm & System CVR & Per-Server VR & U_max & Time (ms) & Feasible Rate \\\\")
    print("\\hline")
    for a in algorithms:
        cvr = summary.loc[a, ('cvr_mean', 'mean')]
        psvr = summary.loc[a, ('per_server_vr_mean', 'mean')]
        umax = summary.loc[a, ('U_max_mean', 'mean')]
        time_ms = summary.loc[a, ('time_mean_ms', 'mean')]
        feas = summary.loc[a, ('feasible_rate', 'mean')]
        print(f"{a} & {cvr:.4f} & {psvr:.4f} & {umax:.3f} & {time_ms:.2f} & {feas:.2f} \\\\")


def plot_ablation(csv_path='results_ablation.csv'):
    """图2: 消融实验 - RG vs Micro-LNS vs RA-LNS"""
    df = pd.read_csv(csv_path)

    summary = df.groupby('algorithm')['cvr_mean'].agg(['mean', 'std'])
    algorithms = ['RG', 'Micro-LNS', 'RA-LNS']

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    x = np.arange(len(algorithms))
    width = 0.5
    means = [summary.loc[a, 'mean'] for a in algorithms]
    stds = [summary.loc[a, 'std'] for a in algorithms]

    bars = ax.bar(x, means, width, yerr=stds, capsize=4,
                  color=[COLORS[1], COLORS[2], COLORS[3]],
                  edgecolor='black', linewidth=0.5)

    ax.axhline(y=ALPHA_TARGET, color='red', linestyle='--', linewidth=1.5, label=f'Target α={ALPHA_TARGET}')
    ax.set_xlabel('Algorithm Variant', fontsize=10)
    ax.set_ylabel('System CVR', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(['RG\n(Construction)', 'Micro-LNS\n(+Stage-1A)', 'RA-LNS\n(+Stage-1B)'], fontsize=8)
    ax.set_ylim(0, max(means) * 1.4)
    ax.legend(fontsize=8)
    ax.set_title('Ablation Study', fontsize=10)

    # 添加数值标签
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.005, f'{m:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig('figures/fig2_ablation.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('figures/fig2_ablation.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Generated: figures/fig2_ablation.pdf")


def plot_rho_sensitivity(csv_path='results_rho_sensitivity.csv'):
    """图3: rho 敏感性 - Safety-Efficiency Trade-off"""
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    algorithms = ['DG', 'RG', 'RA-LNS']

    for i, algo in enumerate(algorithms):
        algo_df = df[df['algorithm'] == algo]
        grouped = algo_df.groupby('rho')['cvr_mean'].agg(['mean', 'std'])

        rhos = grouped.index.values
        means = grouped['mean'].values
        stds = grouped['std'].values

        ax.errorbar(rhos, means, yerr=stds, marker=MARKERS[i],
                    color=COLORS[i], label=algo, capsize=3, linewidth=1.5, markersize=6)

    ax.axhline(y=ALPHA_TARGET, color='red', linestyle='--', linewidth=1, alpha=0.7, label=f'Target α={ALPHA_TARGET}')
    ax.set_xlabel('Target Robust Utilization (ρ)', fontsize=10)
    ax.set_ylabel('System CVR', fontsize=10)
    ax.set_xlim(0.78, 1.0)
    ax.set_ylim(0, 0.5)
    ax.legend(fontsize=8, loc='upper left')
    ax.set_title('Safety-Efficiency Trade-off', fontsize=10)

    plt.tight_layout()
    plt.savefig('figures/fig3_rho_sensitivity.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('figures/fig3_rho_sensitivity.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Generated: figures/fig3_rho_sensitivity.pdf")


def plot_kappa_sensitivity(csv_path='results_sensitivity_kappa.csv'):
    """图4: kappa 敏感性"""
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_DOUBLE)

    kappas = df['kappa'].values
    cvr = df['cvr_mean'].values
    umax = df['U_max_mean'].values

    # (a) CVR vs kappa
    ax = axes[0]
    ax.plot(kappas, cvr, 'o-', color=COLORS[3], markersize=8, linewidth=2)
    ax.axhline(y=ALPHA_TARGET, color='red', linestyle='--', linewidth=1, label=f'Target α={ALPHA_TARGET}')
    ax.axvline(x=2.38, color='gray', linestyle=':', linewidth=1, label='κ*=2.38')
    ax.set_xlabel('Risk Coefficient (κ)', fontsize=10)
    ax.set_ylabel('System CVR', fontsize=10)
    ax.legend(fontsize=8)
    ax.set_title('(a) CVR vs κ', fontsize=10)

    # (b) U_max vs kappa
    ax = axes[1]
    ax.plot(kappas, umax, 's-', color=COLORS[0], markersize=8, linewidth=2)
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1, label='Capacity Limit')
    ax.axvline(x=2.38, color='gray', linestyle=':', linewidth=1, label='κ*=2.38')
    ax.set_xlabel('Risk Coefficient (κ)', fontsize=10)
    ax.set_ylabel('Max Robust Utilization (U_max)', fontsize=10)
    ax.legend(fontsize=8)
    ax.set_title('(b) U_max vs κ', fontsize=10)

    plt.tight_layout()
    plt.savefig('figures/fig4_kappa_sensitivity.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('figures/fig4_kappa_sensitivity.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Generated: figures/fig4_kappa_sensitivity.pdf")


def plot_quality_time(csv_path='results_quality_time.csv'):
    """图5: Anytime 性能 - Quality vs Time"""
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    t_max = df['t_max_ms'].values
    cvr = df['cvr_mean'].values

    ax.semilogx(t_max, cvr, 'o-', color=COLORS[3], markersize=8, linewidth=2)
    ax.axhline(y=ALPHA_TARGET, color='red', linestyle='--', linewidth=1, label=f'Target α={ALPHA_TARGET}')

    ax.set_xlabel('Time Budget (ms)', fontsize=10)
    ax.set_ylabel('System CVR', fontsize=10)
    ax.set_xlim(0.8, 150)
    ax.legend(fontsize=8)
    ax.set_title('Anytime Performance', fontsize=10)

    # 标注关键点
    ax.annotate(f'{cvr[0]:.3f}', (t_max[0], cvr[0]), textcoords="offset points",
                xytext=(5, 10), fontsize=8)
    ax.annotate(f'{cvr[-1]:.3f}', (t_max[-1], cvr[-1]), textcoords="offset points",
                xytext=(-20, 10), fontsize=8)

    plt.tight_layout()
    plt.savefig('figures/fig5_quality_time.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('figures/fig5_quality_time.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Generated: figures/fig5_quality_time.pdf")


def plot_time_series_cvr(csv_path='results_main.csv', run_idx=0):
    """图6: 单次运行的 CVR 时间序列（补充材料）"""
    # 需要从原始数据重新生成，这里用聚合数据近似
    print("Time series plot requires raw per-period data (not in current CSV)")
    print("  To generate, modify run_main.py to save per-period results")


def plot_boxplot_comparison(csv_path='results_main.csv'):
    """图7: 箱线图对比（展示分布）"""
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    algorithms = ['DG', 'RG', 'Micro', 'RA-LNS']
    data = [df[df['algorithm'] == a]['cvr_mean'].values for a in algorithms]

    bp = ax.boxplot(data, labels=algorithms, patch_artist=True)

    for patch, color in zip(bp['boxes'], COLORS[:4]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axhline(y=ALPHA_TARGET, color='red', linestyle='--', linewidth=1, label=f'Target α={ALPHA_TARGET}')
    ax.set_xlabel('Algorithm', fontsize=10)
    ax.set_ylabel('System CVR', fontsize=10)
    ax.legend(fontsize=8)
    ax.set_title('CVR Distribution Across Runs', fontsize=10)

    plt.tight_layout()
    plt.savefig('figures/fig7_boxplot.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('figures/fig7_boxplot.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Generated: figures/fig7_boxplot.pdf")


def generate_all_figures():
    """生成所有图表"""
    print("=" * 50)
    print("Generating Paper Figures")
    print("=" * 50)

    # 检查文件存在性
    import os
    files = {
        'main': 'results_main.csv',
        'ablation': 'results_ablation.csv',
        'rho': 'results_rho_sensitivity.csv',
        'kappa': 'results_sensitivity_kappa.csv',
        'quality_time': 'results_quality_time.csv'
    }

    for name, path in files.items():
        if os.path.exists(path):
            print(f"Found: {path}")
        else:
            print(f"Missing: {path}")

    print("-" * 50)

    # 生成图表
    if os.path.exists(files['main']):
        plot_main_comparison(files['main'])
        plot_boxplot_comparison(files['main'])

    if os.path.exists(files['ablation']):
        plot_ablation(files['ablation'])

    if os.path.exists(files['rho']):
        plot_rho_sensitivity(files['rho'])

    if os.path.exists(files['kappa']):
        plot_kappa_sensitivity(files['kappa'])

    if os.path.exists(files['quality_time']):
        plot_quality_time(files['quality_time'])

    print("=" * 50)
    print("Done! Check figures/ directory")
    print("=" * 50)


if __name__ == '__main__':
    generate_all_figures()
