"""绘图工具"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_cvr_comparison(df: pd.DataFrame, save_path: str = None):
    """绘制 CVR 对比柱状图

    Args:
        df: 包含 scenario, algorithm, cvr_mean 列的 DataFrame
        save_path: 保存路径（可选）
    """
    scenarios = df['scenario'].unique()
    algorithms = df['algorithm'].unique()

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(scenarios))
    width = 0.2
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(algorithms))

    for i, algo in enumerate(algorithms):
        algo_data = df[df['algorithm'] == algo].groupby('scenario')['cvr_mean'].mean()
        values = [algo_data.get(s, 0) for s in scenarios]
        ax.bar(x + offsets[i], values, width, label=algo)

    ax.set_xlabel('Scenario')
    ax.set_ylabel('CVR')
    ax.set_title('CVR Comparison Across Scenarios')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.legend()
    ax.axhline(y=0.15, color='r', linestyle='--', label='Target CVR')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_time_series(results: dict, metric: str = 'cvr', save_path: str = None):
    """绘制时间序列图

    Args:
        results: {algo_name: [values...]}
        metric: 指标名称
        save_path: 保存路径（可选）
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    for algo_name, values in results.items():
        ax.plot(values, label=algo_name, alpha=0.8)

    ax.set_xlabel('Period')
    ax.set_ylabel(metric.upper())
    ax.set_title(f'{metric.upper()} Over Time')
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_quality_time(df: pd.DataFrame, save_path: str = None):
    """绘制 Quality vs Time 曲线

    Args:
        df: 包含 t_max_ms, cvr_mean 列的 DataFrame
        save_path: 保存路径（可选）
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(df['t_max_ms'], df['cvr_mean'], 'o-', markersize=8)
    ax.set_xlabel('Time Budget (ms)')
    ax.set_ylabel('CVR')
    ax.set_title('Quality vs Time Budget')
    ax.set_xscale('log')
    ax.axhline(y=0.15, color='r', linestyle='--', label='Target CVR')
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_sensitivity(df: pd.DataFrame, param_name: str, save_path: str = None):
    """绘制参数敏感性曲线

    Args:
        df: 包含参数列和 cvr_mean 列的 DataFrame
        param_name: 参数名称
        save_path: 保存路径（可选）
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(df[param_name], df['cvr_mean'], 'o-', markersize=8)
    ax.set_xlabel(param_name)
    ax.set_ylabel('CVR')
    ax.set_title(f'Sensitivity: {param_name}')
    ax.axhline(y=0.15, color='r', linestyle='--', label='Target CVR')
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
