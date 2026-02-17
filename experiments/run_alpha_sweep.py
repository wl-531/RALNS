"""Alpha 敏感性实验：固定物理系统，变化决策 κ"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from copy import deepcopy

from config import (KAPPA as KAPPA_BASE, N_PERIODS, N_RUNS, MC_SAMPLES,
                    DECISION_INTERVAL, T_MAX, PATIENCE, DESTROY_K, TYPE_MIX_GOOGLE)
from data.generator import generate_batch, generate_servers_with_target_rho
from solvers import DGSolver, RGSolver, RALNSSolver
from evaluation import compute_metrics, monte_carlo_verify, compute_next_backlog

ALPHA_VALUES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
N_TASKS = 100
M_SERVERS = 10
RHO = 0.90


def run_alpha_sweep(seed=42, verbose=True):
    """Alpha 敏感性实验"""
    results = []

    if verbose:
        print(f"===== Alpha Sweep Experiment =====")
        print(f"  KAPPA_BASE={KAPPA_BASE:.2f} (用于物理系统标定和评估)")
        print(f"  ALPHA_VALUES={ALPHA_VALUES}")
        print(f"  n_tasks={N_TASKS}, m_servers={M_SERVERS}, rho={RHO}")

    for run_idx in range(N_RUNS):
        run_seed = seed + run_idx * 1000
        np.random.seed(run_seed)

        # 固定本 run 的任务和物理系统（在 α 循环外！）
        tasks_list = [generate_batch(N_TASKS, type_mix=TYPE_MIX_GOOGLE)
                      for _ in range(N_PERIODS)]
        sample_tasks = tasks_list[0]
        servers_init_base = generate_servers_with_target_rho(
            M_SERVERS, sample_tasks, RHO, KAPPA_BASE, DECISION_INTERVAL
        )

        for alpha in ALPHA_VALUES:
            kappa_solver = np.sqrt(1/alpha - 1)  # 决策用

            # 创建 solver（用 kappa_solver）
            solvers = {
                'DG': DGSolver(),  # DG 不受 kappa 影响
                'RG': RGSolver(kappa=kappa_solver),
                'RA-LNS': RALNSSolver(kappa=kappa_solver, t_max=T_MAX,
                                      patience=PATIENCE, destroy_k=DESTROY_K),
            }

            for algo_name, solver in solvers.items():
                np.random.seed(run_seed)  # 重置随机种子
                servers = deepcopy(servers_init_base)  # 重置 backlog

                cvr_list, robust_util_list, makespan_list = [], [], []
                util_total_list, exp_makespan_list = [], []
                robust_load_ratio_list = []

                for t in range(N_PERIODS):
                    tasks = tasks_list[t]
                    assignment = solver.solve(tasks, servers)

                    # ★ 评估用 KAPPA_BASE（固定）
                    metrics = compute_metrics(assignment, tasks, servers, KAPPA_BASE)
                    system_cvr, _, _ = monte_carlo_verify(assignment, tasks, servers, MC_SAMPLES)

                    cvr_list.append(system_cvr)
                    robust_util_list.append(metrics['robust_util_total'])
                    makespan_list.append(metrics['O1'])
                    util_total_list.append(metrics['util_total'])
                    exp_makespan_list.append(metrics['exp_makespan'])
                    robust_load_ratio_list.append(metrics['robust_load_ratio'])

                    # 更新 backlog
                    next_backlog = compute_next_backlog(assignment, tasks, servers, DECISION_INTERVAL)
                    for j in range(M_SERVERS):
                        servers[j].L0 = next_backlog[j]

                results.append({
                    'alpha': alpha,
                    'kappa_solver': kappa_solver,
                    'kappa_eval': KAPPA_BASE,
                    'algorithm': algo_name,
                    'run': run_idx,
                    'cvr_mean': np.mean(cvr_list),
                    'robust_util_total_mean': np.mean(robust_util_list),
                    'makespan_mean': np.mean(makespan_list),
                    'util_total_mean': np.mean(util_total_list),
                    'exp_makespan_mean': np.mean(exp_makespan_list),
                    'robust_load_ratio_mean': np.mean(robust_load_ratio_list),
                })

        if verbose and run_idx == 0:
            print(f"\n  Run 0 preview:")
            for alpha in ALPHA_VALUES[:3]:  # 只显示前3个
                for algo in ['DG', 'RG', 'RA-LNS']:
                    row = [r for r in results if r['alpha'] == alpha
                           and r['algorithm'] == algo and r['run'] == 0][0]
                    print(f"    α={alpha:.2f}, {algo}: CVR={row['cvr_mean']:.4f}, "
                          f"robust_util={row['robust_util_total_mean']:.3f}")

    df = pd.DataFrame(results)
    df.to_csv('results_alpha_sweep.csv', index=False)

    if verbose:
        print("\n===== Summary by Alpha =====")
        summary = df.groupby(['alpha', 'algorithm']).agg({
            'cvr_mean': 'mean',
            'robust_util_total_mean': 'mean',
            'util_total_mean': 'mean',
        }).round(4)
        print(summary)

        print("\n===== RA-LNS Trend (α vs metrics) =====")
        ra_lns = df[df['algorithm'] == 'RA-LNS'].groupby('alpha').agg({
            'cvr_mean': 'mean',
            'robust_util_total_mean': 'mean',
        }).round(4)
        print(ra_lns)

        print("\n===== DG Sanity Check =====")
        dg = df[df['algorithm'] == 'DG'].groupby('alpha').agg({
            'cvr_mean': ['mean', 'std'],
            'robust_util_total_mean': ['mean', 'std'],
            'util_total_mean': ['mean', 'std'],
            'exp_makespan_mean': ['mean', 'std'],
        }).round(4)
        print(dg)

        # 计算 DG 跨 alpha 的波动
        dg_by_alpha = df[df['algorithm'] == 'DG'].groupby('alpha').mean(numeric_only=True)
        print("\n  DG 跨 alpha 的 std（应该很小）:")
        for col in ['cvr_mean', 'robust_util_total_mean', 'util_total_mean', 'exp_makespan_mean']:
            std_val = dg_by_alpha[col].std()
            status = "OK" if std_val < 0.01 else "WARN"
            print(f"    {col}: std={std_val:.6f} [{status}]")

    return df


if __name__ == '__main__':
    run_alpha_sweep(seed=42, verbose=True)
