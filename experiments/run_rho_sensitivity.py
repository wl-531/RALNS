"""rho 敏感性实验：展示 safety-efficiency trade-off"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from copy import deepcopy

from config import (KAPPA, N_PERIODS, N_RUNS, MC_SAMPLES, DECISION_INTERVAL,
                    T_MAX, PATIENCE, DESTROY_K, TYPE_MIX_GOOGLE, RHO_VALUES)
from data.generator import generate_batch, generate_servers_with_target_rho
from solvers import DGSolver, RGSolver, RALNSSolver
from evaluation import compute_metrics, monte_carlo_verify, compute_next_backlog


def run_rho_sensitivity(seed=42):
    """rho 敏感性实验"""
    np.random.seed(seed)

    n_tasks = 100
    m_servers = 10
    type_mix = TYPE_MIX_GOOGLE

    algorithms = {
        'DG': DGSolver(),
        'RG': RGSolver(kappa=KAPPA),
        'RA-LNS': RALNSSolver(kappa=KAPPA, t_max=T_MAX, patience=PATIENCE, destroy_k=DESTROY_K),
    }

    results = []

    for rho in RHO_VALUES:
        print(f"\n===== rho = {rho} =====")

        for run_idx in range(N_RUNS):
            run_seed = seed + run_idx * 1000
            np.random.seed(run_seed)

            tasks_list = [generate_batch(n_tasks, type_mix=type_mix) for _ in range(N_PERIODS)]
            sample_tasks = tasks_list[0]
            servers_init = generate_servers_with_target_rho(m_servers, sample_tasks, rho, KAPPA, DECISION_INTERVAL)

            for algo_name, solver in algorithms.items():
                np.random.seed(run_seed)
                servers = deepcopy(servers_init)

                cvr_list, excess_list, u_max_list = [], [], []

                for t in range(N_PERIODS):
                    tasks = tasks_list[t]
                    assignment = solver.solve(tasks, servers)

                    metrics = compute_metrics(assignment, tasks, servers, KAPPA)
                    system_cvr, _, avg_excess = monte_carlo_verify(assignment, tasks, servers, MC_SAMPLES)

                    cvr_list.append(system_cvr)
                    excess_list.append(avg_excess)
                    u_max_list.append(metrics['U_max'])

                    next_backlog = compute_next_backlog(assignment, tasks, servers, DECISION_INTERVAL)
                    for j in range(m_servers):
                        servers[j].L0 = next_backlog[j]

                results.append({
                    'rho': rho,
                    'algorithm': algo_name,
                    'run': run_idx,
                    'cvr_mean': np.mean(cvr_list),
                    'excess_mean': np.mean(excess_list),
                    'U_max_mean': np.mean(u_max_list),
                })

        # 打印当前 rho 的汇总
        df_rho = pd.DataFrame([r for r in results if r['rho'] == rho])
        summary = df_rho.groupby('algorithm')['cvr_mean'].agg(['mean', 'std']).round(4)
        print(summary)

    df = pd.DataFrame(results)
    df.to_csv('results_rho_sensitivity.csv', index=False)

    # 最终汇总
    print("\n===== Final Summary =====")
    final_summary = df.groupby(['rho', 'algorithm'])['cvr_mean'].mean().unstack().round(4)
    print(final_summary)

    return df


if __name__ == '__main__':
    run_rho_sensitivity(seed=42)
