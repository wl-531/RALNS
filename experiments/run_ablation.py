"""消融实验：RA-LNS vs Micro-LNS vs RG"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from copy import deepcopy

from config import KAPPA, N_PERIODS, N_RUNS, MC_SAMPLES, DECISION_INTERVAL, T_MAX, PATIENCE, DESTROY_K
from data.generator import generate_batch, generate_servers_with_target_rho
from solvers import RGSolver, RALNSSolver, MicroLNSSolver
from evaluation import compute_metrics, monte_carlo_verify, compute_next_backlog


def run_ablation_experiment(seed=42):
    """消融实验"""
    np.random.seed(seed)

    algorithms = {
        'RG': RGSolver(kappa=KAPPA),
        'Micro-LNS': MicroLNSSolver(kappa=KAPPA, t_max=T_MAX, patience=PATIENCE),
        'RA-LNS': RALNSSolver(kappa=KAPPA, t_max=T_MAX, patience=PATIENCE, destroy_k=DESTROY_K),
    }

    # 使用 stress 场景
    n_tasks = 120
    m_servers = 10
    rho = 0.92

    results = []

    for run_idx in range(N_RUNS):
        run_seed = seed + run_idx * 1000
        np.random.seed(run_seed)

        tasks_list = [generate_batch(n_tasks, type_mix=[0.10, 0.80, 0.10]) for _ in range(N_PERIODS)]
        total_mu = sum(sum(t.mu for t in tasks) for tasks in tasks_list) / N_PERIODS
        servers_init = generate_servers_with_target_rho(m_servers, total_mu, rho, DECISION_INTERVAL)

        for algo_name, solver in algorithms.items():
            np.random.seed(run_seed)
            servers = deepcopy(servers_init)

            cvr_list = []
            for t in range(N_PERIODS):
                tasks = tasks_list[t]
                assignment = solver.solve(tasks, servers)
                system_cvr, _, _ = monte_carlo_verify(assignment, tasks, servers, MC_SAMPLES)
                cvr_list.append(system_cvr)

                next_backlog = compute_next_backlog(assignment, tasks, servers, DECISION_INTERVAL)
                for j in range(m_servers):
                    servers[j].L0 = next_backlog[j]

            results.append({
                'algorithm': algo_name,
                'run': run_idx,
                'cvr_mean': np.mean(cvr_list),
            })

        if run_idx == 0:
            for algo_name in algorithms.keys():
                row = [r for r in results if r['algorithm'] == algo_name and r['run'] == 0][0]
                print(f"{algo_name}: CVR={row['cvr_mean']:.4f}")

    df = pd.DataFrame(results)

    # 汇总
    summary = df.groupby('algorithm')['cvr_mean'].agg(['mean', 'std']).round(4)
    print("\n===== Ablation Summary =====")
    print(summary)

    df.to_csv('results_ablation.csv', index=False)
    return df


if __name__ == '__main__':
    run_ablation_experiment(seed=42)
