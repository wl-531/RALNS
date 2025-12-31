"""消融实验：RA-LNS vs Micro-LNS vs RG"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from copy import deepcopy

from config import (KAPPA, N_PERIODS, N_RUNS, MC_SAMPLES, DECISION_INTERVAL,
                    T_MAX, PATIENCE, DESTROY_K, MAIN_CONFIG)
from data.generator import generate_batch, generate_servers_with_target_rho
from solvers import RGSolver, RALNSSolver, MicroLNSSolver
from evaluation import compute_metrics, monte_carlo_verify, compute_next_backlog


def run_ablation_experiment(seed=42):
    """消融实验"""
    np.random.seed(seed)

    cfg = MAIN_CONFIG

    algorithms = {
        'RG': RGSolver(kappa=KAPPA),
        'Micro-LNS': MicroLNSSolver(kappa=KAPPA, t_max=T_MAX, patience=PATIENCE),
        'RA-LNS': RALNSSolver(kappa=KAPPA, t_max=T_MAX, patience=PATIENCE, destroy_k=DESTROY_K),
    }

    results = []

    for run_idx in range(N_RUNS):
        run_seed = seed + run_idx * 1000
        np.random.seed(run_seed)

        tasks_list = [generate_batch(cfg['n_tasks'], type_mix=cfg['type_mix']) for _ in range(N_PERIODS)]
        sample_tasks = tasks_list[0]
        servers_init = generate_servers_with_target_rho(cfg['m_servers'], sample_tasks, cfg['rho'], KAPPA, DECISION_INTERVAL)

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
                for j in range(cfg['m_servers']):
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
