"""Quality vs Time 实验：不同时间预算下的性能"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from copy import deepcopy

from config import (KAPPA, N_PERIODS, MC_SAMPLES, DECISION_INTERVAL, PATIENCE, DESTROY_K,
                    MAIN_CONFIG)
from data.generator import generate_batch, generate_servers_with_target_rho
from solvers import RALNSSolver
from evaluation import compute_metrics, monte_carlo_verify, compute_next_backlog


def run_quality_time_experiment(seed=42):
    """Quality vs Time 实验"""
    cfg = MAIN_CONFIG

    # 时间预算列表（毫秒）
    time_budgets_ms = [1, 2, 5, 10, 20, 50, 100]

    # 在循环外生成固定的任务和服务器（所有时间预算共用）
    np.random.seed(seed)
    tasks_list = [generate_batch(cfg['n_tasks'], type_mix=cfg['type_mix']) for _ in range(N_PERIODS)]
    sample_tasks = tasks_list[0]
    servers_init = generate_servers_with_target_rho(cfg['m_servers'], sample_tasks, cfg['rho'], KAPPA, DECISION_INTERVAL)

    results = []

    for t_max_ms in time_budgets_ms:
        np.random.seed(seed)  # 重置种子，确保 MC 验证可比
        t_max = t_max_ms / 1000.0
        solver = RALNSSolver(kappa=KAPPA, t_max=t_max, patience=PATIENCE, destroy_k=DESTROY_K)

        servers = deepcopy(servers_init)
        cvr_list = []
        U_max_list = []

        for t in range(N_PERIODS):
            tasks = tasks_list[t]
            assignment = solver.solve(tasks, servers)

            metrics = compute_metrics(assignment, tasks, servers, KAPPA)
            system_cvr, _, _ = monte_carlo_verify(assignment, tasks, servers, MC_SAMPLES)

            cvr_list.append(system_cvr)
            U_max_list.append(metrics['U_max'])

            next_backlog = compute_next_backlog(assignment, tasks, servers, DECISION_INTERVAL)
            for j in range(cfg['m_servers']):
                servers[j].L0 = next_backlog[j]

        results.append({
            't_max_ms': t_max_ms,
            'cvr_mean': np.mean(cvr_list),
            'cvr_std': np.std(cvr_list),
            'U_max_mean': np.mean(U_max_list),
        })

        print(f"T_max={t_max_ms}ms: CVR={np.mean(cvr_list):.4f}, U_max={np.mean(U_max_list):.3f}")

    df = pd.DataFrame(results)
    df.to_csv('results_quality_time.csv', index=False)
    return df


if __name__ == '__main__':
    run_quality_time_experiment(seed=42)
