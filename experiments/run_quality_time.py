"""Quality vs Time 实验：不同时间预算下的性能"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from copy import deepcopy

from config import KAPPA, N_PERIODS, MC_SAMPLES, DECISION_INTERVAL, PATIENCE, DESTROY_K
from data.generator import generate_batch, generate_servers_with_target_rho
from solvers import RALNSSolver
from evaluation import compute_metrics, monte_carlo_verify, compute_next_backlog


def run_quality_time_experiment(seed=42):
    """Quality vs Time 实验"""
    np.random.seed(seed)

    # 时间预算列表（毫秒）
    time_budgets_ms = [1, 2, 5, 10, 20, 50, 100]

    # 场景配置（使用 stress 场景）
    n_tasks = 120
    m_servers = 10
    rho = 0.92

    results = []

    for t_max_ms in time_budgets_ms:
        t_max = t_max_ms / 1000.0
        solver = RALNSSolver(kappa=KAPPA, t_max=t_max, patience=PATIENCE, destroy_k=DESTROY_K)

        # 预生成任务和服务器
        tasks_list = [generate_batch(n_tasks, type_mix=[0.15, 0.70, 0.15]) for _ in range(N_PERIODS)]
        total_mu = sum(sum(t.mu for t in tasks) for tasks in tasks_list) / N_PERIODS
        servers_init = generate_servers_with_target_rho(m_servers, total_mu, rho, DECISION_INTERVAL)

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
            for j in range(m_servers):
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
