"""参数敏感性实验：kappa, patience, destroy_k"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from copy import deepcopy

from config import (KAPPA, N_PERIODS, MC_SAMPLES, DECISION_INTERVAL, T_MAX, PATIENCE, DESTROY_K,
                    MAIN_CONFIG)
from data.generator import generate_batch, generate_servers_with_target_rho
from solvers import RALNSSolver
from evaluation import compute_metrics, monte_carlo_verify, compute_next_backlog


def run_sensitivity_kappa(seed=42):
    """kappa 敏感性"""
    np.random.seed(seed)

    cfg = MAIN_CONFIG
    kappa_values = [1.5, 2.0, 2.38, 2.8, 3.5]

    tasks_list = [generate_batch(cfg['n_tasks'], type_mix=cfg['type_mix']) for _ in range(N_PERIODS)]
    sample_tasks = tasks_list[0]
    servers_init = generate_servers_with_target_rho(cfg['m_servers'], sample_tasks, cfg['rho'], KAPPA, DECISION_INTERVAL)

    results = []
    for kappa in kappa_values:
        solver = RALNSSolver(kappa=kappa, t_max=T_MAX, patience=PATIENCE, destroy_k=DESTROY_K)
        servers = deepcopy(servers_init)

        cvr_list = []
        U_max_list = []
        for t in range(N_PERIODS):
            tasks = tasks_list[t]
            assignment = solver.solve(tasks, servers)
            metrics = compute_metrics(assignment, tasks, servers, kappa)
            system_cvr, _, _ = monte_carlo_verify(assignment, tasks, servers, MC_SAMPLES)

            cvr_list.append(system_cvr)
            U_max_list.append(metrics['U_max'])

            next_backlog = compute_next_backlog(assignment, tasks, servers, DECISION_INTERVAL)
            for j in range(cfg['m_servers']):
                servers[j].L0 = next_backlog[j]

        results.append({
            'kappa': kappa,
            'cvr_mean': np.mean(cvr_list),
            'U_max_mean': np.mean(U_max_list),
        })
        print(f"kappa={kappa}: CVR={np.mean(cvr_list):.4f}, U_max={np.mean(U_max_list):.3f}")

    return pd.DataFrame(results)


def run_sensitivity_patience(seed=42):
    """patience 敏感性"""
    np.random.seed(seed)

    cfg = MAIN_CONFIG
    patience_values = [5, 10, 15, 20, 30]

    tasks_list = [generate_batch(cfg['n_tasks'], type_mix=cfg['type_mix']) for _ in range(N_PERIODS)]
    sample_tasks = tasks_list[0]
    servers_init = generate_servers_with_target_rho(cfg['m_servers'], sample_tasks, cfg['rho'], KAPPA, DECISION_INTERVAL)

    results = []
    for patience in patience_values:
        solver = RALNSSolver(kappa=KAPPA, t_max=T_MAX, patience=patience, destroy_k=DESTROY_K)
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
            'patience': patience,
            'cvr_mean': np.mean(cvr_list),
        })
        print(f"patience={patience}: CVR={np.mean(cvr_list):.4f}")

    return pd.DataFrame(results)


def run_sensitivity_destroy_k(seed=42):
    """destroy_k 敏感性"""
    np.random.seed(seed)

    cfg = MAIN_CONFIG
    destroy_k_values = [1, 2, 3, 5, 8]

    tasks_list = [generate_batch(cfg['n_tasks'], type_mix=cfg['type_mix']) for _ in range(N_PERIODS)]
    sample_tasks = tasks_list[0]
    servers_init = generate_servers_with_target_rho(cfg['m_servers'], sample_tasks, cfg['rho'], KAPPA, DECISION_INTERVAL)

    results = []
    for destroy_k in destroy_k_values:
        solver = RALNSSolver(kappa=KAPPA, t_max=T_MAX, patience=PATIENCE, destroy_k=destroy_k)
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
            'destroy_k': destroy_k,
            'cvr_mean': np.mean(cvr_list),
        })
        print(f"destroy_k={destroy_k}: CVR={np.mean(cvr_list):.4f}")

    return pd.DataFrame(results)


if __name__ == '__main__':
    print("===== Sensitivity: kappa =====")
    df_kappa = run_sensitivity_kappa(seed=42)
    df_kappa.to_csv('results_sensitivity_kappa.csv', index=False)

    print("\n===== Sensitivity: patience =====")
    df_patience = run_sensitivity_patience(seed=42)
    df_patience.to_csv('results_sensitivity_patience.csv', index=False)

    print("\n===== Sensitivity: destroy_k =====")
    df_destroy_k = run_sensitivity_destroy_k(seed=42)
    df_destroy_k.to_csv('results_sensitivity_destroy_k.csv', index=False)
