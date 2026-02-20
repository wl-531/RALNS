"""可扩展性实验：固定任务密度，变化服务器数量"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
import pandas as pd
from copy import deepcopy

from config import (KAPPA, N_PERIODS, N_RUNS, MC_SAMPLES, TYPE_MIX_GOOGLE,
                    DECISION_INTERVAL, T_MAX, PATIENCE, DESTROY_K)
from data.generator import generate_batch, generate_servers_with_target_rho
from solvers import DGSolver, RGSolver, EPDFFSolver, RALNSSolver
from evaluation import compute_metrics, monte_carlo_verify, compute_next_backlog

M_VALUES = [5, 10, 20, 50]
TASKS_PER_SERVER = 10  # 固定任务密度
RHO = 0.90

# 明确转换为毫秒
T_MAX_MS = T_MAX * 1000 if T_MAX < 10 else T_MAX


def run_scalability(seed=42, verbose=True):
    """可扩展性实验"""
    results = []

    if verbose:
        print(f"===== Scalability Experiment =====")
        print(f"  M_VALUES={M_VALUES}")
        print(f"  TASKS_PER_SERVER={TASKS_PER_SERVER}")
        print(f"  T_MAX_MS={T_MAX_MS}ms")

    for M in M_VALUES:
        n_tasks = TASKS_PER_SERVER * M

        if verbose:
            print(f"\n===== M = {M} servers, n = {n_tasks} tasks =====")

        for run_idx in range(N_RUNS):
            run_seed = seed + run_idx * 1000
            np.random.seed(run_seed)

            # 固定本 run 的任务和物理系统
            tasks_list = [generate_batch(n_tasks, type_mix=TYPE_MIX_GOOGLE)
                          for _ in range(N_PERIODS)]
            sample_tasks = tasks_list[0]
            servers_init = generate_servers_with_target_rho(
                M, sample_tasks, target_rho=RHO, kappa=KAPPA, decision_interval=DECISION_INTERVAL
            )

            algorithms = {
                'DG': lambda: DGSolver(),
                'RG': lambda: RGSolver(kappa=KAPPA),
                'EPD-FF': lambda: EPDFFSolver(kappa=KAPPA),
                'RA-LNS': lambda: RALNSSolver(kappa=KAPPA, t_max=T_MAX,
                                               patience=PATIENCE, destroy_k=DESTROY_K),
            }

            for algo_name, solver_factory in algorithms.items():
                np.random.seed(run_seed)  # 重置随机种子
                solver = solver_factory()
                servers = deepcopy(servers_init)

                cvr_list, time_list, robust_util_list, util_total_list = [], [], [], []
                jfi_list = []

                for t in range(N_PERIODS):
                    tasks = tasks_list[t]

                    t0 = time.perf_counter()
                    assignment = solver.solve(tasks, servers)
                    solve_time = (time.perf_counter() - t0) * 1000

                    metrics = compute_metrics(assignment, tasks, servers, KAPPA)
                    system_cvr, _, _ = monte_carlo_verify(assignment, tasks, servers, MC_SAMPLES)

                    cvr_list.append(system_cvr)
                    time_list.append(solve_time)
                    robust_util_list.append(metrics['robust_util_total'])
                    util_total_list.append(metrics['util_total'])
                    jfi_list.append(metrics['jfi'])

                    next_backlog = compute_next_backlog(assignment, tasks, servers, DECISION_INTERVAL)
                    for j in range(M):
                        servers[j].L0 = next_backlog[j]

                time_arr = np.array(time_list, dtype=float)

                results.append({
                    'M': M,
                    'n_tasks': n_tasks,
                    'algorithm': algo_name,
                    'run': run_idx,
                    'cvr_mean': np.mean(cvr_list),
                    'robust_util_total_mean': np.mean(robust_util_list),
                    'util_total_mean': np.mean(util_total_list),
                    'jfi_mean': np.mean(jfi_list),
                    'time_mean_ms': np.mean(time_arr),
                    'time_std_ms': float(np.std(time_arr)),
                    'time_p99_ms': float(np.percentile(time_arr, 99)),
                    'time_max_ms': float(np.max(time_arr)),
                    'timeout_rate': float(np.mean(time_arr >= T_MAX_MS * 0.999)),
                })

            if verbose and run_idx == 0:
                print(f"  Run 0:")
                for algo in ['DG', 'RG', 'EPD-FF', 'RA-LNS']:
                    row = [r for r in results if r['M'] == M
                           and r['algorithm'] == algo and r['run'] == 0][0]
                    print(f"    {algo}: CVR={row['cvr_mean']:.4f}, "
                          f"Time={row['time_mean_ms']:.2f}ms, "
                          f"timeout_rate={row['timeout_rate']:.2%}")

    df = pd.DataFrame(results)
    df.to_csv('results_scalability.csv', index=False)

    if verbose:
        print("\n===== CVR Summary (M x Algorithm) =====")
        cvr_summary = df.pivot_table(index='M', columns='algorithm',
                                      values='cvr_mean', aggfunc='mean').round(4)
        cvr_summary = cvr_summary[['DG', 'RG', 'EPD-FF', 'RA-LNS']]
        print(cvr_summary)

        print("\n===== Runtime Summary (M x Algorithm) =====")
        time_summary = df.pivot_table(index='M', columns='algorithm',
                                       values='time_mean_ms', aggfunc='mean').round(2)
        time_summary = time_summary[['DG', 'RG', 'EPD-FF', 'RA-LNS']]
        print(time_summary)

        print("\n===== Timeout Rate Summary (M x Algorithm) =====")
        timeout_summary = df.pivot_table(index='M', columns='algorithm',
                                          values='timeout_rate', aggfunc='mean').round(4)
        timeout_summary = timeout_summary[['DG', 'RG', 'EPD-FF', 'RA-LNS']]
        print(timeout_summary)

        print("\n===== Task Density Verification =====")
        density_check = df.groupby('M')['n_tasks'].first()
        print(density_check)

    return df


if __name__ == '__main__':
    run_scalability(seed=42, verbose=True)
