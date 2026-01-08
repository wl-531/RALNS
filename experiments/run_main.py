"""主实验：4 算法对比 (DG/RG/Std-LNS/RA-LNS)"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
import pandas as pd
from copy import deepcopy

from config import (KAPPA, N_PERIODS, N_RUNS, MC_SAMPLES, MAIN_CONFIG,
                    DECISION_INTERVAL, T_MAX, PATIENCE, DESTROY_K)
from data.generator import generate_batch, generate_servers_with_target_rho
from solvers import DGSolver, RGSolver, StdLNSSolver, RALNSSolver
from evaluation import compute_metrics, monte_carlo_verify, compute_next_backlog


def run_online_simulation(solver, tasks_list, servers_init, n_periods):
    """运行在线仿真，收集 backlog"""
    servers = deepcopy(servers_init)
    m = len(servers)

    results = {
        'cvr': [], 'per_server_vr': [], 'excess': [],
        'U_max': [], 'O1': [], 'time_ms': [], 'feasible': [],
        'backlog': []  # 每周期开始时的系统总 backlog
    }

    for t in range(n_periods):
        tasks = tasks_list[t]

        # 记录周期开始时的 backlog
        system_backlog = sum(s.L0 for s in servers)
        results['backlog'].append(system_backlog)

        t0 = time.perf_counter()
        assignment = solver.solve(tasks, servers)
        solve_time = (time.perf_counter() - t0) * 1000

        metrics = compute_metrics(assignment, tasks, servers, KAPPA)
        system_cvr, per_server_vr, avg_excess = monte_carlo_verify(
            assignment, tasks, servers, MC_SAMPLES
        )

        results['cvr'].append(system_cvr)
        results['per_server_vr'].append(per_server_vr)
        results['excess'].append(avg_excess)
        results['U_max'].append(metrics['U_max'])
        results['O1'].append(metrics['O1'])
        results['time_ms'].append(solve_time)
        results['feasible'].append(metrics['feasible'])

        next_backlog = compute_next_backlog(assignment, tasks, servers, DECISION_INTERVAL)
        for j in range(m):
            servers[j].L0 = next_backlog[j]

    return results


def run_main_experiment(seed=42, verbose=True):
    """主实验入口"""
    np.random.seed(seed)

    cfg = MAIN_CONFIG

    algorithms = {
        'DG': DGSolver(),
        'RG': RGSolver(kappa=KAPPA),
        'Std-LNS': StdLNSSolver(t_max=T_MAX, patience=PATIENCE, destroy_k=DESTROY_K),
        'RA-LNS': RALNSSolver(kappa=KAPPA, t_max=T_MAX, patience=PATIENCE, destroy_k=DESTROY_K),
    }

    all_results = []
    backlog_data = []  # 收集 backlog 时间序列

    if verbose:
        print(f"===== Main Experiment =====")
        print(f"  n_tasks={cfg['n_tasks']}, m_servers={cfg['m_servers']}, rho={cfg['rho']}")
        print(f"  type_mix={cfg['type_mix']} (Google Trace distribution)")

    for run_idx in range(N_RUNS):
        run_seed = seed + run_idx * 1000
        np.random.seed(run_seed)

        # 预生成所有周期的任务
        tasks_list = [
            generate_batch(cfg['n_tasks'], type_mix=cfg['type_mix'])
            for _ in range(N_PERIODS)
        ]

        # 用第一个周期的任务计算鲁棒容量
        sample_tasks = tasks_list[0]
        servers_init = generate_servers_with_target_rho(
            cfg['m_servers'], sample_tasks, cfg['rho'], KAPPA, DECISION_INTERVAL
        )

        run_backlog = {'run': run_idx}  # 本次运行的 backlog

        for algo_name, solver in algorithms.items():
            np.random.seed(run_seed)

            results = run_online_simulation(solver, tasks_list, servers_init, N_PERIODS)

            row = {
                'algorithm': algo_name,
                'run': run_idx,
                'cvr_mean': np.mean(results['cvr']),
                'cvr_std': np.std(results['cvr']),
                'per_server_vr_mean': np.mean(results['per_server_vr']),
                'U_max_mean': np.mean(results['U_max']),
                'O1_mean': np.mean(results['O1']),
                'excess_mean': np.mean(results['excess']),
                'time_mean_ms': np.mean(results['time_ms']),
                'time_p99_ms': np.percentile(results['time_ms'], 99),
                'feasible_rate': np.mean(results['feasible']),
            }
            all_results.append(row)

            # 收集 backlog 数据（仅第一次运行，用于绘图）
            if run_idx == 0:
                for t, bl in enumerate(results['backlog']):
                    backlog_data.append({
                        'period': t,
                        'algorithm': algo_name,
                        'backlog': bl
                    })

        if verbose and run_idx == 0:
            print(f"\n  Run 0 results:")
            for algo_name in algorithms.keys():
                row = [r for r in all_results if r['algorithm'] == algo_name and r['run'] == 0][0]
                print(f"    {algo_name}: CVR={row['cvr_mean']:.4f}, U_max={row['U_max_mean']:.3f}, "
                      f"Time={row['time_mean_ms']:.2f}ms")

    df = pd.DataFrame(all_results)
    df.to_csv('results_main.csv', index=False)

    # 保存 backlog 数据（pivot 格式）
    if backlog_data:
        df_backlog = pd.DataFrame(backlog_data)
        df_backlog_pivot = df_backlog.pivot(index='period', columns='algorithm', values='backlog')
        df_backlog_pivot = df_backlog_pivot[['DG', 'RG', 'Std-LNS', 'RA-LNS']]  # 确保顺序
        df_backlog_pivot.to_csv('results_backlog.csv')

    if verbose:
        print("\n===== Summary (mean +/- std) =====")
        summary = df.groupby('algorithm').agg({
            'cvr_mean': ['mean', 'std'],
            'U_max_mean': 'mean',
            'excess_mean': 'mean',
            'time_mean_ms': 'mean'
        }).round(4)
        print(summary)

    return df


if __name__ == '__main__':
    run_main_experiment(seed=42, verbose=True)
