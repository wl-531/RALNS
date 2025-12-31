"""主实验：4 场景 × 4 算法 × N_RUNS"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
import pandas as pd
from copy import deepcopy

from config import (KAPPA, N_PERIODS, N_RUNS, MC_SAMPLES, SCENARIOS,
                    DECISION_INTERVAL, T_MAX, PATIENCE, DESTROY_K)
from models import Task, Server
from data.generator import generate_tasks, generate_servers_with_target_rho
from solvers import DGSolver, RGSolver, RALNSSolver, MicroLNSSolver
from evaluation import compute_metrics, monte_carlo_verify, compute_next_backlog


def run_online_simulation(solver, tasks_list, servers_init, n_periods):
    """运行在线仿真

    Args:
        solver: 求解器
        tasks_list: 每周期的任务列表
        servers_init: 初始服务器列表
        n_periods: 周期数

    Returns:
        metrics: dict
    """
    servers = deepcopy(servers_init)
    m = len(servers)

    results = {
        'cvr': [],
        'per_server_vr': [],
        'excess': [],
        'U_max': [],
        'O1': [],
        'time_ms': [],
        'feasible': []
    }

    for t in range(n_periods):
        tasks = tasks_list[t]

        # 求解
        t0 = time.perf_counter()
        assignment = solver.solve(tasks, servers)
        solve_time = (time.perf_counter() - t0) * 1000

        # 计算指标
        metrics = compute_metrics(assignment, tasks, servers, KAPPA)
        system_cvr, per_server_vr, avg_excess = monte_carlo_verify(
            assignment, tasks, servers, MC_SAMPLES
        )

        # 记录
        results['cvr'].append(system_cvr)
        results['per_server_vr'].append(per_server_vr)
        results['excess'].append(avg_excess)
        results['U_max'].append(metrics['U_max'])
        results['O1'].append(metrics['O1'])
        results['time_ms'].append(solve_time)
        results['feasible'].append(metrics['feasible'])

        # 更新 backlog
        next_backlog = compute_next_backlog(assignment, tasks, servers, DECISION_INTERVAL)
        for j in range(m):
            servers[j].L0 = next_backlog[j]

    return results


def run_main_experiment(seed=42, verbose=True):
    """主实验入口"""
    np.random.seed(seed)

    algorithms = {
        'DG': DGSolver(),
        'RG': RGSolver(kappa=KAPPA),
        'Micro': MicroLNSSolver(kappa=KAPPA, t_max=T_MAX, patience=PATIENCE),
        'RA-LNS': RALNSSolver(kappa=KAPPA, t_max=T_MAX, patience=PATIENCE, destroy_k=DESTROY_K),
    }

    all_results = []

    for scenario_name, scenario_cfg in SCENARIOS.items():
        if verbose:
            print(f"\n===== Scenario: {scenario_name} =====")
            print(f"  n_tasks={scenario_cfg['n_tasks']}, m_servers={scenario_cfg['m_servers']}, rho={scenario_cfg['rho']}")

        for run_idx in range(N_RUNS):
            # 固定随机种子（可复现）
            run_seed = seed + run_idx * 1000
            np.random.seed(run_seed)

            # 预生成所有周期的任务
            tasks_list = [
                generate_tasks(scenario_cfg['n_tasks'], mode='bimodal')
                for _ in range(N_PERIODS)
            ]

            # 计算总期望负载，生成服务器
            total_mu = sum(sum(t.mu for t in tasks) for tasks in tasks_list) / N_PERIODS
            servers_init = generate_servers_with_target_rho(
                scenario_cfg['m_servers'], total_mu,
                scenario_cfg['rho'], DECISION_INTERVAL
            )

            for algo_name, solver in algorithms.items():
                np.random.seed(run_seed)  # 确保每个算法使用相同的随机序列

                results = run_online_simulation(solver, tasks_list, servers_init, N_PERIODS)

                # 汇总
                row = {
                    'scenario': scenario_name,
                    'algorithm': algo_name,
                    'run': run_idx,
                    'cvr_mean': np.mean(results['cvr']),
                    'cvr_std': np.std(results['cvr']),
                    'U_max_mean': np.mean(results['U_max']),
                    'O1_mean': np.mean(results['O1']),
                    'excess_mean': np.mean(results['excess']),
                    'time_mean_ms': np.mean(results['time_ms']),
                    'time_p99_ms': np.percentile(results['time_ms'], 99),
                    'feasible_rate': np.mean(results['feasible']),
                }
                all_results.append(row)

            if verbose and run_idx == 0:
                # 打印第一轮结果
                for algo_name in algorithms.keys():
                    row = [r for r in all_results if r['scenario'] == scenario_name
                           and r['algorithm'] == algo_name and r['run'] == 0][0]
                    print(f"  {algo_name}: CVR={row['cvr_mean']:.4f}, U_max={row['U_max_mean']:.3f}, "
                          f"Time={row['time_mean_ms']:.2f}ms")

    # 保存结果
    df = pd.DataFrame(all_results)
    df.to_csv('results_main.csv', index=False)

    # 打印汇总
    if verbose:
        print("\n===== Summary =====")
        summary = df.groupby(['scenario', 'algorithm']).agg({
            'cvr_mean': ['mean', 'std'],
            'U_max_mean': 'mean',
            'time_mean_ms': 'mean'
        }).round(4)
        print(summary)

    return df


if __name__ == '__main__':
    run_main_experiment(seed=42, verbose=True)
