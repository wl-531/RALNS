"""快速测试 Stage-1A 修复效果"""
import numpy as np
from copy import deepcopy

from config import KAPPA, MC_SAMPLES, DECISION_INTERVAL, T_MAX, PATIENCE, DESTROY_K, MAIN_CONFIG
from data.generator import generate_batch, generate_servers_with_target_rho
from solvers import ConstructionOnlySolver, MicroLNSSolver, RandomDestroySolver, RALNSSolver
from evaluation import monte_carlo_verify, compute_next_backlog

np.random.seed(42)
cfg = MAIN_CONFIG

algorithms = {
    'Construction': ConstructionOnlySolver(kappa=KAPPA),
    'Micro-Only': MicroLNSSolver(kappa=KAPPA, t_max=T_MAX, patience=PATIENCE),
    'Random-Destroy': RandomDestroySolver(kappa=KAPPA, t_max=T_MAX, patience=PATIENCE, destroy_k=DESTROY_K),
    'RA-LNS': RALNSSolver(kappa=KAPPA, t_max=T_MAX, patience=PATIENCE, destroy_k=DESTROY_K),
}

n_runs = 5
n_periods = 20

results = {name: [] for name in algorithms}

for run_idx in range(n_runs):
    run_seed = 42 + run_idx * 1000
    np.random.seed(run_seed)

    tasks_list = [generate_batch(cfg['n_tasks'], type_mix=cfg['type_mix']) for _ in range(n_periods)]
    sample_tasks = tasks_list[0]
    servers_init = generate_servers_with_target_rho(cfg['m_servers'], sample_tasks, cfg['rho'], KAPPA, DECISION_INTERVAL)

    for algo_name, solver in algorithms.items():
        np.random.seed(run_seed)
        servers = deepcopy(servers_init)

        cvr_list = []
        for t in range(n_periods):
            tasks = tasks_list[t]
            assignment = solver.solve(tasks, servers)
            system_cvr, _, _ = monte_carlo_verify(assignment, tasks, servers, MC_SAMPLES)
            cvr_list.append(system_cvr)

            next_backlog = compute_next_backlog(assignment, tasks, servers, DECISION_INTERVAL)
            for j in range(cfg['m_servers']):
                servers[j].L0 = next_backlog[j]

        results[algo_name].append(np.mean(cvr_list))

    print(f"Run {run_idx + 1}/{n_runs} done")

print("\n===== Results =====")
for name in algorithms:
    mean = np.mean(results[name])
    std = np.std(results[name])
    print(f"{name}: CVR={mean:.4f} (std={std:.4f})")
