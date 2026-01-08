"""测试 rho=0.85 的消融实验"""
import numpy as np
from copy import deepcopy

from config import KAPPA, MC_SAMPLES, DECISION_INTERVAL, T_MAX, PATIENCE, DESTROY_K, MAIN_CONFIG
from data.generator import generate_batch, generate_servers_with_target_rho
from solvers import ConstructionOnlySolver, MicroLNSSolver, RandomDestroySolver, RALNSSolver
from evaluation import monte_carlo_verify, compute_next_backlog

np.random.seed(42)
cfg = MAIN_CONFIG
print(f"Config: n={cfg['n_tasks']}, m={cfg['m_servers']}, rho={cfg['rho']}")

# 生成数据
tasks_list = [generate_batch(cfg['n_tasks'], type_mix=cfg['type_mix']) for _ in range(20)]
sample_tasks = tasks_list[0]
servers_init = generate_servers_with_target_rho(cfg['m_servers'], sample_tasks, cfg['rho'], KAPPA, DECISION_INTERVAL)

caps = [f"{s.C:.1f}" for s in servers_init]
print(f"Server capacities: {caps}")

algorithms = {
    'Construction': ConstructionOnlySolver(kappa=KAPPA),
    'Micro-Only': MicroLNSSolver(kappa=KAPPA, t_max=T_MAX, patience=PATIENCE),
    'Random-Destroy': RandomDestroySolver(kappa=KAPPA, t_max=T_MAX, patience=PATIENCE, destroy_k=DESTROY_K),
    'RA-LNS': RALNSSolver(kappa=KAPPA, t_max=T_MAX, patience=PATIENCE, destroy_k=DESTROY_K),
}

for algo_name, solver in algorithms.items():
    np.random.seed(42)
    servers = deepcopy(servers_init)

    cvr_list = []
    for t in range(20):
        tasks = tasks_list[t]
        assignment = solver.solve(tasks, servers)
        system_cvr, _, _ = monte_carlo_verify(assignment, tasks, servers, MC_SAMPLES)
        cvr_list.append(system_cvr)

        next_backlog = compute_next_backlog(assignment, tasks, servers, DECISION_INTERVAL)
        for j in range(cfg['m_servers']):
            servers[j].L0 = next_backlog[j]

    print(f"{algo_name}: CVR={np.mean(cvr_list):.4f} (std={np.std(cvr_list):.4f})")
