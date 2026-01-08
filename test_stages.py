"""测试不同 rho 下各算法的 CVR"""
import numpy as np
from copy import deepcopy

from config import KAPPA, MC_SAMPLES, DECISION_INTERVAL, T_MAX, PATIENCE, DESTROY_K, MAIN_CONFIG
from data.generator import generate_batch, generate_servers_with_target_rho
from solvers import ConstructionOnlySolver, RALNSSolver
from evaluation import monte_carlo_verify, compute_next_backlog

def test_rho(rho_val, n_periods=20):
    np.random.seed(42)
    cfg = MAIN_CONFIG.copy()
    cfg['rho'] = rho_val

    tasks_list = [generate_batch(cfg['n_tasks'], type_mix=cfg['type_mix']) for _ in range(n_periods)]
    sample_tasks = tasks_list[0]
    servers_init = generate_servers_with_target_rho(cfg['m_servers'], sample_tasks, rho_val, KAPPA, DECISION_INTERVAL)

    results = {}
    for algo_name, solver in [
        ('Const', ConstructionOnlySolver(kappa=KAPPA)),
        ('RA-LNS', RALNSSolver(kappa=KAPPA, t_max=T_MAX, patience=PATIENCE, destroy_k=DESTROY_K)),
    ]:
        np.random.seed(42)
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

        results[algo_name] = np.mean(cvr_list)

    return results

print("rho\tConst\tRA-LNS\tImprove")
print("-" * 40)
for rho in [0.80, 0.85, 0.88, 0.90, 0.92, 0.95]:
    r = test_rho(rho)
    improve = (r['Const'] - r['RA-LNS']) / r['Const'] * 100 if r['Const'] > 0 else 0
    print(f"{rho:.2f}\t{r['Const']:.4f}\t{r['RA-LNS']:.4f}\t{improve:+.1f}%")
