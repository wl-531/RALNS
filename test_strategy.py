"""测试不同服务器选择策略"""
import numpy as np
from copy import deepcopy

from config import KAPPA, MC_SAMPLES, DECISION_INTERVAL, T_MAX, PATIENCE, DESTROY_K, MAIN_CONFIG
from data.generator import generate_batch, generate_servers_with_target_rho
from solvers.ra_lns import RALNSSolver, RALNSSolution
from evaluation import monte_carlo_verify, compute_next_backlog

# 策略1: 只选 argmax(RD)
class ArgmaxRDSolver(RALNSSolver):
    def _risk_hedging_move(self, sol, tasks):
        j_hot = int(np.argmax(sol.RD))
        return self._try_server(sol, tasks, j_hot)

    def _try_server(self, sol, tasks, from_j):
        victims = [i for i, j in enumerate(sol.assignment) if j == from_j]
        if not victims:
            return False
        victim_sigmas = [(i, tasks[i].sigma) for i in victims]
        victim_sigmas.sort(key=lambda x: -x[1])

        best_move = None
        best_psi = sol.Psi()
        best_rsum = sol.R_sum

        for victim_idx, _ in victim_sigmas:
            victim_task = tasks[victim_idx]
            for to_j in range(sol.m):
                if to_j == from_j:
                    continue
                sol.apply_move(victim_idx, victim_task, from_j, to_j)
                new_psi = sol.Psi()
                new_rsum = sol.R_sum
                if self._psi_better(new_psi, best_psi, new_rsum, best_rsum):
                    best_psi = new_psi
                    best_rsum = new_rsum
                    best_move = ('relocate', victim_idx, victim_task, from_j, to_j)
                sol.rollback_move(victim_idx, victim_task, from_j, to_j)

        if best_move:
            _, vi, vt, fj, tj = best_move
            sol.apply_move(vi, vt, fj, tj)
            return True
        return False

# 策略2: 轮询所有服务器
class RoundRobinSolver(RALNSSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_server = 0

    def _risk_hedging_move(self, sol, tasks):
        for _ in range(sol.m):
            from_j = self.current_server
            self.current_server = (self.current_server + 1) % sol.m
            if self._try_server(sol, tasks, from_j):
                return True
        return False

    def _try_server(self, sol, tasks, from_j):
        victims = [i for i, j in enumerate(sol.assignment) if j == from_j]
        if not victims:
            return False
        victim_sigmas = [(i, tasks[i].sigma) for i in victims]
        victim_sigmas.sort(key=lambda x: -x[1])

        best_move = None
        best_psi = sol.Psi()
        best_rsum = sol.R_sum

        for victim_idx, _ in victim_sigmas:
            victim_task = tasks[victim_idx]
            for to_j in range(sol.m):
                if to_j == from_j:
                    continue
                sol.apply_move(victim_idx, victim_task, from_j, to_j)
                new_psi = sol.Psi()
                new_rsum = sol.R_sum
                if self._psi_better(new_psi, best_psi, new_rsum, best_rsum):
                    best_psi = new_psi
                    best_rsum = new_rsum
                    best_move = ('relocate', victim_idx, victim_task, from_j, to_j)
                sol.rollback_move(victim_idx, victim_task, from_j, to_j)

        if best_move:
            _, vi, vt, fj, tj = best_move
            sol.apply_move(vi, vt, fj, tj)
            return True
        return False

# 策略3: 随机选服务器
class RandomServerSolver(RALNSSolver):
    def _risk_hedging_move(self, sol, tasks):
        from_j = np.random.randint(sol.m)
        return self._try_server(sol, tasks, from_j)

    def _try_server(self, sol, tasks, from_j):
        victims = [i for i, j in enumerate(sol.assignment) if j == from_j]
        if not victims:
            return False
        victim_sigmas = [(i, tasks[i].sigma) for i in victims]
        victim_sigmas.sort(key=lambda x: -x[1])

        best_move = None
        best_psi = sol.Psi()
        best_rsum = sol.R_sum

        for victim_idx, _ in victim_sigmas:
            victim_task = tasks[victim_idx]
            for to_j in range(sol.m):
                if to_j == from_j:
                    continue
                sol.apply_move(victim_idx, victim_task, from_j, to_j)
                new_psi = sol.Psi()
                new_rsum = sol.R_sum
                if self._psi_better(new_psi, best_psi, new_rsum, best_rsum):
                    best_psi = new_psi
                    best_rsum = new_rsum
                    best_move = ('relocate', victim_idx, victim_task, from_j, to_j)
                sol.rollback_move(victim_idx, victim_task, from_j, to_j)

        if best_move:
            _, vi, vt, fj, tj = best_move
            sol.apply_move(vi, vt, fj, tj)
            return True
        return False

# 测试
np.random.seed(42)
cfg = MAIN_CONFIG

algorithms = {
    'ArgmaxRD': ArgmaxRDSolver(kappa=KAPPA, t_max=T_MAX, patience=PATIENCE, destroy_k=DESTROY_K),
    'RoundRobin': RoundRobinSolver(kappa=KAPPA, t_max=T_MAX, patience=PATIENCE, destroy_k=DESTROY_K),
    'Random': RandomServerSolver(kappa=KAPPA, t_max=T_MAX, patience=PATIENCE, destroy_k=DESTROY_K),
    'RD-Ordered': RALNSSolver(kappa=KAPPA, t_max=T_MAX, patience=PATIENCE, destroy_k=DESTROY_K),
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
        if hasattr(solver, 'current_server'):
            solver.current_server = 0
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

print("===== Server Selection Strategy Comparison =====")
for name in algorithms:
    mean = np.mean(results[name])
    std = np.std(results[name])
    print(f"{name}: CVR={mean:.4f} (std={std:.4f})")
