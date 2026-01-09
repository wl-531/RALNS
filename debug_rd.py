"""调试：为什么 Random-Destroy 比 RA-LNS 更好？"""
import numpy as np
from copy import deepcopy

from config import KAPPA, MC_SAMPLES, DECISION_INTERVAL, T_MAX, PATIENCE, DESTROY_K, MAIN_CONFIG
from data.generator import generate_batch, generate_servers_with_target_rho
from solvers import RALNSSolver, RandomDestroySolver
from solvers.ra_lns import RALNSSolution
from evaluation import monte_carlo_verify, compute_next_backlog

np.random.seed(42)
cfg = MAIN_CONFIG

tasks = generate_batch(cfg['n_tasks'], type_mix=cfg['type_mix'])
servers = generate_servers_with_target_rho(cfg['m_servers'], tasks, cfg['rho'], KAPPA, DECISION_INTERVAL)

print("=" * 60)
print("调试：RA-LNS vs Random-Destroy")
print("=" * 60)

# 运行 RA-LNS 并记录搜索过程
class DebugRALNSSolver(RALNSSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stage1a_calls = 0
        self.stage1a_success = 0
        self.stage1b_calls = 0
        self.stage1b_success = 0
        self.j_hot_history = []

    def _risk_hedging_move(self, sol, tasks):
        self.stage1a_calls += 1
        j_hot = int(np.argmax(sol.RD))
        self.j_hot_history.append(j_hot)
        result = super()._risk_hedging_move(sol, tasks)
        if result:
            self.stage1a_success += 1
        return result

    def _risk_guided_lns(self, sol, tasks):
        self.stage1b_calls += 1
        result = super()._risk_guided_lns(sol, tasks)
        if result:
            self.stage1b_success += 1
        return result

class DebugRandomDestroySolver(RandomDestroySolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stage1a_calls = 0
        self.stage1a_success = 0
        self.stage1b_calls = 0
        self.stage1b_success = 0
        self.j_hot_history = []

    def _risk_hedging_move(self, sol, tasks):
        self.stage1a_calls += 1
        j_hot = np.random.randint(sol.m)
        self.j_hot_history.append(j_hot)
        result = super()._risk_hedging_move(sol, tasks)
        if result:
            self.stage1a_success += 1
        return result

    def _risk_guided_lns(self, sol, tasks):
        self.stage1b_calls += 1
        result = super()._risk_guided_lns(sol, tasks)
        if result:
            self.stage1b_success += 1
        return result

# RA-LNS
solver1 = DebugRALNSSolver(kappa=KAPPA, t_max=T_MAX, patience=PATIENCE, destroy_k=DESTROY_K)
assignment1 = solver1.solve(tasks, deepcopy(servers))
cvr1, _, _ = monte_carlo_verify(assignment1, tasks, servers, MC_SAMPLES)

print(f"\n--- RA-LNS ---")
print(f"CVR: {cvr1:.4f}")
print(f"Stage-1A: {solver1.stage1a_success}/{solver1.stage1a_calls} success")
print(f"Stage-1B: {solver1.stage1b_success}/{solver1.stage1b_calls} success")
print(f"j_hot 选择分布: {dict(zip(*np.unique(solver1.j_hot_history, return_counts=True)))}")

# Random-Destroy
np.random.seed(42)
solver2 = DebugRandomDestroySolver(kappa=KAPPA, t_max=T_MAX, patience=PATIENCE, destroy_k=DESTROY_K)
assignment2 = solver2.solve(tasks, deepcopy(servers))
cvr2, _, _ = monte_carlo_verify(assignment2, tasks, servers, MC_SAMPLES)

print(f"\n--- Random-Destroy ---")
print(f"CVR: {cvr2:.4f}")
print(f"Stage-1A: {solver2.stage1a_success}/{solver2.stage1a_calls} success")
print(f"Stage-1B: {solver2.stage1b_success}/{solver2.stage1b_calls} success")
print(f"j_hot 选择分布: {dict(zip(*np.unique(solver2.j_hot_history, return_counts=True)))}")

# 分析解的质量
def analyze_solution(assignment, tasks, servers, name):
    sol = RALNSSolution(servers, KAPPA)
    sol.assignment = [-1] * len(tasks)
    for i, j in enumerate(assignment):
        sol.mu_sum[j] += tasks[i].mu
        sol.sigma_sq_sum[j] += tasks[i].sigma ** 2
        sol.assignment[i] = j

    print(f"\n--- {name} 解分析 ---")
    print(f"Feasible: {sol.is_feasible()}")
    print(f"RR_max: {sol.RR_max:.4f}")
    print(f"O1: {sol.O1:.2f}")
    print(f"RR per server: {[f'{r:.3f}' for r in sol.RR]}")
    print(f"RD per server: {[f'{r:.3f}' for r in sol.RD]}")

analyze_solution(assignment1, tasks, servers, "RA-LNS")
analyze_solution(assignment2, tasks, servers, "Random-Destroy")
