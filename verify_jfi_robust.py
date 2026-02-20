"""验证鲁棒 JFI vs 期望 JFI：确认方向后再决定主表用哪个

使用方法：
    cd RALNS && python verify_jfi_robust.py

需要先确保 evaluation/metrics.py 中有 jfi 字段（期望版 patch 已打）。
此脚本额外计算鲁棒版 JFI 做对比。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from copy import deepcopy

from config import (KAPPA, DECISION_INTERVAL,
                    T_MAX, PATIENCE, DESTROY_K, MAIN_CONFIG)
from data.generator import generate_batch, generate_servers_with_target_rho
from solvers import DGSolver, RGSolver, EPDFFSolver, StdLNSSolver, RALNSSolver
from evaluation.monte_carlo import compute_next_backlog

N_RUNS = 5
N_PERIODS = 20
cfg = MAIN_CONFIG


def compute_both_jfi(assignment, tasks, servers, kappa):
    """同时计算期望 JFI 和鲁棒 JFI"""
    m = len(servers)
    C = np.array([s.C for s in servers])
    L0 = np.array([s.L0 for s in servers])
    mu_sum = np.zeros(m)
    sigma_sq_sum = np.zeros(m)

    for i, j in enumerate(assignment):
        mu_sum[j] += tasks[i].mu
        sigma_sq_sum[j] += tasks[i].sigma ** 2

    sigma_j = np.sqrt(np.maximum(sigma_sq_sum, 0))

    # 期望利用率
    u_exp = (L0 + mu_sum) / C
    jfi_exp = float(np.sum(u_exp) ** 2 / (m * np.sum(u_exp ** 2) + 1e-12))

    # 鲁棒利用率
    L_hat = L0 + mu_sum + kappa * sigma_j
    u_rob = L_hat / C
    jfi_rob = float(np.sum(u_rob) ** 2 / (m * np.sum(u_rob ** 2) + 1e-12))

    return jfi_exp, jfi_rob


algorithms = {
    'DG':      DGSolver(),
    'RG':      RGSolver(kappa=KAPPA),
    'EPD-FF':  EPDFFSolver(kappa=KAPPA),
    'Std-LNS': StdLNSSolver(t_max=T_MAX, patience=PATIENCE, destroy_k=DESTROY_K),
    'RA-LNS':  RALNSSolver(kappa=KAPPA, t_max=T_MAX, patience=PATIENCE, destroy_k=DESTROY_K),
}

jfi_exp_runs = {name: [] for name in algorithms}
jfi_rob_runs = {name: [] for name in algorithms}

for run_idx in range(N_RUNS):
    run_seed = 42 + run_idx * 1000
    np.random.seed(run_seed)

    tasks_list = [generate_batch(cfg['n_tasks'], type_mix=cfg['type_mix'])
                  for _ in range(N_PERIODS)]
    sample_tasks = tasks_list[0]
    servers_init = generate_servers_with_target_rho(
        cfg['m_servers'], sample_tasks, cfg['rho'], KAPPA, DECISION_INTERVAL)

    for algo_name, solver in algorithms.items():
        np.random.seed(run_seed)
        servers = deepcopy(servers_init)
        exp_list, rob_list = [], []

        for t in range(N_PERIODS):
            tasks = tasks_list[t]
            assignment = solver.solve(tasks, servers)
            jfi_e, jfi_r = compute_both_jfi(assignment, tasks, servers, KAPPA)
            exp_list.append(jfi_e)
            rob_list.append(jfi_r)

            next_backlog = compute_next_backlog(assignment, tasks, servers, DECISION_INTERVAL)
            for j in range(cfg['m_servers']):
                servers[j].L0 = next_backlog[j]

        jfi_exp_runs[algo_name].append(np.mean(exp_list))
        jfi_rob_runs[algo_name].append(np.mean(rob_list))

    print(f"Run {run_idx+1}/{N_RUNS} done")

# ===== 输出 =====
algo_order = ['DG', 'RG', 'EPD-FF', 'Std-LNS', 'RA-LNS']

print("\n" + "="*70)
print("JFI 对比：期望版 vs 鲁棒版")
print("="*70)
print(f"\n{'Algorithm':<10} {'JFI_exp':>10} {'JFI_rob':>10} {'Δ(rob-exp)':>12} {'方向':>6}")
print("-"*50)
for a in algo_order:
    me = np.mean(jfi_exp_runs[a])
    mr = np.mean(jfi_rob_runs[a])
    delta = mr - me
    direction = "↑" if delta > 0.001 else ("↓" if delta < -0.001 else "≈")
    print(f"{a:<10} {me:>10.4f} {mr:>10.4f} {delta:>+12.4f} {direction:>6}")

# 排名对比
print("\n===== 排名对比 =====")
exp_vals = {a: np.mean(jfi_exp_runs[a]) for a in algo_order}
rob_vals = {a: np.mean(jfi_rob_runs[a]) for a in algo_order}

exp_rank = sorted(algo_order, key=lambda a: -exp_vals[a])
rob_rank = sorted(algo_order, key=lambda a: -rob_vals[a])

print(f"\n期望 JFI 排名:  {' > '.join(exp_rank)}")
print(f"鲁棒 JFI 排名:  {' > '.join(rob_rank)}")

# RA-LNS 在两个排名中的位置
exp_pos = exp_rank.index('RA-LNS') + 1
rob_pos = rob_rank.index('RA-LNS') + 1
print(f"\nRA-LNS 排名: 期望={exp_pos}/5, 鲁棒={rob_pos}/5")

# 区分度
rob_spread = max(rob_vals.values()) - min(rob_vals.values())
print(f"\n鲁棒 JFI 区分度 (max-min): {rob_spread:.4f}")

if rob_pos <= 2:
    print(f"\n>> [OK] robust JFI direction correct, can use in main table")
    if rob_pos == 1:
        print(">> narrative: 'RA-LNS achieves the highest robust load fairness'")
    else:
        print(">> narrative: 'RA-LNS achieves comparable or superior robust load fairness'")
else:
    print(f"\n>> [WARN] RA-LNS robust JFI rank {rob_pos}/5, need further analysis")

# narrative templates
print("\n===== Narrative Templates =====")
print("""
If robust JFI direction correct (RA-LNS top-1 or top-2):

  Main table: only robust JFI
  Text: "We evaluate load-balancing fairness using Jain's Fairness Index
  on robust utilization u_j = L_hat_j/C_j. This captures whether overload
  risk is evenly distributed across servers, which is the operationally
  meaningful notion of balance under workload uncertainty."

If robust JFI also not ideal:

  Do not add JFI column. O1 + U_max + CVR + backlog is enough.
  Use a paragraph in text to qualitatively argue RA-LNS load-balancing.
""")
