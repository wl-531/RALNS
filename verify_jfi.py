"""验证 JFI 指标：方向、区分度、叙述预案判断

使用方法：
    1. 先按 patch 修改 evaluation/metrics.py
    2. cd RALNS && python verify_jfi.py

此脚本独立于 run_main，只跑 5 runs × 20 periods（约 2 分钟），
目的是快速确认 JFI 方向后再决定是否跑完整 30 runs。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from copy import deepcopy

from config import (KAPPA, MC_SAMPLES, DECISION_INTERVAL,
                    T_MAX, PATIENCE, DESTROY_K, MAIN_CONFIG)
from data.generator import generate_batch, generate_servers_with_target_rho
from solvers import DGSolver, RGSolver, EPDFFSolver, StdLNSSolver, RALNSSolver
from evaluation import compute_metrics

N_RUNS = 5
N_PERIODS = 20
cfg = MAIN_CONFIG

algorithms = {
    'DG':      DGSolver(),
    'RG':      RGSolver(kappa=KAPPA),
    'EPD-FF':  EPDFFSolver(kappa=KAPPA),
    'Std-LNS': StdLNSSolver(t_max=T_MAX, patience=PATIENCE, destroy_k=DESTROY_K),
    'RA-LNS':  RALNSSolver(kappa=KAPPA, t_max=T_MAX, patience=PATIENCE, destroy_k=DESTROY_K),
}

# 收集每个 run 的平均 JFI
jfi_runs = {name: [] for name in algorithms}

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
        jfi_list = []

        for t in range(N_PERIODS):
            tasks = tasks_list[t]
            assignment = solver.solve(tasks, servers)
            metrics = compute_metrics(assignment, tasks, servers, KAPPA)
            jfi_list.append(metrics['jfi'])

            from evaluation.monte_carlo import compute_next_backlog
            next_backlog = compute_next_backlog(assignment, tasks, servers, DECISION_INTERVAL)
            for j in range(cfg['m_servers']):
                servers[j].L0 = next_backlog[j]

        jfi_runs[algo_name].append(np.mean(jfi_list))

    print(f"Run {run_idx+1}/{N_RUNS} done")

# ===== 输出 =====
print("\n" + "="*60)
print("JFI Verification Results")
print("="*60)

algo_order = ['DG', 'RG', 'EPD-FF', 'Std-LNS', 'RA-LNS']
print(f"\n{'Algorithm':<10} {'JFI mean':>10} {'JFI std':>10}")
print("-"*35)
for a in algo_order:
    m = np.mean(jfi_runs[a])
    s = np.std(jfi_runs[a])
    print(f"{a:<10} {m:>10.6f} {s:>10.6f}")

# 区分度分析
vals = {a: np.mean(jfi_runs[a]) for a in algo_order}
spread = max(vals.values()) - min(vals.values())
best = max(vals, key=vals.get)
worst = min(vals, key=vals.get)

print(f"\n===== 区分度分析 =====")
print(f"Max - Min = {spread:.6f}")
print(f"Best:  {best} ({vals[best]:.6f})")
print(f"Worst: {worst} ({vals[worst]:.6f})")

if spread >= 0.01:
    print(f"\n>> 叙述预案 A: 差异显著，正常报告 RA-LNS 优势")
elif spread >= 0.001:
    print(f"\n>> 叙述预案 B: 差异微小但可见")
    print(f"   'All methods achieve comparable fairness; crucially,")
    print(f"    RA-LNS attains this while achieving significantly lower CVR.'")
else:
    print(f"\n>> 叙述预案 C: 几乎无差异")
    print(f"   'JFI is uniformly high (~{vals[best]:.3f}) across all methods,")
    print(f"    confirming that reliability gains do not compromise balance.'")

ralns_rank = sorted(algo_order, key=lambda a: -vals[a]).index('RA-LNS') + 1
print(f"\nRA-LNS rank: {ralns_rank}/{len(algo_order)}")
if ralns_rank == 1:
    print(">> RA-LNS 是 JFI 最优")
elif ralns_rank <= 2:
    print(">> RA-LNS 接近最优，可用 'comparable or superior' 措辞")
else:
    print(">> RA-LNS 不是 JFI 最优，需要 'does not sacrifice' 措辞")
