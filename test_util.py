"""计算不同 rho 下的期望利用率"""
import numpy as np
from config import KAPPA, DECISION_INTERVAL, MAIN_CONFIG
from data.generator import generate_batch, generate_servers_with_target_rho

np.random.seed(42)
cfg = MAIN_CONFIG.copy()

print("rho\tE[U]\tU_robust")
print("-" * 32)

for rho in [0.85, 0.90, 0.92, 0.95]:
    cfg['rho'] = rho
    tasks = generate_batch(cfg['n_tasks'], type_mix=cfg['type_mix'])
    servers = generate_servers_with_target_rho(cfg['m_servers'], tasks, rho, KAPPA, DECISION_INTERVAL)

    total_mu = sum(t.mu for t in tasks)
    total_var = sum(t.sigma ** 2 for t in tasks)
    total_sigma = np.sqrt(total_var)
    total_C = sum(s.C for s in servers)

    # 期望利用率 = sum(mu) / sum(C)
    E_U = total_mu / total_C
    # 鲁棒利用率 (近似，假设均匀分配)
    U_robust = (total_mu + KAPPA * total_sigma) / total_C

    print(f"{rho:.2f}\t{E_U:.2%}\t{U_robust:.2%}")
