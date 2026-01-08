"""Random-Destroy: RA-LNS 消融版本（去掉 RD 引导）

保留 destroy/repair 结构，但去掉 RD 引导：
- 将 j_hot = argmax(RD_j) 改为随机选择服务器
- 其他逻辑（repair、acceptance）保持不变
"""
import numpy as np
from typing import List
from models.task import Task
from models.server import Server
from .ra_lns import RALNSSolver, RALNSSolution
from config import KAPPA, T_MAX, PATIENCE, DESTROY_K


class RandomDestroySolver(RALNSSolver):
    """Random-Destroy Solver（消融 baseline）

    = RA-LNS with random destroy target
    将 RD 引导的热点选择改为随机选择
    """

    def __init__(self, kappa: float = KAPPA, patience: int = PATIENCE,
                 destroy_k: int = DESTROY_K, t_max: float = T_MAX):
        super().__init__(kappa=kappa, patience=patience,
                         destroy_k=destroy_k, t_max=t_max)

    def _risk_hedging_move(self, sol: RALNSSolution, tasks: List[Task]) -> bool:
        """Stage-1A: Micro Risk Hedging（随机服务器顺序）

        与 RA-LNS 相同的搜索范围，但使用随机服务器顺序（而非 RD 引导）
        """
        best_move = None
        best_psi = sol.Psi()
        best_rsum = sol.R_sum

        # 随机选择一个服务器作为 j_hot（而非 argmax RD）
        j_hot = np.random.randint(sol.m)
        victims = [i for i, j in enumerate(sol.assignment) if j == j_hot]
        if not victims:
            return False

        # 按 sigma 降序排列所有 victims（与 RA-LNS 一致）
        victim_sigmas = [(i, tasks[i].sigma) for i in victims]
        victim_sigmas.sort(key=lambda x: -x[1])

        from_j = j_hot

        # 遍历所有 victims（按 sigma 降序）
        for victim_idx, _ in victim_sigmas:
            victim_task = tasks[victim_idx]

            for to_j in range(sol.m):
                if to_j == from_j:
                    continue

                # Relocate
                sol.apply_move(victim_idx, victim_task, from_j, to_j)
                new_psi = sol.Psi()
                new_rsum = sol.R_sum
                if self._psi_better(new_psi, best_psi, new_rsum, best_rsum):
                    best_psi = new_psi
                    best_rsum = new_rsum
                    best_move = ('relocate', victim_idx, victim_task, from_j, to_j)
                sol.rollback_move(victim_idx, victim_task, from_j, to_j)

                # Swap
                swap_cands = [i for i, j in enumerate(sol.assignment) if j == to_j]
                if swap_cands:
                    swap_sigmas = [tasks[i].sigma for i in swap_cands]
                    swap_idx = swap_cands[int(np.argmin(swap_sigmas))]
                    swap_task = tasks[swap_idx]

                    sol.apply_move(victim_idx, victim_task, from_j, to_j)
                    sol.apply_move(swap_idx, swap_task, to_j, from_j)
                    new_psi = sol.Psi()
                    new_rsum = sol.R_sum
                    if self._psi_better(new_psi, best_psi, new_rsum, best_rsum):
                        best_psi = new_psi
                        best_rsum = new_rsum
                        best_move = ('swap', victim_idx, victim_task, from_j, to_j,
                                     swap_idx, swap_task)
                    sol.rollback_move(swap_idx, swap_task, to_j, from_j)
                    sol.rollback_move(victim_idx, victim_task, from_j, to_j)

        if best_move:
            if best_move[0] == 'relocate':
                _, vi, vt, fj, tj = best_move
                sol.apply_move(vi, vt, fj, tj)
            else:
                _, vi, vt, fj, tj, si, st = best_move
                sol.apply_move(vi, vt, fj, tj)
                sol.apply_move(si, st, tj, fj)
            return True
        return False

    def _risk_guided_lns(self, sol: RALNSSolution, tasks: List[Task]) -> bool:
        """Stage-1B: Macro Risk Rebalancing（随机服务器选择）

        与 RA-LNS 相同的逻辑，但使用随机服务器选择（而非 argmax RD）
        """
        from config import EPS_TOL, EPS_DIV
        backup_sol = sol.copy()

        # 随机选择一个服务器作为 j_hot
        j_hot = np.random.randint(sol.m)
        victims = [i for i, j in enumerate(sol.assignment) if j == j_hot]
        if len(victims) < self.destroy_k:
            return False

        # 只选择可移动的任务
        movable = []
        for i in victims:
            task = tasks[i]
            has_target = False
            for j_to in range(sol.m):
                if j_to == j_hot:
                    continue
                new_sigma_sq = sol.sigma_sq_sum[j_to] + task.sigma ** 2
                new_sigma = np.sqrt(max(new_sigma_sq, 0))
                new_mu = sol.mu_sum[j_to] + task.mu
                new_L_hat = sol.L0[j_to] + new_mu + self.kappa * new_sigma
                new_Gap = sol.C[j_to] - new_L_hat
                if new_Gap >= -EPS_TOL:
                    has_target = True
                    break
            if has_target:
                movable.append((i, tasks[i].sigma))

        if len(movable) < self.destroy_k:
            return False

        movable.sort(key=lambda x: x[1], reverse=True)
        destroy_tasks = [i for i, _ in movable[:self.destroy_k]]

        # Destroy
        for i in destroy_tasks:
            task = tasks[i]
            j = sol.assignment[i]
            sol.mu_sum[j] -= task.mu
            sol.sigma_sq_sum[j] -= task.sigma ** 2
            sol.assignment[i] = -1

        # Repair: 使用 min new_RR_j
        repair_order = sorted(destroy_tasks, key=lambda i: tasks[i].sigma, reverse=True)
        is_first = True

        for i in repair_order:
            task = tasks[i]

            best_j = None
            best_new_rr = np.inf
            fallback_j = None
            fallback_new_rr = np.inf

            for j in range(sol.m):
                new_sigma_sq_j = sol.sigma_sq_sum[j] + task.sigma ** 2
                new_sigma_j = np.sqrt(max(new_sigma_sq_j, 0))
                new_mu_j = sol.mu_sum[j] + task.mu
                new_L_hat_j = sol.L0[j] + new_mu_j + self.kappa * new_sigma_j
                new_Gap_j = sol.C[j] - new_L_hat_j

                if new_Gap_j < -EPS_TOL:
                    continue

                new_margin_j = sol.C[j] - (sol.L0[j] + new_mu_j)
                new_RR_j = new_sigma_j / max(new_margin_j, EPS_DIV)

                if is_first and j == j_hot:
                    if fallback_j is None or new_RR_j < fallback_new_rr:
                        fallback_j = j
                        fallback_new_rr = new_RR_j
                    continue

                if new_RR_j < best_new_rr:
                    best_new_rr = new_RR_j
                    best_j = j

            if best_j is None and fallback_j is not None:
                best_j = fallback_j

            if best_j is not None:
                sol.assignment[i] = best_j
                sol.mu_sum[best_j] += task.mu
                sol.sigma_sq_sum[best_j] += task.sigma ** 2
            else:
                j_min = int(np.argmin(sol.L_hat))
                sol.assignment[i] = j_min
                sol.mu_sum[j_min] += task.mu
                sol.sigma_sq_sum[j_min] += task.sigma ** 2

            is_first = False

        if self._better(sol, backup_sol):
            return True
        else:
            sol.mu_sum = backup_sol.mu_sum.copy()
            sol.sigma_sq_sum = backup_sol.sigma_sq_sum.copy()
            sol.assignment = backup_sol.assignment.copy()
            return False
