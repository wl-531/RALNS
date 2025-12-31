"""RA-LNS: Risk-Aware Large Neighborhood Search

对照论文伪代码实现，符合 algo.md 描述。
字典序：Psi(X) = (-feas, U_max, O1)，tie-break: R_sum
"""
import time
import numpy as np
from typing import List, Tuple
from models.task import Task
from models.server import Server
from .base import BaseSolver
from config import KAPPA, T_MAX, PATIENCE, DESTROY_K, EPS_TOL, EPS_CMP, EPS_DIV


class RALNSSolution:
    """解的表示（支持增量更新）"""

    def __init__(self, servers: List[Server], kappa: float):
        self.m = len(servers)
        self.kappa = kappa
        self.C = np.array([s.C for s in servers])
        self.L0 = np.array([s.L0 for s in servers])
        self.mu_sum = np.zeros(self.m)
        self.sigma_sq_sum = np.zeros(self.m)
        self.assignment = []

    @property
    def sigma_j(self) -> np.ndarray:
        """每台服务器的聚合标准差"""
        return np.sqrt(np.maximum(self.sigma_sq_sum, 0))

    @property
    def L_hat(self) -> np.ndarray:
        """鲁棒负载 L_hat_j = L0_j + mu_j + kappa * sigma_j"""
        return self.L0 + self.mu_sum + self.kappa * self.sigma_j

    @property
    def Gap(self) -> np.ndarray:
        """剩余鲁棒容量 Gap_j = C_j - L_hat_j"""
        return self.C - self.L_hat

    @property
    def RD(self) -> np.ndarray:
        """风险密度 RD_j = sigma_j / max(Gap_j, eps_div)"""
        return self.sigma_j / np.maximum(self.Gap, EPS_DIV)

    @property
    def U_max(self) -> float:
        """Level-1: 最大鲁棒利用率"""
        return float(np.max(self.L_hat / self.C))

    @property
    def O1(self) -> float:
        """Level-2: 鲁棒 Makespan"""
        return float(np.max(self.L_hat))

    @property
    def R_sum(self) -> float:
        """Tie-break: 总风险密度"""
        return float(np.sum(self.RD))

    def is_feasible(self) -> bool:
        """Level-0: 可行性检查（使用 eps_tol）"""
        return bool(np.all(self.Gap >= -EPS_TOL))

    def Psi(self) -> Tuple[int, float, float]:
        """3 层字典序向量：(-feas, U_max, O1)"""
        return (
            0 if self.is_feasible() else 1,  # -feas: 可行=0, 不可行=1
            self.U_max,
            self.O1
        )

    def apply_move(self, task_idx: int, task: Task, from_j: int, to_j: int):
        """应用移动操作"""
        if from_j is not None and from_j >= 0:
            self.mu_sum[from_j] -= task.mu
            self.sigma_sq_sum[from_j] -= task.sigma ** 2
        self.mu_sum[to_j] += task.mu
        self.sigma_sq_sum[to_j] += task.sigma ** 2
        self.assignment[task_idx] = to_j

    def rollback_move(self, task_idx: int, task: Task, from_j: int, to_j: int):
        """回滚移动操作"""
        self.mu_sum[to_j] -= task.mu
        self.sigma_sq_sum[to_j] -= task.sigma ** 2
        if from_j is not None and from_j >= 0:
            self.mu_sum[from_j] += task.mu
            self.sigma_sq_sum[from_j] += task.sigma ** 2
        self.assignment[task_idx] = from_j if from_j is not None else -1

    def copy(self) -> 'RALNSSolution':
        """深拷贝"""
        new_sol = RALNSSolution.__new__(RALNSSolution)
        new_sol.m = self.m
        new_sol.kappa = self.kappa
        new_sol.C = self.C.copy()
        new_sol.L0 = self.L0.copy()
        new_sol.mu_sum = self.mu_sum.copy()
        new_sol.sigma_sq_sum = self.sigma_sq_sum.copy()
        new_sol.assignment = self.assignment.copy()
        return new_sol


class RALNSSolver(BaseSolver):
    """RA-LNS Solver（符合论文伪代码）"""

    def __init__(self, kappa: float = KAPPA, patience: int = PATIENCE,
                 destroy_k: int = DESTROY_K, t_max: float = T_MAX,
                 eps_tol: float = EPS_TOL, eps_cmp: float = EPS_CMP,
                 eps_div: float = EPS_DIV):
        self.kappa = kappa
        self.patience = patience
        self.destroy_k = destroy_k
        self.t_max = t_max
        self.eps_tol = eps_tol
        self.eps_cmp = eps_cmp
        self.eps_div = eps_div

    def solve(self, tasks: List[Task], servers: List[Server]) -> List[int]:
        """主求解入口"""
        start = time.perf_counter()

        # Phase 0: Risk-First Construction
        sol, fallback_count = self._risk_first_construction(tasks, servers)

        # 初始化全局最优
        best = sol.copy() if sol.is_feasible() else None
        best_rsum = sol.R_sum if best else float('inf')

        stagnation = 0
        iteration = 0

        # Phase 1: Risk-Density-Guided Descent
        while time.perf_counter() - start < self.t_max:
            if stagnation < self.patience:
                # Stage-1A: Micro Risk Hedging
                improved = self._risk_hedging_move(sol, tasks)
            else:
                # Stage-1B: Macro Risk Rebalancing
                improved = self._risk_guided_lns(sol, tasks)
                stagnation = 0

            if improved:
                stagnation = 0
                if sol.is_feasible():
                    if best is None or self._better(sol, best):
                        best = sol.copy()
                        best_rsum = sol.R_sum
            else:
                stagnation += 1

            iteration += 1
            if iteration > 1000:
                break

        # 返回结果
        result = best if best else sol
        return result.assignment.copy()

    def _risk_first_construction(self, tasks: List[Task],
                                  servers: List[Server]) -> Tuple[RALNSSolution, int]:
        """Phase 0: Risk-First Construction

        按 delta_i = mu_i + kappa * sigma_i 降序贪心分配。
        选择使 score = Gap'_j / max(sigma_i, eps_div) 最大的可行服务器。
        """
        sol = RALNSSolution(servers, self.kappa)
        fallback_count = 0
        n_tasks = len(tasks)

        # 按 delta_i 降序排序
        deltas = [(i, tasks[i].get_delta(self.kappa)) for i in range(n_tasks)]
        sorted_indices = [i for i, _ in sorted(deltas, key=lambda x: -x[1])]
        sol.assignment = [-1] * n_tasks

        for i in sorted_indices:
            task = tasks[i]

            # 计算分配到每台服务器后的状态
            new_sigma_sq = sol.sigma_sq_sum + task.sigma ** 2
            new_sigma = np.sqrt(np.maximum(new_sigma_sq, 0))
            new_mu = sol.mu_sum + task.mu
            new_L_hat = sol.L0 + new_mu + self.kappa * new_sigma
            new_Gap = sol.C - new_L_hat

            # 计算 score = Gap'_j / max(sigma_i, eps_div)
            task_sigma = max(task.sigma, self.eps_div)
            scores = new_Gap / task_sigma

            # 选择可行且 score 最大的服务器
            best_j = None
            best_score = -np.inf
            for j in range(sol.m):
                if new_Gap[j] >= -self.eps_tol and scores[j] > best_score:
                    best_score = scores[j]
                    best_j = j

            if best_j is not None:
                sol.assignment[i] = best_j
                sol.mu_sum[best_j] += task.mu
                sol.sigma_sq_sum[best_j] += task.sigma ** 2
            else:
                # Fallback: 分配到 L_hat 最小的服务器
                fallback_count += 1
                j_min = int(np.argmin(sol.L_hat))
                sol.assignment[i] = j_min
                sol.mu_sum[j_min] += task.mu
                sol.sigma_sq_sum[j_min] += task.sigma ** 2

        return sol, fallback_count

    def _risk_hedging_move(self, sol: RALNSSolution, tasks: List[Task]) -> bool:
        """Stage-1A: Micro Risk Hedging

        1. 找到风险热点 j_hot = argmax RD_j
        2. 从 j_hot 选择 sigma 最大的任务作为 victim
        3. 尝试 Relocate 和 Swap 操作
        """
        # 找到风险热点
        j_hot = int(np.argmax(sol.RD))
        victims = [i for i, j in enumerate(sol.assignment) if j == j_hot]
        if not victims:
            return False

        # 选择 sigma 最大的任务
        victim_sigmas = [tasks[i].sigma for i in victims]
        victim_idx = victims[int(np.argmax(victim_sigmas))]
        victim_task = tasks[victim_idx]
        from_j = sol.assignment[victim_idx]

        best_move = None
        best_psi = sol.Psi()
        best_rsum = sol.R_sum

        for to_j in range(sol.m):
            if to_j == from_j:
                continue

            # Relocate: 将 victim 从 from_j 移动到 to_j
            sol.apply_move(victim_idx, victim_task, from_j, to_j)
            new_psi = sol.Psi()
            new_rsum = sol.R_sum
            if self._psi_better(new_psi, best_psi, new_rsum, best_rsum):
                best_psi = new_psi
                best_rsum = new_rsum
                best_move = ('relocate', victim_idx, victim_task, from_j, to_j)
            sol.rollback_move(victim_idx, victim_task, from_j, to_j)

            # Swap: 与 to_j 上 sigma 最小的任务交换
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

        # 应用最佳移动
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
        """Stage-1B: Macro Risk Rebalancing

        1. Destroy: 从 RD 最大的服务器移除 top-k 高方差任务
        2. Repair: 按方差降序重新插入，选择使 ΔRD 最小的服务器
        """
        # 找到风险热点
        j_hot = int(np.argmax(sol.RD))
        victims = [i for i, j in enumerate(sol.assignment) if j == j_hot]
        if len(victims) < self.destroy_k:
            return False

        # 选择 top-k 高方差任务
        victim_sigmas = [(i, tasks[i].sigma) for i in victims]
        victim_sigmas.sort(key=lambda x: x[1], reverse=True)
        destroy_tasks = [i for i, _ in victim_sigmas[:self.destroy_k]]

        # 备份当前解
        backup_sol = sol.copy()

        # Destroy: 移除任务
        for i in destroy_tasks:
            task = tasks[i]
            j = sol.assignment[i]
            sol.mu_sum[j] -= task.mu
            sol.sigma_sq_sum[j] -= task.sigma ** 2
            sol.assignment[i] = -1

        # Repair: 按方差降序重新插入
        repair_order = sorted(destroy_tasks, key=lambda i: tasks[i].sigma, reverse=True)
        for i in repair_order:
            task = tasks[i]

            # 计算分配到每台服务器后的 ΔRD
            new_sigma_sq = sol.sigma_sq_sum + task.sigma ** 2
            new_sigma = np.sqrt(np.maximum(new_sigma_sq, 0))
            new_mu = sol.mu_sum + task.mu
            new_L_hat = sol.L0 + new_mu + self.kappa * new_sigma
            new_Gap = sol.C - new_L_hat

            best_j = None
            best_delta_rd = np.inf
            for j in range(sol.m):
                if new_Gap[j] >= -self.eps_tol:
                    rd_before = sol.sigma_j[j] / max(sol.Gap[j], self.eps_div)
                    rd_after = new_sigma[j] / max(new_Gap[j], self.eps_div)
                    delta_rd = rd_after - rd_before
                    if delta_rd < best_delta_rd:
                        best_delta_rd = delta_rd
                        best_j = j

            if best_j is not None:
                sol.assignment[i] = best_j
                sol.mu_sum[best_j] += task.mu
                sol.sigma_sq_sum[best_j] += task.sigma ** 2
            else:
                # Fallback
                j_min = int(np.argmin(sol.L_hat))
                sol.assignment[i] = j_min
                sol.mu_sum[j_min] += task.mu
                sol.sigma_sq_sum[j_min] += task.sigma ** 2

        # 检查是否改进
        if self._better(sol, backup_sol):
            return True
        else:
            # 回滚
            sol.mu_sum = backup_sol.mu_sum.copy()
            sol.sigma_sq_sum = backup_sol.sigma_sq_sum.copy()
            sol.assignment = backup_sol.assignment.copy()
            return False

    def _better(self, sol1: RALNSSolution, sol2: RALNSSolution) -> bool:
        """比较两个解：sol1 是否优于 sol2"""
        return self._psi_better(sol1.Psi(), sol2.Psi(), sol1.R_sum, sol2.R_sum)

    def _psi_better(self, psi1: Tuple, psi2: Tuple, r_sum1: float, r_sum2: float) -> bool:
        """3 层字典序比较 + R_sum tie-break

        Psi = (-feas, U_max, O1)
        越小越好
        """
        # Level-0: feas（严格比较）
        if psi1[0] != psi2[0]:
            return psi1[0] < psi2[0]

        # Level-1, Level-2: 浮点比较
        for v1, v2 in zip(psi1[1:], psi2[1:]):
            if v1 < v2 - self.eps_cmp:
                return True
            if v1 > v2 + self.eps_cmp:
                return False

        # Tie-break: R_sum 最小者胜
        return r_sum1 < r_sum2 - self.eps_cmp
