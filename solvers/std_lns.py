"""Std-LNS: Standard (Risk-Neutral) Large Neighborhood Search

风险中性 LNS 基线，用于证明"搜索 ≠ 鲁棒"：
- 负载代理：仅用期望负载 μ_j，不含 κσ_j
- 目标函数：min max_j μ_j（期望 makespan），单目标，无字典序
- Destroy 策略：选择期望负载最大的服务器（非 RD 热点）
- Repair 策略：选择插入后期望负载最小的服务器
"""
import time
import numpy as np
from typing import List
from models.task import Task
from models.server import Server
from .base import BaseSolver
from config import T_MAX, PATIENCE, DESTROY_K, EPS_TOL


class StdLNSSolution:
    """风险中性解的表示"""

    def __init__(self, servers: List[Server]):
        self.m = len(servers)
        self.C = np.array([s.C for s in servers])
        self.L0 = np.array([s.L0 for s in servers])
        self.mu_sum = np.zeros(self.m)
        self.assignment = []

    @property
    def L_exp(self) -> np.ndarray:
        """期望负载 L_exp_j = L0_j + mu_j"""
        return self.L0 + self.mu_sum

    @property
    def O1(self) -> float:
        """期望 makespan"""
        return float(np.max(self.L_exp))

    def is_feasible(self) -> bool:
        """期望容量检查"""
        return bool(np.all(self.L_exp <= self.C + EPS_TOL))

    def apply_move(self, task_idx: int, task: Task, from_j: int, to_j: int):
        """应用移动操作"""
        if from_j is not None and from_j >= 0:
            self.mu_sum[from_j] -= task.mu
        self.mu_sum[to_j] += task.mu
        self.assignment[task_idx] = to_j

    def rollback_move(self, task_idx: int, task: Task, from_j: int, to_j: int):
        """回滚移动操作"""
        self.mu_sum[to_j] -= task.mu
        if from_j is not None and from_j >= 0:
            self.mu_sum[from_j] += task.mu
        self.assignment[task_idx] = from_j if from_j is not None else -1

    def copy(self) -> 'StdLNSSolution':
        """深拷贝"""
        new_sol = StdLNSSolution.__new__(StdLNSSolution)
        new_sol.m = self.m
        new_sol.C = self.C.copy()
        new_sol.L0 = self.L0.copy()
        new_sol.mu_sum = self.mu_sum.copy()
        new_sol.assignment = self.assignment.copy()
        return new_sol


class StdLNSSolver(BaseSolver):
    """Std-LNS Solver（风险中性 LNS）"""

    def __init__(self, patience: int = PATIENCE, destroy_k: int = DESTROY_K,
                 t_max: float = T_MAX, eps_tol: float = EPS_TOL):
        self.patience = patience
        self.destroy_k = destroy_k
        self.t_max = t_max
        self.eps_tol = eps_tol

    def solve(self, tasks: List[Task], servers: List[Server]) -> List[int]:
        """主求解入口"""
        start = time.perf_counter()

        # Phase 0: Greedy Construction（按期望负载）
        sol = self._greedy_construction(tasks, servers)

        best = sol.copy() if sol.is_feasible() else None
        best_obj = sol.O1 if best else float('inf')

        stagnation = 0
        iteration = 0

        # Phase 1: LNS Search
        while time.perf_counter() - start < self.t_max:
            if stagnation < self.patience:
                improved = self._relocate_move(sol, tasks)
            else:
                improved = self._lns_iteration(sol, tasks)
                stagnation = 0

            if improved:
                stagnation = 0
                if sol.is_feasible() and sol.O1 < best_obj:
                    best = sol.copy()
                    best_obj = sol.O1
            else:
                stagnation += 1

            iteration += 1
            if iteration > 1000:
                break

        result = best if best else sol
        return result.assignment.copy()

    def _greedy_construction(self, tasks: List[Task],
                              servers: List[Server]) -> StdLNSSolution:
        """Phase 0: Greedy Construction（按期望负载贪心）"""
        sol = StdLNSSolution(servers)
        n_tasks = len(tasks)

        # 按 mu 降序排序
        sorted_indices = sorted(range(n_tasks), key=lambda i: -tasks[i].mu)
        sol.assignment = [-1] * n_tasks

        for i in sorted_indices:
            task = tasks[i]
            # 选择使期望 makespan 最小的服务器
            new_L_exp = sol.L_exp + task.mu
            best_j = int(np.argmin(new_L_exp))
            sol.assignment[i] = best_j
            sol.mu_sum[best_j] += task.mu

        return sol

    def _relocate_move(self, sol: StdLNSSolution, tasks: List[Task]) -> bool:
        """尝试 relocate 操作（选择期望负载最大服务器的任务）"""
        # 找到期望负载最大的服务器
        j_max = int(np.argmax(sol.L_exp))
        victims = [i for i, j in enumerate(sol.assignment) if j == j_max]
        if not victims:
            return False

        # 选择 mu 最大的任务
        victim_idx = max(victims, key=lambda i: tasks[i].mu)
        victim_task = tasks[victim_idx]
        from_j = sol.assignment[victim_idx]

        best_move = None
        best_obj = sol.O1

        for to_j in range(sol.m):
            if to_j == from_j:
                continue

            sol.apply_move(victim_idx, victim_task, from_j, to_j)
            new_obj = sol.O1
            if new_obj < best_obj:
                best_obj = new_obj
                best_move = (victim_idx, victim_task, from_j, to_j)
            sol.rollback_move(victim_idx, victim_task, from_j, to_j)

        if best_move:
            vi, vt, fj, tj = best_move
            sol.apply_move(vi, vt, fj, tj)
            return True
        return False

    def _lns_iteration(self, sol: StdLNSSolution, tasks: List[Task]) -> bool:
        """LNS 迭代：从期望负载最大的服务器 destroy，然后 repair"""
        # 找到期望负载最大的服务器
        j_max = int(np.argmax(sol.L_exp))
        victims = [i for i, j in enumerate(sol.assignment) if j == j_max]
        if len(victims) < self.destroy_k:
            return False

        # 选择 top-k mu 最大的任务
        victim_mus = sorted([(i, tasks[i].mu) for i in victims],
                            key=lambda x: -x[1])
        destroy_tasks = [i for i, _ in victim_mus[:self.destroy_k]]

        backup_sol = sol.copy()

        # Destroy
        for i in destroy_tasks:
            task = tasks[i]
            j = sol.assignment[i]
            sol.mu_sum[j] -= task.mu
            sol.assignment[i] = -1

        # Repair：按 mu 降序重新插入，选择使期望负载最小的服务器
        repair_order = sorted(destroy_tasks, key=lambda i: -tasks[i].mu)
        for i in repair_order:
            task = tasks[i]
            new_L_exp = sol.L_exp + task.mu
            best_j = int(np.argmin(new_L_exp))
            sol.assignment[i] = best_j
            sol.mu_sum[best_j] += task.mu

        # 检查是否改进
        if sol.O1 < backup_sol.O1:
            return True
        else:
            sol.mu_sum = backup_sol.mu_sum.copy()
            sol.assignment = backup_sol.assignment.copy()
            return False
