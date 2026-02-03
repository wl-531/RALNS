"""EPD-FF: EPD-style First-Fit

Adapted from Li et al. JSAC'23 (ref11).
结构性复用其"排序+First-fit"调度骨架。
"""
import numpy as np
from typing import List
from models.task import Task
from models.server import Server
from .base import BaseSolver
from config import KAPPA, EPS_TOL, EPS_DIV


class EPDFFSolver(BaseSolver):
    """EPD-FF Solver

    1. 按 Δ_i = μ_i + κσ_i 升序排序任务（小任务优先）
    2. First-to-fit：按服务器 index 顺序扫描，分配到第一个可行服务器
    3. Fallback：若无可行服务器，分配到分配后 L̂_j^new 最小的服务器
    """

    def __init__(self, kappa: float = KAPPA, eps_tol: float = EPS_TOL, eps_div: float = EPS_DIV):
        self.kappa = kappa
        self.eps_tol = eps_tol
        self.eps_div = eps_div
        self.fallback_count = 0

    def solve(self, tasks: List[Task], servers: List[Server]) -> List[int]:
        n = len(tasks)
        m = len(servers)
        self.fallback_count = 0

        # 初始化服务器状态
        mu_sum = np.zeros(m)
        var_sum = np.zeros(m)
        C = np.array([s.C for s in servers])
        L0 = np.array([s.L0 for s in servers])

        assignment = [-1] * n

        # Stage 1: 按 Δ_i 升序排序（Python sorted 是稳定排序）
        task_deltas = [(i, tasks[i].mu + self.kappa * tasks[i].sigma) for i in range(n)]
        sorted_tasks = sorted(task_deltas, key=lambda x: x[1])

        # Stage 2: First-to-fit by server index
        for task_idx, _ in sorted_tasks:
            task = tasks[task_idx]
            assigned = False

            # 按 server index 顺序扫描（0 到 m-1）
            for j in range(m):
                new_mu = mu_sum[j] + task.mu
                new_var = var_sum[j] + task.sigma ** 2
                new_sigma = np.sqrt(new_var)
                new_L_hat = L0[j] + new_mu + self.kappa * new_sigma

                # Robust capacity check（与 RG 一致）
                if new_L_hat <= C[j] + self.eps_tol:
                    assignment[task_idx] = j
                    mu_sum[j] = new_mu
                    var_sum[j] = new_var
                    assigned = True
                    break

            # Fallback: 分配到分配后 L̂_j^new 最小的服务器
            if not assigned:
                self.fallback_count += 1
                best_j = -1
                best_L_hat_new = float('inf')

                for j in range(m):
                    new_mu = mu_sum[j] + task.mu
                    new_var = var_sum[j] + task.sigma ** 2
                    new_sigma = np.sqrt(new_var)
                    new_L_hat = L0[j] + new_mu + self.kappa * new_sigma

                    if new_L_hat < best_L_hat_new:
                        best_L_hat_new = new_L_hat
                        best_j = j

                assignment[task_idx] = best_j
                mu_sum[best_j] += task.mu
                var_sum[best_j] += task.sigma ** 2

        return assignment

    def get_fallback_count(self) -> int:
        """返回上次 solve 调用的 fallback 次数"""
        return self.fallback_count
