"""Robust Greedy (RG) / kappa-Greedy Solver"""
import numpy as np
from typing import List
from models.task import Task
from models.server import Server
from .base import BaseSolver
from config import KAPPA


class RGSolver(BaseSolver):
    """Robust Greedy (kappa-Greedy) 算法

    用鲁棒负载 L_hat = mu + kappa * sigma 代替期望负载决策。
    选择鲁棒利用率最小的服务器。
    """

    def __init__(self, kappa: float = KAPPA):
        self.kappa = kappa

    def solve(self, tasks: List[Task], servers: List[Server]) -> List[int]:
        n_tasks = len(tasks)
        n_servers = len(servers)

        mu_sum = np.array([s.L0 for s in servers])
        var_sum = np.zeros(n_servers)
        capacities = np.array([s.C for s in servers])

        assignment = []
        for i in range(n_tasks):
            mu_i = tasks[i].mu
            var_i = tasks[i].sigma ** 2

            new_mu = mu_sum + mu_i
            new_var = var_sum + var_i
            new_std = np.sqrt(new_var)
            new_robust = new_mu + self.kappa * new_std
            new_util = new_robust / capacities

            j_star = int(np.argmin(new_util))
            assignment.append(j_star)

            mu_sum[j_star] += mu_i
            var_sum[j_star] += var_i

        return assignment
