"""Deterministic Greedy (DG) Solver"""
import numpy as np
from typing import List
from models.task import Task
from models.server import Server
from .base import BaseSolver


class DGSolver(BaseSolver):
    """Deterministic Greedy 算法

    仅基于期望负载的贪心调度，不考虑方差。
    选择当前利用率最小的服务器。
    """

    def solve(self, tasks: List[Task], servers: List[Server]) -> List[int]:
        n_tasks = len(tasks)
        n_servers = len(servers)

        current_load = np.array([s.L0 for s in servers])
        capacities = np.array([s.C for s in servers])

        assignment = []
        for i in range(n_tasks):
            utilization = current_load / capacities
            j_min = int(np.argmin(utilization))
            assignment.append(j_min)
            current_load[j_min] += tasks[i].mu

        return assignment
