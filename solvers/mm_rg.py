"""MM-RG: MinMax Robust Greedy 调度算法

与 RG (kappa_greedy) 的区别:
- RG:    argmin_j [ L̂_j / C_j ] — 局部视角，最小化目标服务器利用率
- MM-RG: argmin_j [ max_k L̂_k ] — 全局视角，最小化系统 makespan
"""
import numpy as np
from typing import List
from models.task import Task
from models.server import Server
from .base import BaseSolver
from config import KAPPA


def minmax_robust_greedy(tasks: List[Task], servers: List[Server], kappa: float) -> List[int]:
    """MinMax Robust Greedy 调度算法

    对于每个任务 i:
        j* = argmin_j [ max_{k∈M} L̂_k (after assigning i to j) ]

    Args:
        tasks: 任务列表，每个任务有 .mu 和 .sigma 属性
        servers: 服务器列表，每个服务器有 .L0, .C 属性
        kappa: 风险系数

    Returns:
        assignment: List[int], assignment[i] = j 表示任务 i 分配到服务器 j
    """
    n = len(tasks)
    m = len(servers)

    # 初始化服务器状态
    L0 = np.array([s.L0 for s in servers])
    mu_sum = np.zeros(m)
    var_sum = np.zeros(m)

    assignment = []

    for i in range(n):
        task = tasks[i]
        best_j = 0
        best_makespan = np.inf

        for j in range(m):
            # 假设将任务 i 分配到服务器 j
            new_mu_sum = mu_sum.copy()
            new_var_sum = var_sum.copy()
            new_mu_sum[j] += task.mu
            new_var_sum[j] += task.sigma ** 2

            # 计算所有服务器的鲁棒负载
            L_hat = L0 + new_mu_sum + kappa * np.sqrt(np.maximum(new_var_sum, 0))

            # 全局 makespan = max_k L̂_k
            makespan = np.max(L_hat)

            if makespan < best_makespan:
                best_makespan = makespan
                best_j = j

        # 应用最佳分配
        assignment.append(best_j)
        mu_sum[best_j] += task.mu
        var_sum[best_j] += task.sigma ** 2

    return assignment


class MMRGSolver(BaseSolver):
    """MM-RG Solver 封装类"""

    def __init__(self, kappa: float = KAPPA):
        self.kappa = kappa

    def solve(self, tasks: List[Task], servers: List[Server]) -> List[int]:
        return minmax_robust_greedy(tasks, servers, self.kappa)


if __name__ == "__main__":
    from models.task import Task
    from models.server import Server

    tasks = [Task(mu=50, sigma=15) for _ in range(20)]
    servers = [Server(f=100, C=500, L0=0) for _ in range(5)]

    assignment = minmax_robust_greedy(tasks, servers, kappa=2.38)
    print(f"Assignment: {assignment}")
    print(f"Tasks per server: {[assignment.count(j) for j in range(5)]}")

    # 验证负载均衡
    mu_sum = np.zeros(5)
    var_sum = np.zeros(5)
    for i, j in enumerate(assignment):
        mu_sum[j] += tasks[i].mu
        var_sum[j] += tasks[i].sigma ** 2
    L_hat = mu_sum + 2.38 * np.sqrt(var_sum)
    print(f"Robust loads: {L_hat.round(1)}")
    print(f"Max robust load: {L_hat.max():.1f}")
