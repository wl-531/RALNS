"""统一指标计算"""
import numpy as np
from typing import List, Dict
from models.task import Task
from models.server import Server
from config import EPS_DIV, EPS_TOL


def compute_metrics(assignment: List[int], tasks: List[Task],
                    servers: List[Server], kappa: float) -> Dict[str, float]:
    """统一计算所有指标

    Args:
        assignment: 任务分配方案
        tasks: 任务列表
        servers: 服务器列表
        kappa: 风险系数

    Returns:
        dict with keys:
            - feasible: bool，可行性
            - U_max: float，最大鲁棒利用率
            - O1: float，鲁棒 Makespan
            - R_sum: float，总风险密度
            - O2: float，负载不平衡度
            - L_hat: ndarray，鲁棒负载向量
            - Gap: ndarray，剩余容量向量
            - RD: ndarray，风险密度向量
    """
    n_tasks = len(tasks)
    m = len(servers)

    assert len(assignment) == n_tasks
    assert all(0 <= a < m for a in assignment)

    C = np.array([s.C for s in servers])
    L0 = np.array([s.L0 for s in servers])
    mu_sum = np.zeros(m)
    sigma_sq_sum = np.zeros(m)

    for i, j in enumerate(assignment):
        mu_sum[j] += tasks[i].mu
        sigma_sq_sum[j] += tasks[i].sigma ** 2

    sigma_j = np.sqrt(np.maximum(sigma_sq_sum, 0))
    L_hat = L0 + mu_sum + kappa * sigma_j
    Gap = C - L_hat
    RD = sigma_j / np.maximum(Gap, EPS_DIV)

    feasible = bool(np.all(Gap >= -EPS_TOL))
    U_max = float(np.max(L_hat / C))
    O1 = float(np.max(L_hat))
    R_sum = float(np.sum(RD))
    L_bar = np.mean(L_hat)
    O2 = float(np.sum((L_hat - L_bar) ** 2))

    # 新增指标
    exp_makespan = float(np.max(L0 + mu_sum))  # 期望 makespan（含 backlog）
    util_new = float(np.mean(mu_sum / C))  # 新任务利用率
    util_total = float(np.mean((L0 + mu_sum) / C))  # 总期望利用率（含 backlog）
    robust_util_total = float(np.mean(L_hat / C))  # 总鲁棒利用率（含 backlog）

    return {
        'feasible': feasible,
        'U_max': U_max,
        'O1': O1,
        'R_sum': R_sum,
        'O2': O2,
        'L_hat': L_hat,
        'Gap': Gap,
        'RD': RD,
        'exp_makespan': exp_makespan,
        'util_new': util_new,
        'util_total': util_total,
        'robust_util_total': robust_util_total,
    }
