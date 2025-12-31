"""Monte Carlo 验证"""
import numpy as np
from typing import List, Tuple
from models.task import Task
from models.server import Server
from config import MC_SAMPLES, DECISION_INTERVAL


def monte_carlo_verify(assignment: List[int], tasks: List[Task],
                       servers: List[Server],
                       n_samples: int = MC_SAMPLES) -> Tuple[float, float, float]:
    """Monte Carlo 验证单周期的违约情况

    Args:
        assignment: 任务分配方案
        tasks: 任务列表
        servers: 服务器列表
        n_samples: 采样次数

    Returns:
        system_cvr: float，系统级违约率（任一服务器过载的频率）
        per_server_vr: float，平均单服务器违约率
        avg_excess: float，平均超额负载（期望值）
    """
    m = len(servers)

    mu = np.array([t.mu for t in tasks])
    sigma = np.array([t.sigma for t in tasks])
    C = np.array([s.C for s in servers])
    L0 = np.array([s.L0 for s in servers])

    system_violations = 0
    server_violations = 0
    total_excess = 0.0

    for _ in range(n_samples):
        # 采样实际工作量（非负截断）
        actual_workload = np.maximum(0, np.random.normal(mu, sigma))

        # 计算每台服务器的实际负载
        actual_load = L0.copy()
        for i, j in enumerate(assignment):
            actual_load[j] += actual_workload[i]

        # 检查过载
        overloaded = actual_load > C

        # 系统级：任一服务器过载
        if np.any(overloaded):
            system_violations += 1

        # 服务器级：统计所有过载服务器
        server_violations += np.sum(overloaded)

        # 超额负载
        excess = np.sum(np.maximum(0, actual_load - C))
        total_excess += excess

    system_cvr = system_violations / n_samples
    per_server_vr = server_violations / (m * n_samples)
    avg_excess = total_excess / n_samples

    return system_cvr, per_server_vr, avg_excess


def compute_next_backlog(assignment: List[int], tasks: List[Task],
                         servers: List[Server],
                         decision_interval: float = DECISION_INTERVAL) -> np.ndarray:
    """计算下一周期的 backlog（单次采样，模拟真实运行）

    Args:
        assignment: 任务分配方案
        tasks: 任务列表
        servers: 服务器列表
        decision_interval: 决策周期（秒）

    Returns:
        next_backlog: ndarray，shape (m,)
    """
    m = len(servers)

    # 采样实际工作量（非负截断）
    actual_workload = np.array([
        max(0, np.random.normal(t.mu, t.sigma)) for t in tasks
    ])

    # 计算每台服务器的实际负载
    actual_load = np.array([s.L0 for s in servers])
    for i, j in enumerate(assignment):
        actual_load[j] += actual_workload[i]

    # 计算处理量和残留
    processed = np.array([s.f * decision_interval for s in servers])
    next_backlog = np.maximum(0, actual_load - processed)

    return next_backlog
