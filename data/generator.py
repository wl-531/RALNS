"""数据生成器"""
import numpy as np
from typing import List, Tuple

from models.task import Task
from models.server import Server
from config import TYPE_A_MU_RANGE, TYPE_A_CV_RANGE, TYPE_B_MU_RANGE, TYPE_B_CV_RANGE


def generate_tasks(n_tasks: int, mode: str = "bimodal") -> List[Task]:
    """生成任务列表

    Args:
        n_tasks: 任务数量
        mode: 生成模式
            - "bimodal": 双峰模式（默认）
              * 40% Type A: 陷阱任务（低μ高σ）
              * 60% Type B: 稳定任务（高μ低σ）

    Returns:
        List[Task]
    """
    tasks = []

    if mode == "bimodal":
        # Type A: 陷阱任务（低 μ 高 σ）
        n_type_A = int(n_tasks * 0.4)
        for _ in range(n_type_A):
            mu = np.random.uniform(*TYPE_A_MU_RANGE)
            cv = np.random.uniform(*TYPE_A_CV_RANGE)
            sigma = mu * cv
            tasks.append(Task(mu=mu, sigma=sigma))

        # Type B: 稳定任务（高 μ 低 σ）
        n_type_B = n_tasks - n_type_A
        for _ in range(n_type_B):
            mu = np.random.uniform(*TYPE_B_MU_RANGE)
            cv = np.random.uniform(*TYPE_B_CV_RANGE)
            sigma = mu * cv
            tasks.append(Task(mu=mu, sigma=sigma))

        np.random.shuffle(tasks)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return tasks


def generate_batch(n_tasks: int, type_mix: List[float] = None) -> List[Task]:
    """按比例混合生成任务批次

    Args:
        n_tasks: 任务数量
        type_mix: [Type A 比例, Type B 比例, Type C 比例]
                  默认 [0.4, 0.6, 0.0]（兼容 bimodal）

    Returns:
        List[Task]
    """
    if type_mix is None:
        type_mix = [0.4, 0.6, 0.0]

    tasks = []
    n_A = int(n_tasks * type_mix[0])
    n_B = int(n_tasks * type_mix[1])
    n_C = n_tasks - n_A - n_B

    # Type A: 陷阱任务
    for _ in range(n_A):
        mu = np.random.uniform(*TYPE_A_MU_RANGE)
        cv = np.random.uniform(*TYPE_A_CV_RANGE)
        tasks.append(Task(mu=mu, sigma=mu * cv))

    # Type B: 稳定任务
    for _ in range(n_B):
        mu = np.random.uniform(*TYPE_B_MU_RANGE)
        cv = np.random.uniform(*TYPE_B_CV_RANGE)
        tasks.append(Task(mu=mu, sigma=mu * cv))

    # Type C: 通用任务（暂时与 Type B 相同）
    for _ in range(n_C):
        mu = np.random.uniform(50, 80)
        cv = np.random.uniform(0.20, 0.40)
        tasks.append(Task(mu=mu, sigma=mu * cv))

    np.random.shuffle(tasks)
    return tasks


def generate_servers_with_target_rho(n_servers: int, total_expected_load: float,
                                      target_rho: float,
                                      decision_interval: float = 30.0) -> List[Server]:
    """根据目标 rho 反算服务器容量

    Args:
        n_servers: 服务器数量
        total_expected_load: 总期望负载 = sum(mu_i)
        target_rho: 目标期望负载率
        decision_interval: 决策周期（秒）

    Returns:
        List[Server]
    """
    total_capacity = total_expected_load / target_rho
    capacity_per_server = total_capacity / n_servers

    # 异构因子（归一化，确保总和精确）
    factors = np.random.uniform(0.9, 1.1, n_servers)
    factors = factors / factors.sum() * n_servers

    servers = []
    for j in range(n_servers):
        C = capacity_per_server * factors[j]
        f = C / decision_interval
        servers.append(Server(f=f, C=C, L0=0.0))

    return servers


def generate_servers(n_servers: int, f_range: Tuple[float, float] = (100, 200),
                     decision_interval: float = 30.0) -> List[Server]:
    """生成服务器列表（简单模式）"""
    servers = []
    for _ in range(n_servers):
        f = np.random.uniform(*f_range)
        C = f * decision_interval
        servers.append(Server(f=f, C=C, L0=0.0))
    return servers
