"""数据生成器 - 基于 Google Cluster Trace 2019"""
import numpy as np
from typing import List

from models.task import Task
from models.server import Server
from config import (
    TYPE_A_MU_RANGE, TYPE_A_CV_RANGE,
    TYPE_B_MU_RANGE, TYPE_B_CV_RANGE,
    TYPE_C_MU_RANGE, TYPE_C_CV_RANGE,
    DECISION_INTERVAL,
)

# Profile 配置
TASK_PROFILES = {
    'A': {'mu_range': TYPE_A_MU_RANGE, 'cv_range': TYPE_A_CV_RANGE, 'name': 'Stable'},
    'B': {'mu_range': TYPE_B_MU_RANGE, 'cv_range': TYPE_B_CV_RANGE, 'name': 'Volatile'},
    'C': {'mu_range': TYPE_C_MU_RANGE, 'cv_range': TYPE_C_CV_RANGE, 'name': 'General'},
}


def generate_task_by_type(task_type: str) -> Task:
    """根据类型生成单个任务"""
    profile = TASK_PROFILES[task_type]
    mu = np.random.uniform(*profile['mu_range'])
    cv = np.random.uniform(*profile['cv_range'])
    sigma = mu * cv
    return Task(mu=mu, sigma=sigma)


def generate_batch(n_tasks: int, type_mix: List[float] = None) -> List[Task]:
    """按比例混合生成任务批次

    Args:
        n_tasks: 任务数量
        type_mix: [Type_A比例, Type_B比例, Type_C比例]
                  A=Stable, B=Volatile, C=General
                  默认 [0.15, 0.70, 0.15]（接近 Google Trace 真实分布）
    """
    if type_mix is None:
        type_mix = [0.15, 0.70, 0.15]

    assert len(type_mix) == 3, "type_mix must have 3 elements [A, B, C]"
    assert abs(sum(type_mix) - 1.0) < 0.01, "type_mix must sum to 1"

    tasks = []
    type_names = ['A', 'B', 'C']

    for type_name, ratio in zip(type_names, type_mix):
        n_type = int(n_tasks * ratio)
        for _ in range(n_type):
            tasks.append(generate_task_by_type(type_name))

    # 补齐因取整丢失的任务
    while len(tasks) < n_tasks:
        type_name = np.random.choice(type_names, p=type_mix)
        tasks.append(generate_task_by_type(type_name))

    np.random.shuffle(tasks)
    return tasks[:n_tasks]


def generate_tasks(n_tasks: int, mode: str = "google_trace") -> List[Task]:
    """生成任务列表（兼容旧接口）

    Args:
        mode: "google_trace" (默认) 或 "bimodal" (向后兼容)
    """
    if mode == "google_trace":
        return generate_batch(n_tasks, type_mix=[0.15, 0.70, 0.15])
    elif mode == "bimodal":
        # 向后兼容：60% Stable + 40% Volatile
        return generate_batch(n_tasks, type_mix=[0.60, 0.40, 0.0])
    else:
        raise ValueError(f"Unknown mode: {mode}")


def generate_servers_with_target_rho(n_servers: int, total_expected_load: float,
                                      target_rho: float,
                                      decision_interval: float = DECISION_INTERVAL) -> List[Server]:
    """根据目标 rho 反算服务器容量"""
    total_capacity = total_expected_load / target_rho
    capacity_per_server = total_capacity / n_servers

    factors = np.random.uniform(0.9, 1.1, n_servers)
    factors = factors / factors.sum() * n_servers

    servers = []
    for j in range(n_servers):
        C = capacity_per_server * factors[j]
        f = C / decision_interval
        servers.append(Server(f=f, C=C, L0=0.0))

    return servers
