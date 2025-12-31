"""Alibaba Trace 加载器（占位）"""
from typing import List
from models.task import Task


def load_alibaba_trace(trace_path: str, period_idx: int) -> List[Task]:
    """加载 Alibaba Trace 数据

    Args:
        trace_path: Trace 文件路径
        period_idx: 周期索引

    Returns:
        List[Task]

    TODO: 实现实际的 trace 加载逻辑
    """
    raise NotImplementedError("Alibaba Trace loader is not implemented yet")
