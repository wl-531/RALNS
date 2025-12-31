"""Solver 基类"""
from abc import ABC, abstractmethod
from typing import List, Tuple
from models.task import Task
from models.server import Server


class BaseSolver(ABC):
    """调度算法基类"""

    @abstractmethod
    def solve(self, tasks: List[Task], servers: List[Server]) -> List[int]:
        """求解任务分配

        Args:
            tasks: 任务列表
            servers: 服务器列表

        Returns:
            assignment: List[int]，assignment[i] = j 表示任务 i 分配给服务器 j
        """
        pass

    @property
    def name(self) -> str:
        """算法名称"""
        return self.__class__.__name__
