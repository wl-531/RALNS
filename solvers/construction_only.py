"""Construction-Only: RA-LNS 消融版本（仅 Phase 0，禁用所有搜索）

仅执行 risk-first construction，禁用所有搜索阶段：
- 复用 RA-LNS 的 _risk_first_construction 方法
- solve() 方法在构造完成后直接返回，不进入 Phase 1
"""
from typing import List
from models.task import Task
from models.server import Server
from .ra_lns import RALNSSolver, RALNSSolution
from config import KAPPA


class ConstructionOnlySolver(RALNSSolver):
    """Construction-Only Solver（消融 baseline）

    = RA-LNS Phase 0 only
    仅保留 Risk-First Construction
    """

    def __init__(self, kappa: float = KAPPA):
        super().__init__(kappa=kappa, patience=0, destroy_k=0, t_max=0)

    def solve(self, tasks: List[Task], servers: List[Server]) -> List[int]:
        """主求解入口：仅执行 Phase 0"""
        sol, _ = self._risk_first_construction(tasks, servers)
        return sol.assignment.copy()
