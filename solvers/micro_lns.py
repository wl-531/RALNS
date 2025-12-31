"""Micro-Only LNS: RA-LNS 的消融版本（禁用 Stage-1B）"""
from .ra_lns import RALNSSolver, RALNSSolution
from typing import List
from models.task import Task


class MicroLNSSolver(RALNSSolver):
    """Micro-Only LNS（消融 baseline）

    = RA-LNS without Stage-1B
    仅保留 Phase 0 + Stage-1A
    """

    def _risk_guided_lns(self, sol: RALNSSolution, tasks: List[Task]) -> bool:
        """禁用 Stage-1B"""
        return False
