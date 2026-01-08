"""Micro-Only LNS: RA-LNS 的消融版本（禁用 Stage-1B，使用局限 Stage-1A）"""
import numpy as np
from .ra_lns import RALNSSolver, RALNSSolution
from typing import List
from models.task import Task


class MicroLNSSolver(RALNSSolver):
    """Micro-Only LNS（消融 baseline）

    = Phase 0 + 局限 Stage-1A（只从 argmax RD 服务器选择一个 victim）
    禁用 Stage-1B

    用于展示全面搜索（RA-LNS）相对于局限搜索的优势。
    """

    def _risk_hedging_move(self, sol: RALNSSolution, tasks: List[Task]) -> bool:
        """Stage-1A: 局限版 Micro Risk Hedging

        只从 argmax(RD) 服务器选择 sigma 最大的一个任务。
        """
        # 只找 argmax(RD) 服务器
        j_hot = int(np.argmax(sol.RD))
        victims = [i for i, j in enumerate(sol.assignment) if j == j_hot]
        if not victims:
            return False

        # 只选 sigma 最大的一个任务
        victim_sigmas = [tasks[i].sigma for i in victims]
        victim_idx = victims[int(np.argmax(victim_sigmas))]
        victim_task = tasks[victim_idx]
        from_j = j_hot

        best_move = None
        best_psi = sol.Psi()
        best_rsum = sol.R_sum

        for to_j in range(sol.m):
            if to_j == from_j:
                continue

            # Relocate
            sol.apply_move(victim_idx, victim_task, from_j, to_j)
            new_psi = sol.Psi()
            new_rsum = sol.R_sum
            if self._psi_better(new_psi, best_psi, new_rsum, best_rsum):
                best_psi = new_psi
                best_rsum = new_rsum
                best_move = ('relocate', victim_idx, victim_task, from_j, to_j)
            sol.rollback_move(victim_idx, victim_task, from_j, to_j)

            # Swap
            swap_cands = [i for i, j in enumerate(sol.assignment) if j == to_j]
            if swap_cands:
                swap_sigmas = [tasks[i].sigma for i in swap_cands]
                swap_idx = swap_cands[int(np.argmin(swap_sigmas))]
                swap_task = tasks[swap_idx]

                sol.apply_move(victim_idx, victim_task, from_j, to_j)
                sol.apply_move(swap_idx, swap_task, to_j, from_j)
                new_psi = sol.Psi()
                new_rsum = sol.R_sum
                if self._psi_better(new_psi, best_psi, new_rsum, best_rsum):
                    best_psi = new_psi
                    best_rsum = new_rsum
                    best_move = ('swap', victim_idx, victim_task, from_j, to_j,
                                 swap_idx, swap_task)
                sol.rollback_move(swap_idx, swap_task, to_j, from_j)
                sol.rollback_move(victim_idx, victim_task, from_j, to_j)

        if best_move:
            if best_move[0] == 'relocate':
                _, vi, vt, fj, tj = best_move
                sol.apply_move(vi, vt, fj, tj)
            else:
                _, vi, vt, fj, tj, si, st = best_move
                sol.apply_move(vi, vt, fj, tj)
                sol.apply_move(si, st, tj, fj)
            return True
        return False

    def _risk_guided_lns(self, sol: RALNSSolution, tasks: List[Task]) -> bool:
        """禁用 Stage-1B"""
        return False
