"""RA-LNS: Risk-Aware Large Neighborhood Search

字典序：Psi(X) = (-feas, RR_max, O1)，tie-break: R_sum

Level-1 使用 RR_max = max_j [σ_j / (C_j - μ_j)] 直接对应 Cantelli CVR bound，
而非 U_max（负载均衡指标）。
"""
import time
import numpy as np
from typing import List, Tuple
from models.task import Task
from models.server import Server
from .base import BaseSolver
from config import KAPPA, T_MAX, PATIENCE, DESTROY_K, EPS_TOL, EPS_CMP, EPS_DIV


class RALNSSolution:
    """解的表示（支持增量更新）"""

    def __init__(self, servers: List[Server], kappa: float):
        self.m = len(servers)
        self.kappa = kappa
        self.C = np.array([s.C for s in servers])
        self.L0 = np.array([s.L0 for s in servers])
        self.mu_sum = np.zeros(self.m)
        self.sigma_sq_sum = np.zeros(self.m)
        self.assignment = []

    @property
    def sigma_j(self) -> np.ndarray:
        """每台服务器的聚合标准差"""
        return np.sqrt(np.maximum(self.sigma_sq_sum, 0))

    @property
    def L_hat(self) -> np.ndarray:
        """鲁棒负载 L_hat_j = L0_j + mu_j + kappa * sigma_j"""
        return self.L0 + self.mu_sum + self.kappa * self.sigma_j

    @property
    def Gap(self) -> np.ndarray:
        """剩余鲁棒容量 Gap_j = C_j - L_hat_j"""
        return self.C - self.L_hat

    @property
    def RD(self) -> np.ndarray:
        """风险密度 RD_j = sigma_j / max(Gap_j, eps_div)，用于搜索引导"""
        return self.sigma_j / np.maximum(self.Gap, EPS_DIV)

    @property
    def margin(self) -> np.ndarray:
        """原始安全余量: C_j - μ_j(X)，不含 κσ 项

        用于计算 Risk Ratio，直接对应 Cantelli bound
        """
        return self.C - (self.L0 + self.mu_sum)

    @property
    def RR(self) -> np.ndarray:
        """Risk Ratio: σ_j / (C_j - μ_j)，直接对应 Cantelli bound

        理论依据:
            Cantelli: Pr{L̃_j > C_j} ≤ 1/(1 + k_j²), k_j = (C_j - μ_j)/σ_j
            所以 RR_j = 1/k_j = σ_j/(C_j - μ_j)
            min max_j RR_j ⟺ min max_j CVR_j

        注意: 与 RD 的区别是分母不含 κσ_j 项
            - RR: σ_j / (C_j - μ_j)      ← Level-1 接受准则
            - RD: σ_j / (C_j - L̂_j)     ← 搜索引导 + tie-break
        """
        return self.sigma_j / np.maximum(self.margin, EPS_DIV)

    @property
    def RR_max(self) -> float:
        """Level-1 目标: 最大风险比（越小越好）

        物理意义: 最小化最坏情况的单服务器 CVR
        """
        return float(np.max(self.RR))

    @property
    def U_max(self) -> float:
        """最大鲁棒利用率（仅用于日志，不再用于 Level-1）"""
        return float(np.max(self.L_hat / self.C))

    @property
    def O1(self) -> float:
        """Level-2: 鲁棒 Makespan"""
        return float(np.max(self.L_hat))

    @property
    def R_sum(self) -> float:
        """Tie-break: 总风险密度"""
        return float(np.sum(self.RD))

    def is_feasible(self) -> bool:
        """Level-0: 可行性检查（使用 eps_tol）"""
        return bool(np.all(self.Gap >= -EPS_TOL))

    def Psi(self) -> Tuple[int, float, float]:
        """3层字典序向量（越小越好）

        Level-0: 可行性 (硬约束)，可行=0, 不可行=1
        Level-1: min max_j RR_j (CVR 代理，直接对应 Cantelli)
        Level-2: min O₁ = min max_j L̂_j (makespan)
        """
        return (
            0 if self.is_feasible() else 1,
            self.RR_max,
            self.O1
        )

    def apply_move(self, task_idx: int, task: Task, from_j: int, to_j: int):
        """应用移动操作"""
        if from_j is not None and from_j >= 0:
            self.mu_sum[from_j] -= task.mu
            self.sigma_sq_sum[from_j] -= task.sigma ** 2
        self.mu_sum[to_j] += task.mu
        self.sigma_sq_sum[to_j] += task.sigma ** 2
        self.assignment[task_idx] = to_j

    def rollback_move(self, task_idx: int, task: Task, from_j: int, to_j: int):
        """回滚移动操作"""
        self.mu_sum[to_j] -= task.mu
        self.sigma_sq_sum[to_j] -= task.sigma ** 2
        if from_j is not None and from_j >= 0:
            self.mu_sum[from_j] += task.mu
            self.sigma_sq_sum[from_j] += task.sigma ** 2
        self.assignment[task_idx] = from_j if from_j is not None else -1

    def copy(self) -> 'RALNSSolution':
        """深拷贝"""
        new_sol = RALNSSolution.__new__(RALNSSolution)
        new_sol.m = self.m
        new_sol.kappa = self.kappa
        new_sol.C = self.C.copy()
        new_sol.L0 = self.L0.copy()
        new_sol.mu_sum = self.mu_sum.copy()
        new_sol.sigma_sq_sum = self.sigma_sq_sum.copy()
        new_sol.assignment = self.assignment.copy()
        return new_sol


class RALNSSolver(BaseSolver):
    """RA-LNS Solver"""

    def __init__(self, kappa: float = KAPPA, patience: int = PATIENCE,
                 destroy_k: int = DESTROY_K, t_max: float = T_MAX,
                 eps_tol: float = EPS_TOL, eps_cmp: float = EPS_CMP,
                 eps_div: float = EPS_DIV):
        self.kappa = kappa
        self.patience = patience
        self.destroy_k = destroy_k
        self.t_max = t_max
        self.eps_tol = eps_tol
        self.eps_cmp = eps_cmp
        self.eps_div = eps_div

    def solve(self, tasks: List[Task], servers: List[Server]) -> List[int]:
        """主求解入口"""
        start = time.perf_counter()

        # Phase 0: Risk-First Construction
        sol, fallback_count = self._risk_first_construction(tasks, servers)

        # 初始化全局最优
        best = sol.copy() if sol.is_feasible() else None
        best_rsum = sol.R_sum if best else float('inf')

        stagnation = 0
        iteration = 0
        max_lns_triggers = max(sol.m * 3, 30)  # 增加 LNS 尝试次数
        lns_triggers = 0

        # Phase 1: Risk-Density-Guided Descent (论文版本)
        while time.perf_counter() - start < self.t_max:
            if stagnation < self.patience:
                # Stage-1A: Micro Risk Hedging
                improved = self._risk_hedging_move(sol, tasks)
            else:
                # Stage-1B: Macro Risk Rebalancing
                improved = self._risk_guided_lns(sol, tasks)
                stagnation = 0  # 论文：无条件重置
                lns_triggers += 1
                if lns_triggers >= max_lns_triggers:
                    break  # 防止死循环

            if improved:
                stagnation = 0
                if sol.is_feasible():
                    if best is None or self._better(sol, best):
                        best = sol.copy()
                        best_rsum = sol.R_sum
            else:
                stagnation += 1

            iteration += 1
            if iteration > 1000:
                break

        # 返回结果
        result = best if best else sol
        return result.assignment.copy()

    def _risk_first_construction(self, tasks: List[Task],
                                  servers: List[Server]) -> Tuple[RALNSSolution, int]:
        """Phase 0: Risk-First Construction

        按到达顺序贪心分配（随机分散高风险任务）。
        选择使 new_RR_j = σ'_j / (C_j - μ'_j) 最小的可行服务器。
        与 Level-1 目标 (min RR_max) 一致。
        """
        sol = RALNSSolution(servers, self.kappa)
        fallback_count = 0
        n_tasks = len(tasks)

        # 按到达顺序（任务已 shuffle，自然分散风险）
        sorted_indices = list(range(n_tasks))
        sol.assignment = [-1] * n_tasks

        for i in sorted_indices:
            task = tasks[i]

            # 计算分配到每台服务器后的状态
            new_sigma_sq = sol.sigma_sq_sum + task.sigma ** 2
            new_sigma = np.sqrt(np.maximum(new_sigma_sq, 0))
            new_mu = sol.mu_sum + task.mu
            new_L_hat = sol.L0 + new_mu + self.kappa * new_sigma
            new_Gap = sol.C - new_L_hat

            # 计算 new_RR_j = σ'_j / margin'_j（与 Level-1 目标一致）
            new_margin = sol.C - (sol.L0 + new_mu)
            new_RR = new_sigma / np.maximum(new_margin, self.eps_div)

            # 选择可行且 RR 最小的服务器
            best_j = None
            best_rr = np.inf
            for j in range(sol.m):
                if new_Gap[j] >= -self.eps_tol and new_RR[j] < best_rr:
                    best_rr = new_RR[j]
                    best_j = j

            if best_j is not None:
                sol.assignment[i] = best_j
                sol.mu_sum[best_j] += task.mu
                sol.sigma_sq_sum[best_j] += task.sigma ** 2
            else:
                # Fallback: 分配到 RR 最小的服务器（即使不可行）
                fallback_count += 1
                j_min = int(np.argmin(new_RR))
                sol.assignment[i] = j_min
                sol.mu_sum[j_min] += task.mu
                sol.sigma_sq_sum[j_min] += task.sigma ** 2

        return sol, fallback_count

    def _risk_hedging_move(self, sol: RALNSSolution, tasks: List[Task]) -> bool:
        """Stage-1A: Micro Risk Hedging

        RR 引导（与 Level-1 目标一致）：
        1. 按 RR 降序遍历服务器（优先优化风险比最高的）
        2. 对每个服务器，按 sigma 降序遍历 victims
        3. 找到改进就应用并返回
        """
        # 按 RR 降序遍历服务器（与 Level-1 目标 min RR_max 一致）
        server_order = list(np.argsort(-sol.RR))

        for from_j in server_order:
            victims = [i for i, j in enumerate(sol.assignment) if j == from_j]
            if not victims:
                continue

            # 按 sigma 降序排列 victims
            victim_sigmas = [(i, tasks[i].sigma) for i in victims]
            victim_sigmas.sort(key=lambda x: -x[1])

            best_move = None
            best_psi = sol.Psi()
            best_rsum = sol.R_sum

            # 遍历所有 victims（按 sigma 降序）
            for victim_idx, _ in victim_sigmas:
                victim_task = tasks[victim_idx]

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

                    # Swap: 与 to_j 上 sigma 最小的任务交换
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

            # 如果当前服务器找到改进，应用并返回
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
        """Stage-1B: Macro Risk Rebalancing

        1. Destroy: RR 引导选择服务器（与 Level-1 目标一致），移除 top-k 最高方差的可移动任务
        2. Repair: 按方差降序重新插入，选择最小化 new_RR_j 的服务器
        """
        backup_sol = sol.copy()

        # 按 RR 降序遍历服务器（与 Level-1 目标 min RR_max 一致）
        server_order = list(np.argsort(-sol.RR))

        j_hot = None
        victims = []
        for j in server_order:
            candidates = [i for i, jj in enumerate(sol.assignment) if jj == j]
            if len(candidates) >= self.destroy_k:
                j_hot = j
                victims = candidates
                break

        if j_hot is None:
            return False

        # BUG FIX: 只选择可移动的任务（有至少一个可行目标服务器）
        movable_victims = []
        for i in victims:
            task = tasks[i]
            # 检查是否有可行的目标服务器（排除 j_hot）
            has_feasible_target = False
            for j_to in range(sol.m):
                if j_to == j_hot:
                    continue
                new_sigma_sq_j = sol.sigma_sq_sum[j_to] + task.sigma ** 2
                new_sigma_j = np.sqrt(max(new_sigma_sq_j, 0))
                new_mu_j = sol.mu_sum[j_to] + task.mu
                new_L_hat_j = sol.L0[j_to] + new_mu_j + self.kappa * new_sigma_j
                new_Gap_j = sol.C[j_to] - new_L_hat_j
                if new_Gap_j >= -self.eps_tol:
                    has_feasible_target = True
                    break
            if has_feasible_target:
                movable_victims.append((i, tasks[i].sigma))

        # 如果可移动任务不足，返回失败
        if len(movable_victims) < self.destroy_k:
            return False

        # 从可移动任务中选择 top-k 最高方差
        movable_victims.sort(key=lambda x: x[1], reverse=True)
        destroy_tasks = [i for i, _ in movable_victims[:self.destroy_k]]

        # Destroy: 移除任务
        for i in destroy_tasks:
            task = tasks[i]
            j = sol.assignment[i]
            sol.mu_sum[j] -= task.mu
            sol.sigma_sq_sum[j] -= task.sigma ** 2
            sol.assignment[i] = -1

        # Repair: 按方差降序重新插入，选择最小化 new_RR_j 的服务器
        # BUG FIX: 使用 RR (与目标一致) 而不是 RD
        repair_order = sorted(destroy_tasks, key=lambda i: tasks[i].sigma, reverse=True)
        is_first = True

        for i in repair_order:
            task = tasks[i]

            # 计算插入到每台服务器后的 RR
            best_j = None
            best_new_rr = np.inf
            fallback_j = None
            fallback_new_rr = np.inf

            for j in range(sol.m):
                # 计算插入后的状态
                new_sigma_sq_j = sol.sigma_sq_sum[j] + task.sigma ** 2
                new_sigma_j = np.sqrt(max(new_sigma_sq_j, 0))
                new_mu_j = sol.mu_sum[j] + task.mu
                new_L_hat_j = sol.L0[j] + new_mu_j + self.kappa * new_sigma_j
                new_Gap_j = sol.C[j] - new_L_hat_j

                # 跳过不可行的服务器
                if new_Gap_j < -self.eps_tol:
                    continue

                # 计算新的 RR_j = sigma_j / margin_j (与目标一致)
                new_margin_j = sol.C[j] - (sol.L0[j] + new_mu_j)
                new_RR_j = new_sigma_j / max(new_margin_j, self.eps_div)

                # 第一个任务优先不返回 j_hot
                if is_first and j == j_hot:
                    if fallback_j is None or new_RR_j < fallback_new_rr:
                        fallback_j = j
                        fallback_new_rr = new_RR_j
                    continue

                if new_RR_j < best_new_rr:
                    best_new_rr = new_RR_j
                    best_j = j

            # 如果没有可行的非 j_hot 选项，允许返回 j_hot
            if best_j is None and fallback_j is not None:
                best_j = fallback_j

            # 分配任务到最小 ΔRD 的服务器
            if best_j is not None:
                sol.assignment[i] = best_j
                sol.mu_sum[best_j] += task.mu
                sol.sigma_sq_sum[best_j] += task.sigma ** 2
            else:
                # Fallback: 放到 L_hat 最小的服务器（排除 j_hot 如果是第一个任务）
                candidates = [j for j in range(sol.m) if not (is_first and j == j_hot)]
                if candidates:
                    L_hat_cands = [sol.L_hat[j] for j in candidates]
                    j_min = candidates[int(np.argmin(L_hat_cands))]
                else:
                    j_min = int(np.argmin(sol.L_hat))
                sol.assignment[i] = j_min
                sol.mu_sum[j_min] += task.mu
                sol.sigma_sq_sum[j_min] += task.sigma ** 2

            is_first = False

        # 检查是否改进
        if self._better(sol, backup_sol):
            return True
        else:
            # 回滚
            sol.mu_sum = backup_sol.mu_sum.copy()
            sol.sigma_sq_sum = backup_sol.sigma_sq_sum.copy()
            sol.assignment = backup_sol.assignment.copy()
            return False

    def _better(self, sol1: RALNSSolution, sol2: RALNSSolution) -> bool:
        """比较两个解：sol1 是否优于 sol2"""
        return self._psi_better(sol1.Psi(), sol2.Psi(), sol1.R_sum, sol2.R_sum)

    def _psi_better(self, psi1: Tuple, psi2: Tuple, r_sum1: float, r_sum2: float) -> bool:
        """3 层字典序比较 + R_sum tie-break

        Psi = (-feas, RR_max, O1)
        越小越好
        """
        # Level-0: feas（严格比较）
        if psi1[0] != psi2[0]:
            return psi1[0] < psi2[0]

        # Level-1, Level-2: 浮点比较
        for v1, v2 in zip(psi1[1:], psi2[1:]):
            if v1 < v2 - self.eps_cmp:
                return True
            if v1 > v2 + self.eps_cmp:
                return False

        # Tie-break: R_sum 最小者胜
        return r_sum1 < r_sum2 - self.eps_cmp


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    from models.task import Task
    from models.server import Server

    print("=" * 60)
    print("测试 RA-LNS Level-1 修改: U_max -> RR_max")
    print("=" * 60)

    # 创建测试数据：包含"陷阱任务"（低μ高σ）
    tasks = []
    # Type A: 稳定任务（高μ低σ）
    for _ in range(15):
        tasks.append(Task(mu=80, sigma=10))
    # Type B: 陷阱任务（低μ高σ，Implicit Overload 元凶）
    for _ in range(5):
        tasks.append(Task(mu=30, sigma=60))

    servers = [Server(f=100, C=500, L0=0) for _ in range(5)]

    print(f"\n任务分布:")
    print(f"  - 稳定任务 (mu=80, sigma=10): 15 个")
    print(f"  - 陷阱任务 (mu=30, sigma=60): 5 个")
    print(f"  - 服务器: 5 台, C=500 each")

    # 运行 RA-LNS
    solver = RALNSSolver(kappa=2.38, patience=15, destroy_k=3, t_max=0.01)
    assignment = solver.solve(tasks, servers)

    # 重建解对象以输出指标
    sol = RALNSSolution(servers, kappa=2.38)
    sol.assignment = [-1] * len(tasks)
    for i, j in enumerate(assignment):
        sol.mu_sum[j] += tasks[i].mu
        sol.sigma_sq_sum[j] += tasks[i].sigma ** 2
        sol.assignment[i] = j

    print(f"\n===== RA-LNS 结果 =====")
    print(f"Feasible: {sol.is_feasible()}")
    print(f"RR_max:   {sol.RR_max:.4f}  <- 新 Level-1 (直接对应 CVR)")
    print(f"U_max:    {sol.U_max:.4f}   <- 旧 Level-1 (仅供对比)")
    print(f"O1:       {sol.O1:.2f}      <- Level-2 (makespan)")
    print(f"R_sum:    {sol.R_sum:.4f}   <- Tie-break")
    print(f"Psi:      {sol.Psi()}")

    # 检查每台服务器的任务数
    tasks_per_server = [assignment.count(j) for j in range(5)]
    print(f"\nTasks per server: {tasks_per_server}")

    # 检查陷阱任务的分布（应该分散到不同服务器）
    trap_indices = list(range(15, 20))
    trap_servers = [assignment[i] for i in trap_indices]
    print(f"\n陷阱任务 (高sigma) 分配:")
    print(f"  服务器分布: {trap_servers}")
    print(f"  分散度: {len(set(trap_servers))}/5 台服务器")

    # 输出每台服务器的 RR 值
    print(f"\n每台服务器的风险比 RR_j:")
    for j in range(5):
        print(f"  Server {j}: RR={sol.RR[j]:.4f}, margin={sol.margin[j]:.1f}, sigma={sol.sigma_j[j]:.1f}")

    # 验证：陷阱任务应该被分散
    if len(set(trap_servers)) >= 3:
        print(f"\n[OK] 陷阱任务被正确分散（{len(set(trap_servers))} 台服务器）")
    else:
        print(f"\n[WARN] 陷阱任务过于集中（仅 {len(set(trap_servers))} 台服务器）")

    print("\n" + "=" * 60)
