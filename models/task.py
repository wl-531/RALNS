"""任务模型"""
from dataclasses import dataclass


@dataclass
class Task:
    """任务类

    Attributes:
        mu: 期望工作量
        sigma: 工作量标准差
    """
    mu: float
    sigma: float

    def get_delta(self, kappa: float) -> float:
        """计算边际鲁棒负载 delta_i = mu_i + kappa * sigma_i"""
        return self.mu + kappa * self.sigma
