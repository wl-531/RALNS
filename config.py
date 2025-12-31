"""全局配置"""

# 风险参数
ALPHA = 0.15                  # CVR 目标阈值
KAPPA = 2.38                  # 不确定性系数 sqrt(1/alpha - 1)

# RA-LNS 参数
T_MAX = 0.005                 # 5ms 默认时间预算
PATIENCE = 15                 # Stage-1A 不改进次数阈值
DESTROY_K = 3                 # Stage-1B destroy 任务数
EPS_TOL = 1e-9                # 可行性容差
EPS_CMP = 1e-6                # 浮点比较容差
EPS_DIV = 1e-6                # 除零保护

# 实验配置
N_PERIODS = 100               # 在线周期数
N_RUNS = 30                   # 重复次数
MC_SAMPLES = 1000             # Monte Carlo 采样数
DECISION_INTERVAL = 30.0      # 决策周期（秒）

# 场景配置
SCENARIOS = {
    'stable':   {'n_tasks': 80,  'm_servers': 10, 'type_mix': [0.5, 0.1, 0.4], 'rho': 0.75},
    'mixed':    {'n_tasks': 100, 'm_servers': 10, 'type_mix': [0.3, 0.2, 0.5], 'rho': 0.80},
    'volatile': {'n_tasks': 100, 'm_servers': 10, 'type_mix': [0.2, 0.4, 0.4], 'rho': 0.85},
    'stress':   {'n_tasks': 120, 'm_servers': 10, 'type_mix': [0.2, 0.5, 0.3], 'rho': 0.92},
}

# 任务类型参数（双峰模式）
# Type A: 陷阱任务（低 μ 高 σ）
TYPE_A_MU_RANGE = (30, 50)
TYPE_A_CV_RANGE = (2.0, 3.5)

# Type B: 稳定任务（高 μ 低 σ）
TYPE_B_MU_RANGE = (80, 120)
TYPE_B_CV_RANGE = (0.10, 0.20)
