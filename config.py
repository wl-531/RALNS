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

# ============================================================
# Google Trace 2019 Profile (v1)
# ============================================================
# 分类定义：
#   Type A (Stable):   CV < 0.15，占比 ~14%
#   Type B (Volatile): CV > 0.40，占比 ~72%  <- 隐性过载的主要来源
#   Type C (General):  0.15 <= CV <= 0.40，占比 ~14%

# 缩放因子：将归一化 CPU (0-1) 放大到合理工作量范围
TRACE_SCALE_FACTOR = 10000

# Type A: Stable (低风险)
TYPE_A_MU_RANGE = (10, 165)     # 下界提升到 10，避免过小
TYPE_A_CV_RANGE = (0.02, 0.15)  # 略微放宽下界，避免 sigma 约等于 0

# Type B: Volatile (高风险) <- 这是展示风险感知优势的关键
TYPE_B_MU_RANGE = (10, 197)
TYPE_B_CV_RANGE = (0.40, 2.5)   # 上界 cap 在 2.5，避免极端值

# Type C: General (中等风险)
TYPE_C_MU_RANGE = (10, 264)
TYPE_C_CV_RANGE = (0.15, 0.40)

# 场景配置
SCENARIOS = {
    'stable':   {'n_tasks': 80,  'm_servers': 10, 'type_mix': [0.50, 0.30, 0.20], 'rho': 0.75},
    'mixed':    {'n_tasks': 100, 'm_servers': 10, 'type_mix': [0.30, 0.50, 0.20], 'rho': 0.80},
    'volatile': {'n_tasks': 100, 'm_servers': 10, 'type_mix': [0.15, 0.70, 0.15], 'rho': 0.85},
    'stress':   {'n_tasks': 120, 'm_servers': 10, 'type_mix': [0.10, 0.80, 0.10], 'rho': 0.92},
}
