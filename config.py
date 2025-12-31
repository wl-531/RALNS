"""全局配置"""

# 风险参数
ALPHA = 0.15
KAPPA = 2.38

# RA-LNS 参数
T_MAX = 0.005
PATIENCE = 15
DESTROY_K = 3
EPS_TOL = 1e-9
EPS_CMP = 1e-6
EPS_DIV = 1e-6

# 实验配置
N_PERIODS = 100
N_RUNS = 30
MC_SAMPLES = 1000
DECISION_INTERVAL = 50.0

# ============================================================
# Google Trace 2019 Profile (v1)
# ============================================================
# Type A (Stable):   CV < 0.15，占比 ~14%
# Type B (Volatile): CV > 0.40，占比 ~72%  <- 隐性过载来源
# Type C (General):  0.15 <= CV <= 0.40，占比 ~14%

TYPE_A_MU_RANGE = (10, 165)
TYPE_A_CV_RANGE = (0.02, 0.15)

TYPE_B_MU_RANGE = (10, 197)
TYPE_B_CV_RANGE = (0.40, 2.5)

TYPE_C_MU_RANGE = (10, 264)
TYPE_C_CV_RANGE = (0.15, 0.40)

# Google Trace 真实分布（固定，不可调）
TYPE_MIX_GOOGLE = [0.14, 0.72, 0.14]

# ============================================================
# 实验配置
# ============================================================
# 主实验：target_rho 是鲁棒利用率（不是期望利用率）
# target_rho = 0.90 意味着 U_max ≈ 0.90，系统有 10% 安全余量
MAIN_CONFIG = {
    'n_tasks': 100,
    'm_servers': 10,
    'type_mix': TYPE_MIX_GOOGLE,
    'rho': 0.90,  # 鲁棒利用率 90%
}

# ρ 敏感性实验（鲁棒利用率）
RHO_VALUES = [0.80, 0.85, 0.90, 0.95, 0.98]
