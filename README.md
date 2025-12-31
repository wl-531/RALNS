# RA-LNS 实验项目

Risk-Aware Large Neighborhood Search for Online Task Scheduling in Edge Computing.

## 项目结构

```
RALNS/
├── config.py                 # 全局配置
├── models/
│   ├── task.py              # 任务模型
│   └── server.py            # 服务器模型
├── data/
│   ├── generator.py         # 任务生成器
│   └── trace_loader.py      # Alibaba Trace 加载器（占位）
├── solvers/
│   ├── base.py              # Solver 基类
│   ├── dg.py                # Deterministic Greedy
│   ├── rg.py                # Robust Greedy (κ-Greedy)
│   ├── micro_lns.py         # Micro-Only LNS（消融）
│   └── ra_lns.py            # RA-LNS 完整版
├── evaluation/
│   ├── metrics.py           # 统一指标计算
│   └── monte_carlo.py       # Monte Carlo CVR 验证
├── experiments/
│   ├── run_main.py          # 主实验
│   ├── run_quality_time.py  # Quality vs Time
│   ├── run_ablation.py      # 消融实验
│   └── run_sensitivity.py   # 参数敏感性
├── utils/
│   └── plotting.py          # 绘图工具
└── README.md
```

## 快速开始

### 环境要求

```bash
pip install numpy pandas matplotlib
```

### 运行主实验

```bash
cd C:\Users\wl\Desktop\MODRO\RALNS
python -m experiments.run_main
```

### 运行其他实验

```bash
# Quality vs Time
python -m experiments.run_quality_time

# 消融实验
python -m experiments.run_ablation

# 参数敏感性
python -m experiments.run_sensitivity
```

## 算法说明

### RA-LNS

RA-LNS (Risk-Aware Large Neighborhood Search) 是一个确定性的 anytime 求解器，专为在线任务调度设计。

**两阶段结构：**
1. **Phase 0: Risk-First Construction** - 按 δ_i = μ_i + κσ_i 降序贪心分配
2. **Phase 1: Risk-Density-Guided Descent**
   - Stage-1A: Micro Risk Hedging (Relocate/Swap)
   - Stage-1B: Macro Risk Rebalancing (Destroy/Repair)

**字典序目标：**
```
Ψ(X) = (-feas, U_max, O1)
Tie-break: R_sum
```

### 对比算法

| 算法 | 描述 |
|------|------|
| DG | Deterministic Greedy，仅基于期望负载 |
| RG | Robust Greedy (κ-Greedy)，使用鲁棒负载 |
| Micro-LNS | RA-LNS without Stage-1B |
| RA-LNS | 完整版 |

## 实验场景

| 场景 | n_tasks | m_servers | ρ |
|------|---------|-----------|-----|
| stable | 80 | 10 | 0.75 |
| mixed | 100 | 10 | 0.80 |
| volatile | 100 | 10 | 0.85 |
| stress | 120 | 10 | 0.92 |

## 参数配置

```python
# config.py
ALPHA = 0.15      # CVR 目标阈值
KAPPA = 2.38      # 不确定性系数 sqrt(1/α - 1)
T_MAX = 0.005     # 5ms 时间预算
PATIENCE = 15     # Stage-1A 耐心值
DESTROY_K = 3     # Stage-1B 销毁任务数
```

## 输出文件

- `results_main.csv` - 主实验结果
- `results_quality_time.csv` - Quality vs Time 结果
- `results_ablation.csv` - 消融实验结果
- `results_sensitivity_*.csv` - 参数敏感性结果
