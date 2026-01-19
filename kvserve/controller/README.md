# KVServe Online Controller

两层在线决策系统，用于自适应压缩策略选择。

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Online Controller                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Tier 1: Analytical Model (理论筛选)                │   │
│  │  - Theorem 1: Benefit condition (B < B_crit)       │   │
│  │  - Theorem 2: Piecewise-optimal (下包络)           │   │
│  │  - 结果: 从大量 profiles 缩减到 top-3 候选         │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Tier 2: ε-greedy Bandit (在线学习)                │   │
│  │  - 预测: T_eff = T_hat + delta_bar                 │   │
│  │  - 选择: ε-greedy (探索/利用)                       │   │
│  │  - 更新: EWMA 残差学习                             │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 核心组件

### 1. Profile（配置文件）
- **定义**: 压缩配置 + 性能指标 + 质量指标
- **文件**: `profile.py`
- **关键属性**:
  - `compression_config`: 传给 CompressionManager 的配置
  - `compression_ratio`: 压缩率 (cr)
  - `harmonic_speed`: 编解码调和速度 (S)
  - `accuracy`: 质量指标

### 2. ProfileLibrary（Profile 库）
- **定义**: 按精度和带宽分桶的 Pareto 前沿
- **文件**: `profile_library.py`
- **功能**:
  - 加载 JSON 格式的 profile 库
  - 根据精度要求找到对应桶
  - 根据带宽找到对应区间
  - 返回候选 profiles（已经是 top-3）

### 3. AnalyticalModel（解析模型）
- **定义**: 理论延迟预测模型
- **文件**: `analytical_model.py`
- **公式**:
  - `T_p = T_model + V/S + V/(B*cr)`（压缩）
  - `T_0 = T_model + V/B`（无压缩）
  - `B_crit = (1 - 1/cr) * S`（临界带宽）

### 4. BanditStateManager（Bandit 状态）
- **定义**: 管理在线学习状态
- **文件**: `bandit_state.py`
- **状态**: `{(bucket_id, interval_id, profile_id): (N, delta_bar)}`
  - `N`: 使用次数
  - `delta_bar`: 残差的 EWMA
- **更新**: `delta_bar ← (1-α)*delta_bar + α*delta_obs`

### 5. OnlineController（主控制器）
- **定义**: 两层决策的核心逻辑
- **文件**: `online_controller.py`
- **接口**:
  - `select_profile()`: 选择压缩策略
  - `update()`: 更新 Bandit 状态

## 使用方法

### 基本用法

```python
from kvserve.controller import ProfileLibrary, OnlineController

# 1. 加载 Profile 库
library = ProfileLibrary("profiles/llama3.1-8b_example.json")

# 2. 创建控制器
controller = OnlineController(
    profile_library=library,
    epsilon=0.1,    # 探索率
    alpha=0.2       # EWMA 学习率
)

# 3. 选择压缩策略
profile, context = controller.select_profile(
    V_bytes=500 * 1024 * 1024,  # KV 体积
    B_mbps=150.0,                # 带宽
    T_model_ms=100.0,            # 模型延迟
    T_SLO_ms=5000.0,             # SLO 约束
    acc_req=0.92                 # 精度要求
)

if profile is not None:
    # 使用选择的压缩配置
    compression_config = profile.compression_config
    # ... 执行压缩和传输 ...
    
    # 4. 观测实际延迟并更新
    T_obs_ms = 450.0  # 实际延迟
    controller.update(context, T_obs_ms)
else:
    # 不使用压缩
    print(f"No compression: {context['reason']}")
```

### 保存/加载 Bandit 状态

```python
# 保存状态（持久化）
controller.save_state("profiles/bandit_state.json")

# 加载状态（恢复学习进度）
controller.load_state("profiles/bandit_state.json")
```

### 查看统计信息

```python
stats = controller.get_statistics()
print(f"Total decisions: {stats['total_decisions']}")
print(f"Compression rate: {stats['compression_rate']:.2%}")
print(f"Exploitation rate: {stats['exploitation_rate']:.2%}")
print(f"Bandit states: {stats['bandit_states']}")
```

## Profile 库格式

参考 `profiles/llama3.1-8b_example.json`：

```json
{
  "library_metadata": {
    "model_name": "Llama-3.1-8B-Instruct",
    "dataset": "longbench_qasper",
    "created_at": "2026-01-16",
    "num_profiles": 9
  },
  
  "accuracy_buckets": [
    {
      "bucket_id": 0,
      "acc_range": [0.90, 0.93],
      "pareto_profiles": [...],
      "bandwidth_intervals": [
        {
          "interval_id": 0,
          "x_range": [0.0, 0.004],
          "B_range": [250.0, 999999.0],
          "model_optimal_profile_id": "...",
          "candidate_profile_ids": ["...", "..."]
        }
      ]
    }
  ]
}
```

## 集成到 KVServe

### 在 PrefillEngine 中集成

```python
from kvserve.controller import ProfileLibrary, OnlineController

class PrefillEngine:
    def __init__(self, ..., controller_config=None):
        if controller_config:
            library = ProfileLibrary(controller_config['profile_library_path'])
            self.controller = OnlineController(library, epsilon=0.1)
        else:
            self.controller = None
    
    async def step(self):
        # ... prefill 逻辑 ...
        
        if self.controller is not None:
            # 选择压缩策略
            profile, context = self.controller.select_profile(
                V_bytes=calculate_kv_size(request),
                B_mbps=self.config['bandwidth_mbps'],
                T_model_ms=last_compute_time,
                T_SLO_ms=request.slo_ms,
                acc_req=request.accuracy_requirement
            )
            
            # 应用压缩配置
            compression_config = profile.compression_config if profile else None
            
            # 执行传输
            t_start = time.time()
            await transfer_kv(compression_config=compression_config)
            t_end = time.time()
            
            # 更新 Bandit
            T_obs_ms = (t_end - t_start) * 1000 + T_model_ms
            self.controller.update(context, T_obs_ms)
```

## 理论基础

### Theorem 1: Benefit Condition

压缩带来加速的条件：

```
T_p < T_0
=> V/S + V/(B*cr) < V/B
=> B < (1 - 1/cr) * S = B_crit
```

只有当带宽 `B` 小于临界带宽 `B_crit` 时，压缩才能加速。

### Theorem 2: Piecewise-Optimal Policy

在固定精度桶下，最优策略是 `x = 1/B` 空间的分段线性函数（下包络）。

通过预计算带宽区间和每个区间的最优 profile，可以快速找到候选集。

## 性能特性

- **决策延迟**: < 1ms（O(1) 查表 + O(3) 预测）
- **内存占用**: ~5KB（300 个状态 × 16 bytes）
- **状态空间**: ~10 精度桶 × ~10 带宽区间 × ~3 候选 = 300 状态
- **收敛速度**: ε-greedy + EWMA，快速适应

## 测试

运行完整测试套件：

```bash
python test/test_online_controller.py
```

测试包括：
1. Profile 库加载
2. 解析模型预测
3. Bandit 状态管理
4. 端到端控制器
5. 边界情况

## 参数调优

### epsilon（探索率）
- **推荐值**: 0.05 - 0.15
- **高 epsilon**: 更多探索，适合动态环境
- **低 epsilon**: 更多利用，适合稳定环境

### alpha（EWMA 学习率）
- **推荐值**: 0.1 - 0.3
- **高 alpha**: 快速适应，对噪声敏感
- **低 alpha**: 平滑估计，响应慢

### 冷启动策略
- 初始阶段使用更高的 epsilon（如 0.3）
- 经过 N 次请求后逐渐退火到目标值（如 0.1）

## 扩展性

### 添加新 Profile
1. 离线测量新配置的性能指标
2. 更新 Profile 库 JSON
3. 重新加载 ProfileLibrary

### 支持多模型/数据集
- 为每个 (model, dataset) 对准备独立的 Profile 库
- 在运行时根据请求类型选择对应的库

### 集成其他 Bandit 算法
- 替换 `BanditStateManager` 为 LinUCB、Thompson Sampling 等
- 保持 `OnlineController` 接口不变



