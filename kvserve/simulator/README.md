# PD Separation Simulator

完整的 Prefill-Decode 分离模拟系统，用于论文实验验证。

## 功能特性

✅ **独立 vLLM 实例**：Prefill 和 Decode 使用完全独立的 vLLM 引擎（支持 TP）  
✅ **网络传输模拟**：带宽限制、并发控制（默认2）、排队模型、随机抖动  
✅ **KV 存储管理**：CPU 打包/拆包，IO 时间估算（PCIe）  
✅ **事件驱动时间线**：t0 → t_p_end → t_net_arrive → t_d_end  
✅ **详细统计输出**：端到端时延、计算时间、网络时间、IO 时间分解  
✅ **TP 对比测试**：可对比 TP=1 vs TP=2 的性能差异

## 架构

```
Request → [Prefill vLLM] → KV Pack (GPU→CPU) 
                                ↓
                          Network Sim (queue + transfer)
                                ↓
                          KV Unpack (CPU→GPU) → [Decode vLLM] → Output
```

## 使用方法

### 1. 快速测试

```bash
cd /root/lzd/kvserve_project
python test/test_simulator.py
```

默认会测试 TP=1 和 TP=2，输出端到端时延对比。

### 2. 代码示例

```python
import asyncio
from kvserve.simulator import SimulatorBackend

async def main():
    # 创建模拟器
    sim = SimulatorBackend(
        model_path="/path/to/model",
        tensor_parallel_size=2,  # TP配置
        network_gbps=80.0,  # 网络带宽（Gbps）
        max_concurrent_transfers=2,  # 最大并发传输数
    )
    
    # 初始化
    await sim.initialize()
    
    # 运行模拟
    prompts = ["What is AI?", "Explain ML.", "Define DL."]
    results = await sim.run_requests(prompts, max_tokens=50)
    
    # 输出统计
    sim.print_stats(results)

asyncio.run(main())
```

### 3. 配置参数

**网络模拟：**
- `network_gbps`: 带宽（Gbps），默认 80.0（PCIe Gen4 x16）
- `max_concurrent_transfers`: 最大并发数，默认 2
- `network_efficiency`: 有效利用率，默认 0.8
- `network_jitter_ms`: 随机抖动（ms），默认 1.0

**IO 模拟：**
- `pcie_gbps`: PCIe 带宽（GB/s），默认 24.0
- `io_jitter_ms`: IO 抖动（ms），默认 0.5

**vLLM 配置：**
- `tensor_parallel_size`: TP 大小
- `gpu_memory_utilization`: GPU 内存利用率
- `max_model_len`: 最大序列长度

## 输出统计

每个请求输出：
- `t_p_end`: Prefill 完成时间
- `t_net_arrive`: KV 到达 Decode 时间
- `t_d_end`: Decode 完成时间
- `total_latency_ms`: 端到端时延
- `compute_only_ms`: 纯计算时间
- `network_queue_ms`: 网络排队时间
- `network_transfer_ms`: 网络传输时间
- `io_pack_ms`, `io_unpack_ms`: IO 时间

聚合统计：
- 平均时延、计算时间、网络时间、IO 时间
- 各部分时间占比
- TP=1 vs TP=2 加速比

## 模块说明

- `network_simulator.py`: 网络传输模拟（带宽+并发+排队）
- `kv_storage.py`: KV cache 存储管理（CPU 打包/拆包）
- `simulator_backend.py`: 核心模拟器（管理 Prefill/Decode 实例和事件流）

## 实现细节

1. **Prefill 阶段**：使用独立 vLLM 实例，记录 t_p_end，估算 KV 大小，模拟 GPU→CPU 打包
2. **传输模拟**：根据 KV 大小、带宽、当前并发数计算排队和传输时间
3. **Decode 阶段**：模拟 CPU→GPU 拆包，使用独立 vLLM 实例完成解码，记录 t_d_end
4. **统计输出**：汇总所有时间事件，计算端到端时延和各阶段占比

## 注意事项

- 模拟器使用独立的 vLLM 实例，因此 TP=2 时需要 4 张 GPU（Prefill 2 + Decode 2）
- KV 大小估算：当前简化为固定值（64MB），实际可根据 prompt 长度调整
- 时间模型基于带宽和 IO 理论值，加入随机抖动模拟真实环境
- 适用于论文实验的快速验证，不追求生产级的完整性


