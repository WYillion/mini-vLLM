# Mini-vLLM

从零实现的简化版 vLLM 推理框架。

## 核心特性

- **PagedAttention**：基于 Block 分页的 KV Cache 管理，解决内存碎片化问题
- **Continuous Batching**：Iteration 级动态批调度，提升 GPU 利用率
- **REST API**：基于 FastAPI 的推理服务，支持 OpenAI 兼容接口
- **流式输出**：Server-Sent Events (SSE) 实现 Token 级实时推送

## 安装

```bash
pip install -e .
```

## 快速开始

```python
from mini_vllm.config import EngineConfig, ModelConfig, CacheConfig, SchedulerConfig
from mini_vllm.engine import InferenceEngine
import asyncio

async def main():
    config = EngineConfig(
        model=ModelConfig(model_name="Qwen/Qwen2.5-0.5B"),
        cache=CacheConfig(block_size=16, num_gpu_blocks=256),
        scheduler=SchedulerConfig(max_batch_size=8, max_num_seqs=8),
    )
    
    engine = InferenceEngine(config)
    engine.initialize(use_dummy_model=True)
    
    async for token in engine.generate("你好，世界！", stream=True):
        print(token, end="", flush=True)

asyncio.run(main())
```

## 项目结构

```
mini_vllm/
├── __init__.py          # 包初始化，导出主要类
├── config.py            # 配置管理（Model/Cache/Scheduler）
├── block.py             # Block 分页管理（PagedAttention 核心）
├── cache.py             # KV Cache 管理
├── attention.py         # PagedAttention 实现
├── batch.py             # Sequence 和 Batch 管理
├── scheduler.py         # Continuous Batching 调度器
├── model.py             # 模型封装（HuggingFace）
├── engine.py            # 主推理引擎
└── api/
    ├── __init__.py
    └── server.py        # FastAPI REST API 服务

tests/
├── test_block.py        # Block 管理器单元测试
└── test_batch.py        # Batch 管理单元测试

examples/
├── simple_inference.py  # 简单推理示例
└── benchmark.py         # 性能基准测试脚本
```

## 架构设计

### PagedAttention

传统 KV Cache 使用连续内存块分配，当序列长度不一致时会产生内存碎片。PagedAttention 通过以下方式解决：

1. 将 KV Cache 按固定大小 block 组织（默认：16 tokens/block）
2. 动态建立虚拟位置到物理 block 的映射关系
3. 基于引用计数实现 block 高效回收复用

### Continuous Batching

静态批处理的 batch 大小在推理开始时固定，而 Continuous Batching：

1. 每个 iteration 动态将新序列加入运行中的 batch
2. 即时移除已完成的序列
3. 通过保持 batch 尽可能满来最大化 GPU 利用率

### API 接口

| 接口 | 说明 |
|------|------|
| `POST /v1/completions` | 文本补全（OpenAI 兼容） |
| `POST /v1/chat/completions` | 对话补全（OpenAI 兼容） |
| `GET /health` | 健康检查 |
| `GET /stats` | 引擎运行统计 |
| `GET /v1/models` | 可用模型列表 |

## 启动服务

```bash
python -m mini_vllm.api.server
```

或使用 uvicorn 直接启动：

```bash
uvicorn mini_vllm.api.server:app --host 0.0.0.0 --port 8000
```

## 配置参数

### ModelConfig

| 参数 | 默认值 | 说明 |
|------|--------|------|
| model_name | Qwen/Qwen2.5-0.5B | HuggingFace 模型名称 |
| max_model_len | 2048 | 最大模型序列长度 |
| dtype | fp16 | 模型数据类型（fp32/fp16/bf16/int8） |
| device | cuda | 运行设备 |
| tensor_parallel_size | 1 | 张量并行的 GPU 数量 |
| gpu_memory_utilization | 0.9 | GPU 显存使用比例 |

### CacheConfig

| 参数 | 默认值 | 说明 |
|------|--------|------|
| block_size | 16 | 每个 KV Cache block 的 token 数 |
| num_gpu_blocks | 1024 | 分配的 GPU block 数量 |
| num_cpu_blocks | 2048 | CPU block 数量（用于 offloading） |

### SchedulerConfig

| 参数 | 默认值 | 说明 |
|------|--------|------|
| max_batch_size | 256 | 最大 batch 大小 |
| max_num_seqs | 256 | 单个 batch 最大序列数 |
| max_num_batched_tokens | 8192 | 单个 batch 最大 token 数 |
| prefill_chunk_size | 512 | Chunked Prefill 的 chunk 大小 |
| enable_chunked_prefill | True | 是否启用 chunked prefill |

## 许可证

MIT
