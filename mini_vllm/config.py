"""
Configuration management for mini-vLLM
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for the language model"""

    model_name: str = "Qwen/Qwen2.5-0.5B"
    max_model_len: int = 2048
    dtype: str = "fp16"
    device: str = "cuda"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9


@dataclass
class CacheConfig:
    """Configuration for KV Cache management"""

    block_size: int = 16
    num_gpu_blocks: int = 1024
    num_cpu_blocks: int = 2048
    enable_prefix_caching: bool = False
    warmup_blocks: int = 0


@dataclass
class SchedulerConfig:
    """Configuration for the continuous batching scheduler"""

    max_batch_size: int = 256
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192
    prefill_chunk_size: int = 512
    enable_chunked_prefill: bool = True
    num_scheduler_loops: int = 1


@dataclass
class EngineConfig:
    """Overall configuration for the inference engine"""

    model: ModelConfig = field(default_factory=ModelConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    trust_remote_code: bool = True
    download_dir: Optional[str] = None
    enforce_eager: bool = False
