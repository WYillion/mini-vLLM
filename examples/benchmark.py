"""
Benchmark script for mini-vLLM performance testing
"""

import asyncio
import time
import sys
import os
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mini_vllm.config import EngineConfig, ModelConfig, CacheConfig, SchedulerConfig
from mini_vllm.engine import InferenceEngine


async def benchmark_throughput(
    engine: InferenceEngine,
    prompts: List[str],
    max_tokens: int = 50,
) -> Dict:
    """
    Benchmark throughput (Tokens/s vs Batch Size).
    """
    print(f"\n=== Throughput Benchmark ===")
    print(f"Running {len(prompts)} prompts with max_tokens={max_tokens}")

    start_time = time.time()
    tasks = []

    for prompt in prompts:
        task = engine.add_request(prompt, max_tokens=max_tokens)
        tasks.append(task)

    await asyncio.gather(*tasks)

    engine.start()

    await asyncio.sleep(2)

    engine.stop()

    elapsed = time.time() - start_time
    total_tokens = sum(len(p) + max_tokens for p in prompts)
    tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0

    return {
        "elapsed_time": elapsed,
        "total_tokens": total_tokens,
        "tokens_per_second": tokens_per_sec,
        "num_requests": len(prompts),
    }


async def benchmark_latency(
    engine: InferenceEngine,
    num_requests: int = 10,
    max_tokens: int = 50,
) -> Dict:
    """
    Benchmark latency (P50/P95/P99).
    """
    print(f"\n=== Latency Benchmark ===")
    print(f"Running {num_requests} sequential requests")

    latencies = []

    for i in range(num_requests):
        prompt = f"Tell me a short story about number {i}"
        
        start = time.time()
        tokens_generated = 0
        
        async for _ in engine.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            stream=False,
        ):
            tokens_generated += 1
        
        latency = time.time() - start
        latencies.append(latency * 1000)

    latencies.sort()
    p50 = latencies[int(len(latencies) * 0.50)]
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]
    avg = sum(latencies) / len(latencies)

    return {
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "avg_ms": avg,
        "num_requests": num_requests,
    }


async def main():
    """Run all benchmarks"""
    print("Initializing mini-vLLM engine for benchmarking...")

    config = EngineConfig(
        model=ModelConfig(model_name="Qwen/Qwen2.5-0.5B"),
        cache=CacheConfig(block_size=16, num_gpu_blocks=512),
        scheduler=SchedulerConfig(
            max_batch_size=32,
            max_num_seqs=32,
            max_num_batched_tokens=1024,
        ),
    )

    engine = InferenceEngine(config)
    engine.initialize(use_dummy_model=True)

    prompts = [
        "What is artificial intelligence?",
        "Explain machine learning in simple terms.",
        "What are neural networks?",
        "How does deep learning work?",
        "What is natural language processing?",
        "Describe computer vision.",
        "What is reinforcement learning?",
        "Explain supervised learning.",
    ]

    throughput_results = await benchmark_throughput(engine, prompts, max_tokens=30)
    print(f"\nThroughput Results:")
    print(f"  Elapsed: {throughput_results['elapsed_time']:.2f}s")
    print(f"  Total tokens: {throughput_results['total_tokens']}")
    print(f"  Tokens/s: {throughput_results['tokens_per_second']:.2f}")

    latency_results = await benchmark_latency(engine, num_requests=5, max_tokens=20)
    print(f"\nLatency Results:")
    print(f"  P50: {latency_results['p50_ms']:.2f}ms")
    print(f"  P95: {latency_results['p95_ms']:.2f}ms")
    print(f"  P99: {latency_results['p99_ms']:.2f}ms")
    print(f"  Avg: {latency_results['avg_ms']:.2f}ms")

    print("\n=== Benchmark Complete ===")
    print(f"Final Engine Stats: {engine.stats}")


if __name__ == "__main__":
    asyncio.run(main())
