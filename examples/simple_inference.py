"""
Simple inference example with mini-vLLM
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mini_vllm.config import EngineConfig, ModelConfig, CacheConfig, SchedulerConfig
from mini_vllm.engine import InferenceEngine


async def main():
    """Run a simple inference example"""
    print("Initializing mini-vLLM engine...")

    config = EngineConfig(
        model=ModelConfig(model_name="Qwen/Qwen2.5-0.5B"),
        cache=CacheConfig(block_size=16, num_gpu_blocks=256),
        scheduler=SchedulerConfig(max_batch_size=8, max_num_seqs=8),
    )

    engine = InferenceEngine(config)
    engine.initialize(use_dummy_model=True)

    print("\nGenerating completion...")
    print("Prompt: 'What is the capital of France?'")
    print("Response: ", end="", flush=True)

    async for token in engine.generate(
        prompt="What is the capital of France?",
        max_tokens=50,
        temperature=0.7,
        stream=True,
    ):
        print(token, end="", flush=True)

    print("\n\nGeneration complete!")
    print(f"Engine stats: {engine.stats}")


if __name__ == "__main__":
    asyncio.run(main())
