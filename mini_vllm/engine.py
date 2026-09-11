"""
Main inference engine for mini-vLLM
"""

import asyncio
import time
from typing import Dict, List, Optional, AsyncIterator, Callable
import torch
import logging

from mini_vllm.config import EngineConfig, ModelConfig, CacheConfig, SchedulerConfig
from mini_vllm.model import ModelWrapper, DummyModelWrapper
from mini_vllm.cache import KVCacheManager
from mini_vllm.batch import BatchManager, Sequence, SequenceStatus
from mini_vllm.scheduler import Scheduler, AsyncScheduler

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    Main inference engine that orchestrates all components.
    Handles request processing, scheduling, and generation.
    """

    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()

        self.model_config = self.config.model
        self.cache_config = self.config.cache
        self.scheduler_config = self.config.scheduler

        self.model_wrapper: Optional[ModelWrapper] = None
        self.kv_cache_manager: Optional[KVCacheManager] = None
        self.batch_manager: Optional[BatchManager] = None
        self.scheduler: Optional[Scheduler] = None

        self._is_running = False
        self._generation_loop_task: Optional[asyncio.Task] = None

        self._callbacks: Dict[str, Callable] = {}

    def initialize(self, use_dummy_model: bool = False) -> None:
        """
        Initialize all engine components.
        
        Args:
            use_dummy_model: If True, use DummyModelWrapper for testing
        """
        logger.info("Initializing mini-vLLM engine...")

        if use_dummy_model:
            self.model_wrapper = DummyModelWrapper(self.model_config)
        else:
            self.model_wrapper = ModelWrapper(self.model_config)
            self.model_wrapper.load_model()

        model_cfg = self.model_wrapper.model_config

        self.kv_cache_manager = KVCacheManager(self.cache_config, model_cfg)

        self.batch_manager = BatchManager(
            max_batch_size=self.scheduler_config.max_batch_size,
            max_num_seqs=self.scheduler_config.max_num_seqs,
        )

        self.scheduler = Scheduler(
            config=self.scheduler_config,
            kv_cache_manager=self.kv_cache_manager,
            batch_manager=self.batch_manager,
        )

        logger.info("Engine initialization complete")

    async def add_request(
        self,
        prompt: str,
        request_id: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
    ) -> str:
        """
        Add a new generation request.
        
        Args:
            prompt: Input text prompt
            request_id: Optional custom request ID
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling parameter
            
        Returns:
            Request ID
        """
        if self.model_wrapper is None:
            raise RuntimeError("Engine not initialized")

        token_ids = self.model_wrapper.encode(prompt)

        seq = self.batch_manager.add_sequence(token_ids, max_tokens)

        if request_id is None:
            request_id = seq.seq_id

        self._callbacks[seq.seq_id] = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        }

        return request_id

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        stream: bool = True,
    ) -> AsyncIterator[str]:
        """
        Generate text from prompt with optional streaming.
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling parameter
            stream: Whether to stream tokens
            
        Yields:
            Generated text tokens
        """
        request_id = await self.add_request(
            prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p, top_k=top_k
        )

        if stream:
            async for token in self._stream_request(request_id):
                yield token
        else:
            output = await self._complete_request(request_id)
            yield output

    async def _stream_request(self, request_id: str) -> AsyncIterator[str]:
        """Stream tokens for a request"""
        while True:
            seq = self.batch_manager.get_sequence(request_id)
            if seq is None:
                break

            if seq.is_finished:
                break

            await asyncio.sleep(0.01)

            if seq.output_tokens:
                last_token = seq.output_tokens[-1]
                token_text = self.model_wrapper.decode([last_token])
                yield token_text

    async def _complete_request(self, request_id: str) -> str:
        """Wait for request completion"""
        while True:
            seq = self.batch_manager.get_sequence(request_id)
            if seq is None:
                return ""

            if seq.is_finished:
                return self.model_wrapper.decode(seq.output_tokens)

            await asyncio.sleep(0.01)

    async def step(self) -> None:
        """
        Execute one step of the generation loop.
        Called by the async generation loop.
        """
        if self.scheduler is None or self.batch_manager is None:
            return

        schedule_result = self.scheduler.schedule()

        finished_ids = []
        new_tokens: Dict[str, int] = {}

        if schedule_result.prefill_batch:
            await self._execute_prefill(schedule_result.prefill_batch)

        if schedule_result.decode_batch:
            finished, new_toks = await self._execute_decode(schedule_result.decode_batch)
            finished_ids.extend(finished)
            new_tokens.update(new_toks)

        self.scheduler.update_after_forward(finished_ids, new_tokens)

    async def _execute_prefill(self, batch: Batch) -> None:
        """Execute prefill phase for a batch"""
        if self.model_wrapper is None:
            return

        for seq in batch.sequences:
            prompt_tokens = torch.tensor(
                [seq.prompt_tokens], dtype=torch.long, device="cuda"
            )

            logits, _ = self.model_wrapper.forward(
                prompt_tokens, self.kv_cache_manager, seq.seq_id
            )

            next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
            seq.append_output_token(next_token)

            if next_token == self.model_wrapper.tokenizer.eos_token_id:
                self.batch_manager.finish_sequence(seq.seq_id, "eos")

    async def _execute_decode(self, batch: Batch) -> Tuple[List[str], Dict[str, int]]:
        """Execute decode phase for a batch"""
        if self.model_wrapper is None:
            return [], {}

        finished_ids = []
        new_tokens: Dict[str, int] = {}

        for seq in batch.sequences:
            if seq.is_finished:
                continue

            input_ids = torch.tensor(
                [[seq.output_tokens[-1]]], dtype=torch.long, device="cuda"
            )

            logits, _ = self.model_wrapper.forward(
                input_ids, self.kv_cache_manager, seq.seq_id
            )

            callback_info = self._callbacks.get(seq.seq_id, {})
            temperature = callback_info.get("temperature", 1.0)
            top_p = callback_info.get("top_p", 1.0)
            top_k = callback_info.get("top_k", 50)

            if temperature == 0:
                next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
            else:
                probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
                if top_k > 0:
                    top_k_vals, top_k_indices = torch.topk(probs, top_k)
                    mask = probs < top_k_vals[:, -1:]
                    probs = torch.where(mask, torch.zeros_like(probs), probs)
                next_token = torch.multinomial(probs, num_samples=1).item()

            seq.append_output_token(next_token)
            new_tokens[seq.seq_id] = next_token

            if next_token == self.model_wrapper.tokenizer.eos_token_id:
                finished_ids.append(seq.seq_id)
            elif seq.num_tokens >= seq.max_tokens:
                finished_ids.append(seq.seq_id)

        return finished_ids, new_tokens

    async def run_generation_loop(self) -> None:
        """Main async generation loop"""
        self._is_running = True

        while self._is_running:
            if self.batch_manager.num_waiting > 0 or self.batch_manager.num_running > 0:
                await self.step()
            else:
                await asyncio.sleep(0.05)

    def start(self) -> None:
        """Start the engine in background task"""
        if self._is_running:
            return
        self._generation_loop_task = asyncio.create_task(self.run_generation_loop())

    def stop(self) -> None:
        """Stop the engine"""
        self._is_running = False
        if self._generation_loop_task:
            self._generation_loop_task.cancel()

    @property
    def stats(self) -> Dict:
        """Get engine statistics"""
        return {
            "num_waiting": self.batch_manager.num_waiting if self.batch_manager else 0,
            "num_running": self.batch_manager.num_running if self.batch_manager else 0,
            "num_finished": len(self.batch_manager.finished) if self.batch_manager else 0,
            "cache_usage": (
                self.kv_cache_manager.cache.block_manager.allocated_blocks
                if self.kv_cache_manager else 0
            ),
            "cache_capacity": self.cache_config.num_gpu_blocks,
        }
