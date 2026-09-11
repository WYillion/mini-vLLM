"""
Continuous Batching Scheduler for iteration-level dynamic batch scheduling
"""

import asyncio
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import torch

from mini_vllm.batch import Batch, BatchManager, Sequence, SequenceStatus
from mini_vllm.cache import KVCacheManager
from mini_vllm.config import SchedulerConfig


@dataclass
class SchedulingResult:
    """Result of a scheduling decision"""
    prefill_batch: Optional[Batch] = None
    decode_batch: Optional[Batch] = None
    num_scheduled_tokens: int = 0
    num_scheduled_seqs: int = 0
    scheduled: List[str] = field(default_factory=list)
    unscheduled: List[str] = field(default_factory=list)


class Scheduler:
    """
    Continuous Batching Scheduler implementing iteration-level scheduling.
    Dynamically decides which sequences to process at each step.
    """

    def __init__(
        self,
        config: SchedulerConfig,
        kv_cache_manager: KVCacheManager,
        batch_manager: BatchManager,
    ):
        self.config = config
        self.kv_cache_manager = kv_cache_manager
        self.batch_manager = batch_manager

        self.prefill_enabled = True
        self.decode_enabled = True

        self._scheduled_seqs: Dict[str, float] = {}
        self._last_scheduling_time: float = 0

    def schedule(self) -> SchedulingResult:
        """
        Main scheduling function called every iteration.
        Decides which sequences to include in prefill and decode phases.
        
        Returns:
            SchedulingResult containing batch information
        """
        self._last_scheduling_time = time.time()
        scheduled = []
        unscheduled = []

        prefill_batch = None
        decode_batch = None

        if self.prefill_enabled and self.batch_manager.num_waiting > 0:
            prefill_batch = self._schedule_prefill()
            if prefill_batch:
                scheduled.extend([s.seq_id for s in prefill_batch.sequences])

        if self.decode_enabled:
            decode_batch = self._schedule_decode()
            if decode_batch:
                scheduled.extend([s.seq_id for s in decode_batch.sequences])

        if not prefill_batch and not decode_batch:
            prefill_batch = self._get_minimal_decode_batch()

        num_scheduled_tokens = 0
        if prefill_batch:
            num_scheduled_tokens += prefill_batch.total_num_tokens
        if decode_batch:
            num_scheduled_tokens += decode_batch.num_seqs

        num_scheduled_seqs = len(scheduled)

        return SchedulingResult(
            prefill_batch=prefill_batch,
            decode_batch=decode_batch,
            num_scheduled_tokens=num_scheduled_tokens,
            num_scheduled_seqs=num_scheduled_seqs,
            scheduled=scheduled,
            unscheduled=unscheduled,
        )

    def _schedule_prefill(self) -> Optional[Batch]:
        """
        Schedule sequences for prefill (prompt processing) phase.
        Uses chunked prefill if enabled to limit memory usage.
        
        Returns:
            Batch for prefill, or None if scheduling not possible
        """
        max_tokens = self.config.max_num_batched_tokens
        chunk_size = self.config.prefill_chunk_size

        if self.config.enable_chunked_prefill:
            max_tokens = min(max_tokens, chunk_size)

        prefill_batch = self.batch_manager.get_prefill_batch(max_tokens)

        if prefill_batch and len(prefill_batch.sequences) > 0:
            for seq in prefill_batch.sequences:
                if not self.kv_cache_manager.allocate(seq.seq_id, len(seq.prompt_tokens)):
                    prefill_batch.remove_sequence(seq.seq_id)

            if prefill_batch.num_seqs == 0:
                return None

        return prefill_batch

    def _schedule_decode(self) -> Optional[Batch]:
        """
        Schedule sequences for decode (token generation) phase.
        Limits by max_batch_size to control memory usage.
        
        Returns:
            Batch for decode, or None if no sequences to decode
        """
        max_seqs = min(
            self.config.max_batch_size,
            self.batch_manager.num_running
        )

        if max_seqs == 0:
            return None

        running = [s for s in self.batch_manager.running if not s.is_finished]
        if not running:
            return None

        decode_seqs = running[:max_seqs]

        for seq in decode_seqs:
            if not self.kv_cache_manager.append(seq.seq_id, 1):
                pass

        batch = Batch(
            batch_id=f"decode_{int(time.time() * 1000)}",
            sequences=decode_seqs,
            is_prefill=False,
        )

        return batch

    def _get_minimal_decode_batch(self) -> Optional[Batch]:
        """Get a minimal decode batch to keep GPU busy"""
        running = [s for s in self.batch_manager.running if not s.is_finished]
        if running:
            return Batch(
                batch_id=f"decode_minimal_{int(time.time() * 1000)}",
                sequences=[running[0]],
                is_prefill=False,
            )
        return None

    def update_after_forward(
        self,
        finished_seq_ids: List[str],
        new_token_ids: Dict[str, int],
    ) -> None:
        """
        Update scheduler state after forward pass.
        
        Args:
            finished_seq_ids: Sequence IDs that finished generation
            new_token_ids: Mapping of seq_id to new generated token
        """
        for seq_id, token_id in new_token_ids.items():
            seq = self.batch_manager.get_sequence(seq_id)
            if seq:
                seq.append_output_token(token_id)

                if token_id == 151643:
                    self.batch_manager.finish_sequence(seq_id, "eos")
                elif seq.num_tokens >= seq.max_tokens:
                    self.batch_manager.finish_sequence(seq_id, "length")

        for seq_id in finished_seq_ids:
            self.batch_manager.finish_sequence(seq_id, "stop")

        self.batch_manager.free_finished_sequences()

    def can_schedule_more(self) -> bool:
        """Check if more sequences can be scheduled"""
        return (
            self.batch_manager.num_waiting > 0
            or self.batch_manager.num_running > 0
        )

    @property
    def num_blocks_available(self) -> int:
        """Get number of available KV cache blocks"""
        return self.kv_cache_manager.cache.block_manager.num_free_blocks


class AsyncScheduler:
    """
    Asynchronous wrapper for the scheduler.
    Provides async scheduling for concurrent request handling.
    """

    def __init__(self, scheduler: Scheduler):
        self.scheduler = scheduler
        self._lock = asyncio.Lock()
        self._scheduling_task: Optional[asyncio.Task] = None

    async def schedule_async(self) -> SchedulingResult:
        """Async wrapper for schedule()"""
        async with self._lock:
            return self.scheduler.schedule()

    async def add_sequence_async(
        self,
        prompt: List[int],
        max_tokens: int = 256,
    ) -> Sequence:
        """Async wrapper for adding a sequence"""
        async with self._lock:
            return self.scheduler.batch_manager.add_sequence(prompt, max_tokens)

    async def update_after_forward_async(
        self,
        finished_seq_ids: List[str],
        new_token_ids: Dict[str, int],
    ) -> None:
        """Async wrapper for update_after_forward()"""
        async with self._lock:
            self.scheduler.update_after_forward(finished_seq_ids, new_token_ids)
