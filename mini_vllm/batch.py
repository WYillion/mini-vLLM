"""
Sequence and Batch management for continuous batching
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set
import time

import torch


class SequenceStatus(Enum):
    """Status of a sequence in the scheduler"""
    WAITING = "waiting"
    RUNNING = "running"
    DECODING = "decoding"
    FINISHED = "finished"
    CANCELLED = "cancelled"


@dataclass
class Sequence:
    """
    Represents a single sequence in the batch.
    Tracks prompt, generated tokens, and state.
    """
    seq_id: str
    prompt: List[int]
    prompt_tokens: List[int] = field(default_factory=list)
    output_tokens: List[int] = field(default_factory=list)
    status: SequenceStatus = SequenceStatus.WAITING
    block_ids: List[int] = field(default_factory=list)
    num_tokens: int = 0
    max_tokens: int = 256
    created_at: float = field(default_factory=time.time)
    last_token_time: float = field(default_factory=time.time)
    finished_reason: Optional[str] = None

    @property
    def total_len(self) -> int:
        """Total sequence length (prompt + output)"""
        return len(self.prompt_tokens) + len(self.output_tokens)

    @property
    def is_finished(self) -> bool:
        """Check if sequence is in terminal state"""
        return self.status in (SequenceStatus.FINISHED, SequenceStatus.CANCELLED)

    def append_output_token(self, token: int) -> None:
        """Append a new output token"""
        self.output_tokens.append(token)
        self.num_tokens += 1
        self.last_token_time = time.time()

    def get_output_tokens(self) -> List[int]:
        """Get all output tokens"""
        return self.output_tokens

    def truncate_output(self, max_len: int) -> None:
        """Truncate output tokens to max length"""
        if len(self.output_tokens) > max_len:
            self.output_tokens = self.output_tokens[:max_len]


@dataclass
class Batch:
    """
    Represents a batch of sequences for processing.
    Contains both prefill (prompt processing) and decode (token generation) phases.
    """
    batch_id: str
    sequences: List[Sequence] = field(default_factory=list)
    is_prefill: bool = True
    max_model_len: int = 2048

    _seq_id_to_idx: Dict[str, int] = field(default_factory=dict, init=False)

    def __post_init__(self):
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        """Rebuild sequence ID to index mapping"""
        self._seq_id_to_idx = {
            seq.seq_id: idx for idx, seq in enumerate(self.sequences)
        }

    def add_sequence(self, seq: Sequence) -> None:
        """Add a sequence to the batch"""
        self.sequences.append(seq)
        self._seq_id_to_idx[seq.seq_id] = len(self.sequences) - 1

    def remove_sequence(self, seq_id: str) -> Optional[Sequence]:
        """Remove a sequence from the batch"""
        if seq_id not in self._seq_id_to_idx:
            return None
        idx = self._seq_id_to_idx[seq_id]
        seq = self.sequences.pop(idx)
        self._rebuild_index()
        return seq

    def get_sequence(self, seq_id: str) -> Optional[Sequence]:
        """Get sequence by ID"""
        idx = self._seq_id_to_idx.get(seq_id)
        return self.sequences[idx] if idx is not None else None

    @property
    def num_seqs(self) -> int:
        """Number of sequences in batch"""
        return len(self.sequences)

    @property
    def num_running_seqs(self) -> int:
        """Number of running (non-finished) sequences"""
        return sum(1 for s in self.sequences if not s.is_finished)

    @property
    def total_num_tokens(self) -> int:
        """Total tokens across all sequences"""
        return sum(s.total_len for s in self.sequences)

    @property
    def prompt_len(self) -> int:
        """Maximum prompt length in batch"""
        return max((len(s.prompt_tokens) for s in self.sequences), default=0)

    @property
    def max_output_len(self) -> int:
        """Maximum output length in batch"""
        return max((len(s.output_tokens) for s in self.sequences), default=0)

    def get_running_sequences(self) -> List[Sequence]:
        """Get all non-finished sequences"""
        return [s for s in self.sequences if not s.is_finished]

    def finish_sequence(self, seq_id: str, reason: str = "stop") -> None:
        """Mark a sequence as finished"""
        seq = self.get_sequence(seq_id)
        if seq:
            seq.status = SequenceStatus.FINISHED
            seq.finished_reason = reason

    def update_status(self, seq_id: str, status: SequenceStatus) -> None:
        """Update sequence status"""
        seq = self.get_sequence(seq_id)
        if seq:
            seq.status = status


class BatchManager:
    """
    Manages batches of sequences for continuous batching.
    Handles prefill/decode mixing and batch composition.
    """

    def __init__(self, max_batch_size: int = 256, max_num_seqs: int = 256):
        self.max_batch_size = max_batch_size
        self.max_num_seqs = max_num_seqs

        self.waiting: List[Sequence] = []
        self.running: List[Sequence] = []
        self.finished: Set[str] = set()

        self._seq_counter = 0

    def add_sequence(
        self,
        prompt: List[int],
        max_tokens: int = 256,
    ) -> Sequence:
        """
        Add a new sequence to the waiting queue.
        
        Args:
            prompt: List of token IDs for the prompt
            max_tokens: Maximum tokens to generate
            
        Returns:
            Created Sequence object
        """
        self._seq_counter += 1
        seq_id = f"seq_{self._seq_counter}"

        seq = Sequence(
            seq_id=seq_id,
            prompt=[],
            prompt_tokens=prompt,
            max_tokens=max_tokens,
        )

        self.waiting.append(seq)
        return seq

    def can_allocate(self, num_new_seqs: int, num_new_tokens: int) -> bool:
        """Check if we can allocate resources for new sequences"""
        return (
            len(self.running) + num_new_seqs <= self.max_num_seqs
        )

    def get_prefill_batch(self, max_tokens: int = 512) -> Optional[Batch]:
        """
        Get a batch for prefill processing (prompt phase).
        
        Args:
            max_tokens: Maximum tokens to include in prefill
            
        Returns:
            Batch for prefill, or None if no sequences waiting
        """
        if not self.waiting:
            return None

        prefill_seqs = []
        total_tokens = 0

        for seq in self.waiting[:]:
            seq_tokens = len(seq.prompt_tokens)
            if total_tokens + seq_tokens <= max_tokens:
                prefill_seqs.append(seq)
                total_tokens += seq_tokens
                self.waiting.remove(seq)

        if not prefill_seqs:
            return None

        batch = Batch(
            batch_id=f"prefill_{self._seq_counter}",
            sequences=prefill_seqs,
            is_prefill=True,
        )

        for seq in prefill_seqs:
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)

        return batch

    def get_decode_batch(self, max_seqs: int = 32) -> Optional[Batch]:
        """
        Get a batch for decode processing (token generation phase).
        
        Args:
            max_seqs: Maximum sequences to include
            
        Returns:
            Batch for decode, or None if no running sequences
        """
        running_seqs = [s for s in self.running if not s.is_finished]
        if not running_seqs:
            return None

        decode_seqs = running_seqs[:max_seqs]

        batch = Batch(
            batch_id=f"decode_{self._seq_counter}",
            sequences=decode_seqs,
            is_prefill=False,
        )

        return batch

    def free_finished_sequences(self) -> List[Sequence]:
        """Remove finished sequences and return them"""
        finished = []
        still_running = []

        for seq in self.running:
            if seq.is_finished:
                finished.append(seq)
                self.finished.add(seq.seq_id)
            else:
                still_running.append(seq)

        self.running = still_running
        return finished

    def get_sequence(self, seq_id: str) -> Optional[Sequence]:
        """Get sequence by ID from waiting or running lists"""
        for seq in self.waiting:
            if seq.seq_id == seq_id:
                return seq
        for seq in self.running:
            if seq.seq_id == seq_id:
                return seq
        return None

    def finish_sequence(self, seq_id: str, reason: str = "stop") -> None:
        """Mark a sequence as finished"""
        for seq in self.running:
            if seq.seq_id == seq_id:
                seq.status = SequenceStatus.FINISHED
                seq.finished_reason = reason
                return

    @property
    def num_waiting(self) -> int:
        """Number of sequences waiting to be processed"""
        return len(self.waiting)

    @property
    def num_running(self) -> int:
        """Number of sequences currently running"""
        return len(self.running)
