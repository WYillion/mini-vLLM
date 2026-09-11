"""
Tests for batch management
"""

import pytest
import time
from mini_vllm.batch import Sequence, SequenceStatus, Batch, BatchManager


class TestSequence:
    """Test cases for Sequence dataclass"""

    def test_sequence_creation(self):
        """Test creating a sequence"""
        seq = Sequence(seq_id="test_1", prompt=[1, 2, 3], prompt_tokens=[1, 2, 3])
        
        assert seq.seq_id == "test_1"
        assert seq.status == SequenceStatus.WAITING
        assert len(seq.output_tokens) == 0
        assert seq.num_tokens == 0

    def test_append_output_token(self):
        """Test appending output token"""
        seq = Sequence(seq_id="test_1", prompt=[], prompt_tokens=[1, 2])
        
        seq.append_output_token(100)
        seq.append_output_token(200)
        
        assert len(seq.output_tokens) == 2
        assert seq.output_tokens == [100, 200]
        assert seq.num_tokens == 2

    def test_total_len(self):
        """Test total length calculation"""
        seq = Sequence(
            seq_id="test_1",
            prompt=[],
            prompt_tokens=[1, 2, 3],
            output_tokens=[10, 20],
        )
        
        assert seq.total_len == 5

    def test_is_finished(self):
        """Test is_finished property"""
        seq = Sequence(seq_id="test_1", prompt=[])
        
        assert not seq.is_finished
        
        seq.status = SequenceStatus.FINISHED
        assert seq.is_finished
        
        seq.status = SequenceStatus.CANCELLED
        assert seq.is_finished


class TestBatch:
    """Test cases for Batch"""

    def test_batch_creation(self):
        """Test creating a batch"""
        seq1 = Sequence(seq_id="s1", prompt=[])
        seq2 = Sequence(seq_id="s2", prompt=[])
        
        batch = Batch(batch_id="batch_1", sequences=[seq1, seq2])
        
        assert batch.batch_id == "batch_1"
        assert batch.num_seqs == 2

    def test_add_sequence(self):
        """Test adding sequence to batch"""
        batch = Batch(batch_id="batch_1")
        seq = Sequence(seq_id="s1", prompt=[])
        
        batch.add_sequence(seq)
        
        assert batch.num_seqs == 1

    def test_remove_sequence(self):
        """Test removing sequence from batch"""
        seq1 = Sequence(seq_id="s1", prompt=[])
        seq2 = Sequence(seq_id="s2", prompt=[])
        batch = Batch(batch_id="batch_1", sequences=[seq1, seq2])
        
        removed = batch.remove_sequence("s1")
        
        assert removed == seq1
        assert batch.num_seqs == 1

    def test_get_sequence(self):
        """Test retrieving sequence by ID"""
        seq1 = Sequence(seq_id="s1", prompt=[])
        seq2 = Sequence(seq_id="s2", prompt=[])
        batch = Batch(batch_id="batch_1", sequences=[seq1, seq2])
        
        retrieved = batch.get_sequence("s2")
        
        assert retrieved == seq2

    def test_num_running_seqs(self):
        """Test counting running sequences"""
        seq1 = Sequence(seq_id="s1", prompt=[], status=SequenceStatus.RUNNING)
        seq2 = Sequence(seq_id="s2", prompt=[], status=SequenceStatus.FINISHED)
        seq3 = Sequence(seq_id="s3", prompt=[], status=SequenceStatus.RUNNING)
        batch = Batch(batch_id="batch_1", sequences=[seq1, seq2, seq3])
        
        assert batch.num_running_seqs == 2


class TestBatchManager:
    """Test cases for BatchManager"""

    def test_batch_manager_creation(self):
        """Test creating batch manager"""
        manager = BatchManager(max_batch_size=10, max_num_seqs=10)
        
        assert manager.max_batch_size == 10
        assert manager.max_num_seqs == 10
        assert manager.num_waiting == 0
        assert manager.num_running == 0

    def test_add_sequence(self):
        """Test adding sequence to batch manager"""
        manager = BatchManager()
        
        seq = manager.add_sequence([1, 2, 3], max_tokens=100)
        
        assert seq is not None
        assert len(seq.prompt_tokens) == 3
        assert seq.max_tokens == 100
        assert manager.num_waiting == 1

    def test_get_prefill_batch(self):
        """Test getting prefill batch"""
        manager = BatchManager()
        
        manager.add_sequence([1, 2, 3])
        manager.add_sequence([4, 5])
        
        batch = manager.get_prefill_batch(max_tokens=10)
        
        assert batch is not None
        assert batch.num_seqs == 2
        assert batch.is_prefill is True
        assert manager.num_waiting == 0

    def test_get_decode_batch(self):
        """Test getting decode batch"""
        manager = BatchManager()
        
        seq = manager.add_sequence([1, 2, 3])
        manager.get_prefill_batch()
        
        batch = manager.get_decode_batch()
        
        assert batch is not None
        assert batch.is_prefill is False
