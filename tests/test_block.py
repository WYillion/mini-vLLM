"""
Tests for BlockManager (PagedAttention memory management)
"""

import pytest
from mini_vllm.block import BlockManager, PhysicalBlock


class TestBlockManager:
    """Test cases for BlockManager"""

    def test_initialization(self):
        """Test block manager initialization"""
        manager = BlockManager(num_blocks=10, block_size=16)
        
        assert manager.num_blocks == 10
        assert manager.block_size == 16
        assert manager.num_free_blocks == 10
        assert manager.allocated_blocks == 0

    def test_allocate_single_sequence(self):
        """Test allocating blocks for a single sequence"""
        manager = BlockManager(num_blocks=10, block_size=16)
        
        blocks = manager.allocate("seq_1", num_blocks=2)
        
        assert blocks is not None
        assert len(blocks) == 2
        assert manager.num_free_blocks == 8
        assert manager.allocated_blocks == 2

    def test_allocate_multiple_sequences(self):
        """Test allocating blocks for multiple sequences"""
        manager = BlockManager(num_blocks=10, block_size=16)
        
        blocks1 = manager.allocate("seq_1", num_blocks=3)
        blocks2 = manager.allocate("seq_2", num_blocks=2)
        
        assert blocks1 is not None
        assert blocks2 is not None
        assert manager.num_free_blocks == 5
        assert manager.allocated_blocks == 5

    def test_allocate_insufficient_blocks(self):
        """Test allocation failure when insufficient blocks"""
        manager = BlockManager(num_blocks=5, block_size=16)
        
        blocks = manager.allocate("seq_1", num_blocks=6)
        
        assert blocks is None
        assert manager.num_free_blocks == 5

    def test_free_sequence(self):
        """Test freeing blocks for a sequence"""
        manager = BlockManager(num_blocks=10, block_size=16)
        
        blocks = manager.allocate("seq_1", num_blocks=3)
        assert manager.num_free_blocks == 7
        
        manager.free("seq_1")
        
        assert manager.num_free_blocks == 10
        assert manager.allocated_blocks == 0

    def test_get_physical_blocks(self):
        """Test retrieving physical blocks for a sequence"""
        manager = BlockManager(num_blocks=10, block_size=16)
        
        blocks = manager.allocate("seq_1", num_blocks=3)
        retrieved = manager.get_physical_blocks("seq_1")
        
        assert retrieved == blocks

    def test_append_blocks(self):
        """Test appending additional blocks to existing sequence"""
        manager = BlockManager(num_blocks=10, block_size=16)
        
        blocks = manager.allocate("seq_1", num_blocks=2)
        initial_free = manager.num_free_blocks
        
        new_blocks = manager.append_blocks("seq_1", 2)
        
        assert new_blocks is not None
        assert len(new_blocks) == 2
        assert manager.num_free_blocks == initial_free - 2

    def test_cache_layout(self):
        """Test cache layout generation"""
        manager = BlockManager(num_blocks=10, block_size=16)
        
        blocks = manager.allocate("seq_1", num_blocks=2)
        layout = manager.get_cache_layout("seq_1", num_tokens=20)
        
        assert len(layout) == 20
        for i, block_id in enumerate(layout[:16]):
            assert block_id == blocks[0]
        for i, block_id in enumerate(layout[16:]):
            assert block_id == blocks[1]

    def test_free_nonexistent_sequence(self):
        """Test freeing non-existent sequence (should not error)"""
        manager = BlockManager(num_blocks=10, block_size=16)
        
        manager.free("nonexistent_seq")
        
        assert manager.num_free_blocks == 10


class TestPhysicalBlock:
    """Test cases for PhysicalBlock"""

    def test_physical_block_creation(self):
        """Test creating a physical block"""
        block = PhysicalBlock(block_id=5)
        
        assert block.block_id == 5
        assert block.ref_count == 0
        assert block.is_allocated is False

    def test_physical_block_with_initial_ref_count(self):
        """Test creating physical block with initial ref count"""
        block = PhysicalBlock(block_id=3, ref_count=1)
        
        assert block.ref_count == 1
        assert block.is_allocated is False
