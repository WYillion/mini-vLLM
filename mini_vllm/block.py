"""
Block-based memory management for KV Cache (PagedAttention)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set
import torch


@dataclass
class PhysicalBlock:
    """Represents a physical memory block"""

    block_id: int
    ref_count: int = 0
    is_allocated: bool = False


class BlockManager:
    """
    Manages physical block allocation and virtual-to-physical mapping.
    Implements reference counting for efficient block reuse.
    """

    def __init__(self, num_blocks: int, block_size: int = 16):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._physical_blocks: Dict[int, PhysicalBlock] = {
            i: PhysicalBlock(block_id=i) for i in range(num_blocks)
        }
        self._free_blocks: Set[int] = set(range(num_blocks))
        self._virtual_to_physical: Dict[str, List[int]] = {}
        self._physical_to_virtual: Dict[int, str] = {}

    def allocate(self, seq_id: str, num_blocks: int) -> Optional[List[int]]:
        """
        Allocate physical blocks for a sequence.
        
        Args:
            seq_id: Unique sequence identifier
            num_blocks: Number of blocks needed
            
        Returns:
            List of physical block IDs, or None if insufficient memory
        """
        if len(self._free_blocks) < num_blocks:
            return None

        allocated = []
        for _ in range(num_blocks):
            block_id = self._free_blocks.pop()
            self._physical_blocks[block_id].is_allocated = True
            self._physical_blocks[block_id].ref_count = 1
            allocated.append(block_id)

        self._virtual_to_physical[seq_id] = allocated
        for pb_id in allocated:
            self._physical_to_virtual[pb_id] = seq_id

        return allocated

    def free(self, seq_id: str) -> None:
        """
        Free all blocks allocated for a sequence.
        
        Args:
            seq_id: Unique sequence identifier
        """
        if seq_id not in self._virtual_to_physical:
            return

        for pb_id in self._virtual_to_physical[seq_id]:
            if pb_id in self._physical_blocks:
                self._physical_blocks[pb_id].ref_count -= 1
                if self._physical_blocks[pb_id].ref_count <= 0:
                    self._physical_blocks[pb_id].is_allocated = False
                    self._physical_blocks[pb_id].ref_count = 0
                    self._free_blocks.add(pb_id)
                    self._physical_to_virtual.pop(pb_id, None)

        del self._virtual_to_physical[seq_id]

    def get_physical_blocks(self, seq_id: str) -> List[int]:
        """Get physical block IDs for a sequence."""
        return self._virtual_to_physical.get(seq_id, [])

    def append_blocks(self, seq_id: str, num_new_blocks: int) -> Optional[List[int]]:
        """
        Allocate additional blocks for an existing sequence.
        
        Args:
            seq_id: Unique sequence identifier
            num_new_blocks: Number of additional blocks needed
            
        Returns:
            List of new physical block IDs, or None if insufficient memory
        """
        if seq_id not in self._virtual_to_physical:
            return None

        if len(self._free_blocks) < num_new_blocks:
            return None

        allocated = []
        for _ in range(num_new_blocks):
            block_id = self._free_blocks.pop()
            self._physical_blocks[block_id].is_allocated = True
            self._physical_blocks[block_id].ref_count = 1
            allocated.append(block_id)

        self._virtual_to_physical[seq_id].extend(allocated)
        for pb_id in allocated:
            self._physical_to_virtual[pb_id] = seq_id

        return allocated

    @property
    def num_free_blocks(self) -> int:
        """Get number of currently free blocks."""
        return len(self._free_blocks)

    @property
    def allocated_blocks(self) -> int:
        """Get number of currently allocated blocks."""
        return self.num_blocks - len(self._free_blocks)

    def get_cache_layout(
        self, seq_id: str, num_tokens: int
    ) -> List[int]:
        """
        Get the physical block indices for caching tokens.
        
        Args:
            seq_id: Unique sequence identifier
            num_tokens: Number of tokens to cache
            
        Returns:
            List of physical block indices for each token position
        """
        physical_blocks = self.get_physical_blocks(seq_id)
        if not physical_blocks:
            return []

        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        if num_blocks_needed > len(physical_blocks):
            return []

        layout = []
        for token_idx in range(num_tokens):
            block_idx = token_idx // self.block_size
            offset = token_idx % self.block_size
            if block_idx < len(physical_blocks):
                layout.append(physical_blocks[block_idx])

        return layout
