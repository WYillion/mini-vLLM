"""
KV Cache management with block-based paging
"""

from typing import Dict, List, Optional, Tuple
import torch
import numpy as np

from mini_vllm.block import BlockManager
from mini_vllm.config import CacheConfig


class KVCache:
    """
    Key-Value Cache storage with block-based memory management.
    Stores cache as separate tensors for keys and values.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        block_size: int = 16,
        num_blocks: int = 1024,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.dtype = dtype
        self.device = device

        self.block_manager = BlockManager(num_blocks, block_size)

        self._cache: Dict[int, torch.Tensor] = {}
        self._max_seq_len_per_block = {}

        for block_id in range(num_blocks):
            self._allocate_block(block_id)

    def _allocate_block(self, block_id: int) -> None:
        """Allocate GPU memory for a single block."""
        block = torch.zeros(
            2,
            self.num_heads,
            self.block_size,
            self.head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        self._cache[block_id] = block
        self._max_seq_len_per_block[block_id] = 0

    def get_kv_tensor(
        self, block_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the key and value tensors for a block."""
        block = self._cache[block_id]
        return block[0], block[1]

    def write_to_block(
        self,
        block_id: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        offset: int = 0,
    ) -> None:
        """
        Write keys and values to a specific block.
        
        Args:
            block_id: Physical block ID
            keys: Key tensor [num_heads, seq_len, head_dim]
            values: Value tensor [num_heads, seq_len, head_dim]
            offset: Starting position in the block
        """
        if block_id not in self._cache:
            return

        block = self._cache[block_id]
        seq_len = keys.shape[1]

        assert offset + seq_len <= self.block_size, (
            f"Write would exceed block size: {offset + seq_len} > {self.block_size}"
        )

        block[0, :, offset : offset + seq_len, :] = keys
        block[1, :, offset : offset + seq_len, :] = values
        self._max_seq_len_per_block[block_id] = offset + seq_len

    def read_from_block(
        self, block_id: int, offset: int = 0, seq_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Read keys and values from a specific block.
        
        Args:
            block_id: Physical block ID
            offset: Starting position in the block
            seq_len: Number of tokens to read (None = read all)
            
        Returns:
            Tuple of (keys, values) tensors
        """
        if block_id not in self._cache:
            return None, None

        block = self._cache[block_id]
        if seq_len is None:
            seq_len = self._max_seq_len_per_block.get(block_id, 0) or 0

        return block[0, :, offset : offset + seq_len, :], block[1, :, offset : offset + seq_len, :]

    def fork(self, src_block_id: int, dst_block_id: int) -> None:
        """
        Copy content from source block to destination block.
        Used for sequence forking in beam search.
        """
        if src_block_id in self._cache and dst_block_id in self._cache:
            self._cache[dst_block_id].copy_(self._cache[src_block_id])
            self._max_seq_len_per_block[dst_block_id] = self._max_seq_len_per_block.get(src_block_id, 0)


class KVCacheManager:
    """
    Manages KV caches for multiple sequences with block-level allocation.
    Implements PagedAttention memory management.
    """

    def __init__(self, config: CacheConfig, model_config: dict):
        self.config = config
        self.model_config = model_config

        self.num_layers = model_config.get("num_hidden_layers", 32)
        self.num_heads = model_config.get("num_attention_heads", 32)
        self.head_dim = model_config.get("hidden_size", 4096) // self.num_heads

        self.dtype = torch.float16
        self.device = "cuda"

        self.cache = KVCache(
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            block_size=config.block_size,
            num_blocks=config.num_gpu_blocks,
            dtype=self.dtype,
            device=self.device,
        )

        self._seq_block_mapping: Dict[str, List[int]] = {}
        self._seq_len: Dict[str, int] = {}

    def allocate(self, seq_id: str, seq_len: int) -> bool:
        """
        Allocate blocks for a new sequence.
        
        Args:
            seq_id: Unique sequence identifier
            seq_len: Initial sequence length (usually prompt length)
            
        Returns:
            True if allocation successful, False otherwise
        """
        num_blocks_needed = (seq_len + self.config.block_size - 1) // self.config.block_size

        physical_blocks = self.cache.block_manager.allocate(seq_id, num_blocks_needed)
        if physical_blocks is None:
            return False

        self._seq_block_mapping[seq_id] = physical_blocks
        self._seq_len[seq_id] = seq_len
        return True

    def free(self, seq_id: str) -> None:
        """Free all blocks for a sequence."""
        self.cache.block_manager.free(seq_id)
        self._seq_block_mapping.pop(seq_id, None)
        self._seq_len.pop(seq_id, None)

    def append(self, seq_id: str, num_tokens: int) -> bool:
        """
        Allocate additional blocks for sequence continuation.
        
        Args:
            seq_id: Unique sequence identifier
            num_tokens: Number of new tokens to append
            
        Returns:
            True if allocation successful, False otherwise
        """
        if seq_id not in self._seq_block_mapping:
            return False

        current_len = self._seq_len.get(seq_id, 0)
        new_len = current_len + num_tokens
        current_blocks = len(self._seq_block_mapping[seq_id])
        new_blocks_needed = (new_len + self.config.block_size - 1) // self.config.block_size

        if new_blocks_needed > current_blocks:
            additional = new_blocks_needed - current_blocks
            new_physical = self.cache.block_manager.append_blocks(seq_id, additional)
            if new_physical is None:
                return False
            self._seq_block_mapping[seq_id].extend(new_physical)

        self._seq_len[seq_id] = new_len
        return True

    def get_block_locations(self, seq_id: str) -> List[List[int]]:
        """
        Get block IDs and offsets for each token in a sequence.
        
        Args:
            seq_id: Unique sequence identifier
            
        Returns:
            List of [block_id, offset] pairs for each token position
        """
        if seq_id not in self._seq_block_mapping:
            return []

        physical_blocks = self._seq_block_mapping[seq_id]
        seq_len = self._seq_len.get(seq_id, 0)

        locations = []
        for token_idx in range(seq_len):
            block_idx = token_idx // self.config.block_size
            offset = token_idx % self.config.block_size
            if block_idx < len(physical_blocks):
                locations.append([physical_blocks[block_idx], offset])
            else:
                locations.append([-1, -1])

        return locations

    def get_seq_len(self, seq_id: str) -> int:
        """Get current sequence length."""
        return self._seq_len.get(seq_id, 0)
