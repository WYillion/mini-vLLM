"""
PagedAttention implementation with block-level memory access
"""

import math
from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F

from mini_vllm.cache import KVCacheManager


class PagedAttention:
    """
    PagedAttention implementation that enables block-level KV cache access.
    Reduces memory fragmentation by using physical blocks with dynamic mapping.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: Optional[float] = None,
        block_size: int = 16,
    ):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale or (1.0 / math.sqrt(head_dim))
        self.block_size = block_size

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        block_locations: Optional[List[List[int]]] = None,
        seq_lens: Optional[List[int]] = None,
        kv_cache_manager: Optional[KVCacheManager] = None,
        seq_id: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Compute attention with block-level KV cache access.
        
        Args:
            query: Query tensor [batch_size, num_heads, seq_len, head_dim]
            key: Key tensor [batch_size, num_heads, seq_len, head_dim]
            value: Value tensor [batch_size, num_heads, seq_len, head_dim]
            block_locations: List of [block_id, offset] for each token
            seq_lens: List of sequence lengths for each query
            kv_cache_manager: KVCacheManager instance for reading cached K/V
            seq_id: Sequence ID for cache lookup
            
        Returns:
            Attention output tensor [batch_size, num_heads, seq_len, head_dim]
        """
        batch_size, num_heads, seq_len, head_dim = query.shape

        if seq_lens is None:
            seq_lens = [seq_len] * batch_size

        if kv_cache_manager is not None and seq_id is not None and block_locations is not None:
            key, value = self._read_from_cache(
                kv_cache_manager, seq_id, block_locations, seq_lens
            )

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        attn_weights = torch.matmul(query, key.transpose(-2, -1))
        attn_weights = attn_weights * self.scale

        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2)

        return attn_output

    def _read_from_cache(
        self,
        kv_cache_manager: KVCacheManager,
        seq_id: str,
        block_locations: List[List[int]],
        seq_lens: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Read keys and values from KV cache using block locations.
        
        Args:
            kv_cache_manager: KVCacheManager instance
            seq_id: Sequence identifier
            block_locations: List of [block_id, offset] pairs
            seq_lens: List of sequence lengths
            
        Returns:
            Tuple of (keys, values) tensors retrieved from cache
        """
        total_len = sum(seq_lens)
        device = self.block_size  # Placeholder for actual device detection

        keys = torch.zeros(
            1, self.num_heads, total_len, self.head_dim,
            dtype=torch.float16, device="cuda"
        )
        values = torch.zeros(
            1, self.num_heads, total_len, self.head_dim,
            dtype=torch.float16, device="cuda"
        )

        current_pos = 0
        for seq_idx, seq_len in enumerate(seq_lens):
            for token_pos in range(seq_len):
                if token_pos < len(block_locations):
                    block_id, offset = block_locations[token_pos]
                    if block_id >= 0:
                        k, v = kv_cache_manager.cache.read_from_block(
                            block_id, offset, 1
                        )
                        if k is not None:
                            keys[0, :, current_pos, :] = k[:, 0, :]
                            values[0, :, current_pos, :] = v[:, 0, :]
                current_pos += 1

        return keys, values

    def forward_with_paging(
        self,
        query: torch.Tensor,
        kv_cache_manager: KVCacheManager,
        seq_id: str,
        block_locations: List[List[int]],
    ) -> torch.Tensor:
        """
        Forward pass using cached KV values from paged memory.
        
        Args:
            query: Query tensor [batch_size, num_heads, seq_len, head_dim]
            kv_cache_manager: KVCacheManager for cache access
            seq_id: Sequence identifier
            block_locations: Block and offset for each position
            
        Returns:
            Attention output
        """
        seq_lens = [len(block_locations)]
        k_cache, v_cache = self._read_from_cache(
            kv_cache_manager, seq_id, block_locations, seq_lens
        )

        return self.forward(
            query, k_cache, v_cache,
            block_locations=block_locations,
            seq_lens=seq_lens,
        )


class FlashPagedAttention:
    """
    FlashAttention-style implementation with PagedAttention memory layout.
    Provides memory-efficient attention computation.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        block_size: int = 16,
        num_blocks: int = 1024,
    ):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.scale = 1.0 / math.sqrt(head_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        block_locations: List[List[int]],
        output_lengths: List[int],
    ) -> torch.Tensor:
        """
        Memory-efficient attention with block-level access pattern.
        
        Args:
            query: [batch_size, num_heads, seq_len, head_dim]
            key: [batch_size, num_heads, seq_len, head_dim]
            value: [batch_size, num_heads, seq_len, head_dim]
            block_locations: Physical block mapping
            output_lengths: Length of each output sequence
            
        Returns:
            Attention output
        """
        batch_size, num_heads, seq_len, head_dim = query.shape

        query = query * self.scale

        attn_scores = torch.zeros(
            batch_size, num_heads, seq_len, seq_len,
            device=query.device, dtype=query.dtype
        )

        for block_idx in range((seq_len + self.block_size - 1) // self.block_size):
            start_idx = block_idx * self.block_size
            end_idx = min(start_idx + self.block_size, seq_len)

            k_block = key[:, :, start_idx:end_idx, :]
            q_expanded = query.unsqueeze(2)

            block_scores = torch.matmul(q_expanded, k_block.transpose(-2, -1))
            block_scores = block_scores.squeeze(2)

            attn_scores[:, :, :, start_idx:end_idx] = block_scores

        attn_weights = F.softmax(attn_scores, dim=-1)

        attn_output = torch.matmul(attn_weights, value)

        return attn_output
