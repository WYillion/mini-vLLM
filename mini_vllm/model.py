"""
Model wrapper and generation utilities
"""

import time
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import logging

from mini_vllm.config import ModelConfig
from mini_vllm.cache import KVCacheManager
from mini_vllm.batch import Batch, Sequence
from mini_vllm.attention import PagedAttention

logger = logging.getLogger(__name__)


class ModelWrapper:
    """
    Wrapper around HuggingFace model for use with mini-vLLM.
    Handles model loading, forward passes, and generation.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = torch.device(config.device)
        self.dtype = self._get_dtype(config.dtype)

        self._paged_attn: Optional[PagedAttention] = None
        self._model_config: Optional[AutoConfig] = None

    def _get_dtype(self, dtype_str: str) -> torch.dtype:
        """Convert dtype string to torch dtype"""
        dtype_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "int8": torch.int8,
        }
        return dtype_map.get(dtype_str, torch.float16)

    def load_model(self) -> None:
        """Load the model and tokenizer"""
        logger.info(f"Loading model: {self.config.model_name}")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=self.dtype,
            trust_remote_code=self.config.trust_remote_code,
            device_map=self.config.device,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._model_config = self.model.config
        logger.info(f"Model loaded successfully. Config: {self._model_config}")

    @property
    def model_config(self) -> Dict[str, Any]:
        """Get model configuration as dict"""
        if self._model_config is None:
            return {}
        return {
            "num_hidden_layers": self._model_config.num_hidden_layers,
            "num_attention_heads": self._model_config.num_attention_heads,
            "hidden_size": self._model_config.hidden_size,
            "vocab_size": self._model_config.vocab_size,
        }

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Generate tokens autoregressively.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling parameter
            
        Returns:
            Tuple of (output_ids, generated_token_list)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        batch_size, seq_len = input_ids.shape
        generated_ids = []
        past_key_values = None

        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

            if temperature == 0:
                next_token_id = torch.argmax(logits, dim=-1)
            else:
                logits = logits / temperature
                if top_k > 0:
                    top_k_vals, top_k_indices = torch.topk(logits, top_k)
                    logits = torch.where(
                        logits < top_k_vals[:, -1:],
                        torch.full_like(logits, float('-inf')),
                        logits,
                    )
                probs = F.softmax(logits, dim=-1)

                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_mask = cumsum_probs > top_p
                    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                    sorted_mask[..., 0] = False
                    indices_to_remove = sorted_mask.scatter(
                        1, sorted_indices, sorted_mask
                    )
                    probs = torch.where(indices_to_remove, torch.zeros_like(probs), probs)
                    probs = probs / probs.sum(dim=-1, keepdim=True)

                next_token_id = torch.multinomial(probs, num_samples=1)

            generated_ids.append(next_token_id.item())
            input_ids = next_token_id.unsqueeze(0)

            if next_token_id.item() == self.tokenizer.eos_token_id:
                break

        return input_ids, generated_ids

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_cache_manager: Optional[KVCacheManager] = None,
        seq_id: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Optional[List]]:
        """
        Single forward pass through the model.
        
        Args:
            input_ids: Input token IDs
            kv_cache_manager: KVCacheManager for cache writes
            seq_id: Sequence ID for cache lookup
            
        Returns:
            Tuple of (logits, past_key_values)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                past_key_values=None,
                use_cache=True,
                return_dict=True,
            )

        logits = outputs.logits
        past_key_values = outputs.past_key_values

        if kv_cache_manager is not None and seq_id is not None:
            pass

        return logits, past_key_values

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs"""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")
        return self.tokenizer.encode(text, add_special_tokens=True)

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text"""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)


class DummyModelWrapper:
    """
    Dummy model wrapper for testing without GPU.
    Simulates model forward passes.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self._model_config = {
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "hidden_size": 4096,
            "vocab_size": 151936,
        }

    def load_model(self) -> None:
        """No-op for dummy model"""
        pass

    @property
    def model_config(self) -> Dict[str, Any]:
        return self._model_config

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
    ) -> Tuple[torch.Tensor, List[int]]:
        """Simulated generation"""
        batch_size, seq_len = input_ids.shape
        generated_ids = [0] * min(max_new_tokens, 10)
        return torch.zeros(1, 1, dtype=torch.long), generated_ids

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_cache_manager: Optional[KVCacheManager] = None,
        seq_id: Optional[str] = None,
    ) -> Tuple[torch.Tensor, None]:
        """Simulated forward"""
        batch_size, seq_len = input_ids.shape
        vocab_size = self._model_config["vocab_size"]
        logits = torch.randn(batch_size, seq_len, vocab_size)
        return logits, None

    def encode(self, text: str) -> List[int]:
        """Simulated encoding"""
        return [1, 2, 3, 4, 5]

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Simulated decoding"""
        return "Simulated response"
