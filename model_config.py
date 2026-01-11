"""
Configuration for CODN-350M Model (The 'Micro-Llama').
Strict hardware constraints for RTX 4050 (6GB VRAM).
"""
from dataclasses import dataclass
import torch


@dataclass
class CODNConfig:
    """
    Configuration specs for CODN-350M.
    """
    vocab_size: int = 32000          # Strict limit to save RAM
    d_model: int = 1024              # Narrow width
    n_layers: int = 24               # Deep reasoning
    n_heads: int = 16
    max_seq_len: int = 256           # Optimized for short clinical narratives
    dropout: float = 0.1
    # Hardware specific
    dtype: torch.dtype = torch.bfloat16
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        """Validation of constraints."""
        if self.vocab_size > 32000:
            raise ValueError("Vocab size exceeds 32k limit for 6GB VRAM.")
        if self.d_model > 1024:
            raise ValueError(
                "Hidden size exceeds 1024 limit for 350M param budget.")
