
"""
CODN-350M Model Architecture.
Decoder-only Transformer for Causal Language Modeling.
"""
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CODN350M(nn.Module):
    """
    Clinical Operational Diagnostics Network (350M Parameters).
    Decoder-only Transformer architecture (GPT-style) for Causal Language Modeling.
    """

    def __init__(
        self,
        vocab_size=32000,
        d_model=1024,   # Scaled for ~350M parameters
        n_layers=24,    # Deep network
        n_heads=16,
        max_seq_len=512,
        dropout=0.1
    ):
        """
        Initialize the CODN350M model.

        Args:
            vocab_size (int): Size of the vocabulary.
            d_model (int): Dimension of the model embeddings.
            n_layers (int): Number of transformer layers.
            n_heads (int): Number of attention heads.
            max_seq_len (int): Maximum sequence length.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # 1. Token Embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # 2. Positional Embedding (Learned)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        # 3. Transformer Backbone (Decoder-only)
        # We use TransformerEncoder because in PyTorch, TransformerDecoder enforces
        # cross-attention which we don't want for a pure GPT-style model.
        # By setting is_causal=True in the usage (masks), TransformerEncoder behaves as a Decoder.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers)

        # 4. Layer Norm final
        self.ln_f = nn.LayerNorm(d_model)

        # 5. Language Model Head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying (optional but standard in GPT)
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

        # Metadata for sizing
        self.num_params = sum(p.numel() for p in self.parameters())
        logger.info("Initialized CODN350M with %.1fM parameters",
                    self.num_params / 1e6)

    def _init_weights(self, module):
        """
        Initialize weights for the model.

        Args:
            module (nn.Module): The module to initialize.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx, targets=None):
        """
        Forward pass of the model.

        Args:
            idx (torch.Tensor): Input tensor of token indices (batch_size, seq_len).
            targets (torch.Tensor, optional): Target indices for loss calculation.

        Returns:
            logits (torch.Tensor): Output logits.
            loss (torch.Tensor or None): Calculated loss if targets are provided.
        """
        _, seq_len = idx.size()

        # Check sequence length
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}")

        # Create device-aware position indices
        pos = torch.arange(0, seq_len, dtype=torch.long, device=idx.device)

        # Embeddings
        tok_emb = self.token_embedding(idx)  # (B, T, C)
        pos_emb = self.position_embedding(pos)  # (T, C)
        x = tok_emb + pos_emb

        # Causal Mask (Upper triangular is -inf)
        # nn.TransformerEncoder expects a mask of shape (T, T)
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=idx.device) * float('-inf'),
            diagonal=1
        )

        # Transformer Pass
        # Note: we pass is_causal=True implicitly via the mask structure,
        # but PyTorch 2.0+ optimized attention might prefer the is_causal arg
        # if using flash attention. For compatibility, we use the mask.
        x = self.transformer(x, mask=mask)

        # Final Layer Norm
        x = self.ln_f(x)

        # Logits
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Shift logits and targets for next-token prediction
            # logits: (B, T, V) -> (B*T, V)
            # targets: (B, T) -> (B*T)

            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

        return logits, loss
