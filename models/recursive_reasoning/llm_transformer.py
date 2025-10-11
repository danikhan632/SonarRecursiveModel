from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel
from models.common import trunc_normal_init_
from models.layers import rms_norm, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear

class LLMTransformerConfig(BaseModel):
    batch_size: int
    seq_len: int
    vocab_size: int
    num_layers: int
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    forward_dtype: str = "bfloat16"

class LLMTransformerBlock(nn.Module):
    def __init__(self, config: LLMTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=True  # Causal attention for language modeling
        )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        # Post Norm
        # Self Attention
        attn_output = self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states)
        hidden_states = rms_norm(hidden_states + attn_output, variance_epsilon=self.norm_eps)
        
        # Fully Connected
        mlp_output = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + mlp_output, variance_epsilon=self.norm_eps)
        
        return hidden_states

class LLMTransformer(nn.Module):
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = LLMTransformerConfig(**config_dict)
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # Input/Output Embeddings
        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale
        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)

        # Positional Embeddings
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=self.config.hidden_size // self.config.num_heads,
                max_position_embeddings=self.config.seq_len,
                base=self.config.rope_theta
            )
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.config.seq_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        else:
            # No explicit positional encoding
            pass

        # Transformer Blocks
        self.layers = nn.ModuleList(
            [LLMTransformerBlock(self.config) for _ in range(self.config.num_layers)]
        )

    def _input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Token embedding
        embedding = self.embed_tokens(input_ids.to(torch.int32))

        # Position embeddings
        if self.config.pos_encodings == "learned":
            pos_embedding = self.embed_pos.embedding_weight.to(self.forward_dtype)
            embedding = embedding + pos_embedding

        # Scale
        return self.embed_scale * embedding

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # 1. Get input embeddings
        hidden_states = self._input_embeddings(input_ids)

        # 2. Get rotary embeddings if using RoPE
        cos_sin = self.rotary_emb() if hasattr(self, "rotary_emb") else None

        # 3. Pass through transformer blocks
        for layer in self.layers:
            hidden_states = layer(cos_sin=cos_sin, hidden_states=hidden_states)

        # 4. Language modeling head
        logits = self.lm_head(hidden_states)

        return logits
