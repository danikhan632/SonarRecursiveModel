from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math
import torch
import copy
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel
from transformers import AutoModel
import random
from models.common import trunc_normal_init_
from models.layers import rms_norm, LinearSwish, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear

IGNORE_LABEL_ID = -100

@dataclass
class RecursiveLLM_InnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor

@dataclass
class RecursiveLLM_Carry:
    inner_carry: RecursiveLLM_InnerCarry
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]
    # for deep supervision: list of intermediate z_H states from all H_cycles
    zH_intermediates: List[torch.Tensor]

class RecursiveLLMConfig(BaseModel):
    batch_size: int
    seq_len: int
    vocab_size: int
    H_cycles: int
    L_cycles: int
    L_layers: int
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    halt_max_steps: int
    halt_exploration_prob: float
    forward_dtype: str = "bfloat16"
    no_ACT_continue: bool = True
    pretrained_model_name: Optional[str] = None
    freeze_embeddings: bool = True
    # Deep supervision: supervise all intermediate H_cycle outputs
    enable_deep_supervision: bool = False
    deep_supervision_weight: float = 0.5  # Weight for intermediate losses

class RecursiveLLMBlock(nn.Module):
    def __init__(self, config: RecursiveLLMConfig) -> None:
        super().__init__()
        self.config = config
        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False  # Set to False to allow full sequence reasoning like original model
        )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        # Post Norm
        # Self Attention
        hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        # Fully Connected
        out = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
        return hidden_states

class RecursiveLLMReasoningModule(nn.Module):
    def __init__(self, layers: List[RecursiveLLMBlock]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states

class RecursiveLLM_Inner(nn.Module):
    def __init__(self, config: RecursiveLLMConfig) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O
        self.embed_scale = math.sqrt(self.config.hidden_size)

        if config.pretrained_model_name:
            print(f"Loading pretrained embeddings from {config.pretrained_model_name}")
            # Use a smaller model for memory efficiency if needed
            pretrained_model = AutoModel.from_pretrained(config.pretrained_model_name)
            self.embed_tokens = pretrained_model.get_input_embeddings()

            # Ensure vocab size is consistent with the tokenizer
            if self.embed_tokens.weight.shape[0] != config.vocab_size:
                self.embed_tokens.resize_token_embeddings(config.vocab_size)
                print(f"Resized token embeddings from {self.embed_tokens.weight.shape[0]} to {config.vocab_size}")

            # Freeze embeddings to prevent catastrophic forgetting
            if config.freeze_embeddings:
                print("Freezing embedding layer weights.")
                self.embed_tokens.requires_grad_(False)
        else:
            print("Training token embeddings from scratch.")
            embed_init_std = 1.0 / self.embed_scale
            self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)

        self.lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True)

        # LM Blocks
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
                                              max_position_embeddings=self.config.seq_len,
                                              base=self.config.rope_theta)
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.config.seq_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        else:
            pass

        # Reasoning Layers
        self.L_level = RecursiveLLMReasoningModule(layers=[RecursiveLLMBlock(self.config) for _i in range(self.config.L_layers)])

        # Initial states
        self.H_init = nn.Parameter(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), requires_grad=True)
        self.L_init = nn.Parameter(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), requires_grad=True)

        # Q head special init
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)

    def _input_embeddings(self, input_ids: torch.Tensor):
        embedding = self.embed_tokens(input_ids.to(torch.int32))
        if self.config.pos_encodings == "learned":
            embedding = embedding + self.embed_pos.embedding_weight.to(self.forward_dtype)
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int):
        # create carry on same device as model parameters to avoid device mismatches
        device = self.H_init.device
        return RecursiveLLM_InnerCarry(
            z_H=torch.empty(batch_size, self.config.seq_len, self.config.hidden_size,
                            dtype=self.forward_dtype, device=device),
            z_L=torch.empty(batch_size, self.config.seq_len, self.config.hidden_size,
                            dtype=self.forward_dtype, device=device),
        )
        
    def reset_carry(self, reset_flag: torch.Tensor, carry: RecursiveLLM_InnerCarry):
        return RecursiveLLM_InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    def forward(self, carry: RecursiveLLM_InnerCarry, batch: Dict[str, torch.Tensor], t: Optional[int] = None, enable_deep_supervision: bool = False) -> Tuple[RecursiveLLM_InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], List[torch.Tensor]]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # allow precomputed continuous embeddings (e.g. SONAR) via key "input_embeddings"
        if "input_embeddings" in batch:
            input_embeddings = batch["input_embeddings"].to(self.forward_dtype)
        else:
            input_embeddings = self._input_embeddings(batch["inputs"])

        z_H, z_L = carry.z_H, carry.z_L

        # Store intermediate z_H states for deep supervision
        z_H_intermediates = []

        if enable_deep_supervision:
            # All H_cycles with grad for deep supervision
            for _H_step in range(self.config.H_cycles):
                for _L_step in range(self.config.L_cycles):
                    z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
                z_H = self.L_level(z_H, z_L, **seq_info)
                z_H_intermediates.append(z_H)
        else:
            # Original behavior: H_cycles-1 without grad, last 1 with grad
            with torch.no_grad():
                for _H_step in range(self.config.H_cycles - 1):
                    for _L_step in range(self.config.L_cycles):
                        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
                    z_H = self.L_level(z_H, z_L, **seq_info)

            # 1 with grad
            for _L_step in range(self.config.L_cycles):
                z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
            z_H = self.L_level(z_H, z_L, **seq_info)
            z_H_intermediates.append(z_H)

        new_carry = RecursiveLLM_InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
        if t is not None:
            output = self.lm_head(z_H[:, t:t+1, :])
        else:
            output = self.lm_head(z_H)
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32) # Q-head uses the first token's state
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1]), z_H_intermediates

class RecursiveLLM(nn.Module):
    """ACT wrapper."""
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = RecursiveLLMConfig(**config_dict)
        self.inner = RecursiveLLM_Inner(self.config)

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        # infer batch size from embeddings or token inputs
        if "input_embeddings" in batch:
            batch_size = batch["input_embeddings"].shape[0]
        else:
            batch_size = batch["inputs"].shape[0]

        # place counters on same device as model parameters (use inner.H_init)
        device = self.inner.H_init.device
        steps = torch.zeros((batch_size,), dtype=torch.int32, device=device)
        halted = torch.ones((batch_size,), dtype=torch.bool, device=device)

        return RecursiveLLM_Carry(
            inner_carry=self.inner.empty_carry(batch_size),
            steps=steps,
            halted=halted,
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
            zH_intermediates=[]
        )
        
    def forward(self, carry: RecursiveLLM_Carry, batch: Dict[str, torch.Tensor], t: Optional[int] = None, enable_deep_supervision: bool = False) -> Tuple[RecursiveLLM_Carry, Dict[str, torch.Tensor]]:
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        new_steps = torch.where(carry.halted, 0, carry.steps)
        new_current_data = {k: torch.where(carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}

        new_inner_carry, logits, (q_halt_logits, q_continue_logits), z_H_intermediates = \
            self.inner(new_inner_carry, new_current_data, t=t, enable_deep_supervision=enable_deep_supervision)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }

        # Add intermediate logits for deep supervision
        if enable_deep_supervision and z_H_intermediates:
            intermediate_logits = []
            for z_H_step in z_H_intermediates:
                if t is not None:
                    step_logits = self.inner.lm_head(z_H_step[:, t:t+1, :])
                else:
                    step_logits = self.inner.lm_head(z_H_step)
                intermediate_logits.append(step_logits)
            outputs["intermediate_logits"] = intermediate_logits

        with torch.no_grad():
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            halted = is_last_step

            if self.training and (self.config.halt_max_steps > 1):
                if self.config.no_ACT_continue:
                    halted = halted | (q_halt_logits > 0)
                else:
                    halted = halted | (q_halt_logits > q_continue_logits)

                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                halted = halted & (new_steps >= min_halt_steps)

                if not self.config.no_ACT_continue:
                    _, _, (next_q_halt_logits, next_q_continue_logits), _ = self.inner(new_inner_carry, new_current_data)
                    outputs["target_q_continue"] = torch.sigmoid(torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits)))

        # record the intermediate z_H states for deep supervision
        new_carry = RecursiveLLM_Carry(
            new_inner_carry, new_steps, halted, new_current_data,
            carry.zH_intermediates + z_H_intermediates
        )
        return new_carry, outputs
