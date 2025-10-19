"""
LLaDA-TRM Hybrid Model: Integrating Diffusion and Recursive Reasoning

This module implements a hybrid architecture that combines:
1. LLaDA's diffusion-based parallel generation
2. TRM's recursive refinement for precise corrections

The model operates by:
- Generating text chunks via LLaDA's diffusion process
- Refining each chunk recursively using a lightweight TRM head
- Iterating between diffusion and recursion for optimal output

Author: Claude Code
Date: 2025-10-17
"""

from typing import Tuple, List, Dict, Optional, Union
from dataclasses import dataclass
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer

from models.common import trunc_normal_init_
from models.layers import rms_norm, SwiGLU, CastedLinear
from models.recursive_reasoning.slot_trm_refiner import SlotTRMRefiner, create_slot_trm_refiner

# Debug mode controlled by environment variable
DEBUG = os.environ.get('DEBUG', 'False').lower() in ('true', '1', 'yes')

# Color codes for terminal output
class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def print_blue(text, prefix=""):
    """Print text in blue color"""
    if prefix:
        print(f"{prefix}{Colors.BLUE}{text}{Colors.END}")
    else:
        print(f"{Colors.BLUE}{text}{Colors.END}")

def print_green(text, prefix=""):
    """Print text in green color"""
    if prefix:
        print(f"{prefix}{Colors.GREEN}{text}{Colors.END}")
    else:
        print(f"{Colors.GREEN}{text}{Colors.END}")

def print_yellow(text, prefix=""):
    """Print text in yellow color"""
    if prefix:
        print(f"{prefix}{Colors.YELLOW}{text}{Colors.END}")
    else:
        print(f"{Colors.YELLOW}{text}{Colors.END}")


@dataclass
class RecursiveRefinementState:
    """State for recursive chunk refinement"""
    chunk_embeddings: torch.Tensor  # [B, num_chunks, chunk_size, hidden_dim]
    chunk_scores: torch.Tensor  # [B, num_chunks]
    refinement_steps: torch.Tensor  # [B, num_chunks]
    converged: torch.Tensor  # [B, num_chunks]


class LLaDATRMConfig(BaseModel):
    """Configuration for LLaDA-TRM Hybrid Model"""
    # LLaDA backbone config
    llada_model_name: str = "inclusionAI/LLaDA-MoE-7B-A1B-Instruct"
    freeze_llada_backbone: bool = True

    # Chunk-based processing
    chunk_size: int = 16  # tokens per chunk
    max_chunks: int = 8  # maximum chunks to process

    # Recursive head config
    hidden_size: int = 2048  # must match LLaDA hidden size
    head_hidden_size: int = 512  # internal head dimension
    head_layers: int = 2  # number of refinement layers

    # Refiner type selection
    refiner_type: str = "basic"  # "basic" or "attn_refiner" (attention-based with slots)
    refiner_size: str = "base"  # "tiny", "base", or "large" (only for attn_refiner)

    # Adaptive Computation Time (ACT) for learned halting
    use_act: bool = False  # Enable ACT for adaptive halting
    act_exploration_prob: float = 0.1  # Exploration probability during training
    act_q_loss_weight: float = 0.1  # Weight for Q-learning loss

    # Recursion control
    max_recursive_steps: int = 8
    convergence_threshold: float = 0.01  # cosine similarity threshold
    min_confidence: float = 0.5  # minimum token confidence to skip refinement

    # Training
    enable_deep_supervision: bool = True
    deep_supervision_weight: float = 0.3
    sft_mode: bool = False  # Teacher forcing SFT (skip diffusion, use ground truth embeddings)
    mask_probability: float = 0.3  # Probability of masking each token during SFT training
    mask_token_id: int = 156895  # LLaDA mask token ID

    # Efficiency
    refine_low_confidence_only: bool = True  # only refine chunks below threshold
    forward_dtype: str = "bfloat16"


class RecursiveRefinementHead(nn.Module):
    """
    Lightweight recursive head for chunk refinement.

    Inspired by TRM's delta-based updates, this head:
    1. Takes a chunk embedding [chunk_size, hidden_dim]
    2. Computes a delta (proposed change)
    3. Applies the delta and scores the result
    4. Iterates until convergence
    """

    def __init__(self, config: LLaDATRMConfig):
        super().__init__()
        self.config = config
        # Handle dtype: use float32 for CPU, bfloat16 for CUDA if available
        if config.forward_dtype == "bfloat16" and not torch.cuda.is_available():
            print("Warning: BFloat16 not well supported on CPU, using Float32")
            self.forward_dtype = torch.float32
        else:
            self.forward_dtype = getattr(torch, config.forward_dtype, torch.float32)

        # Chunk encoder: compress chunk to single vector
        self.chunk_encoder = nn.Sequential(
            nn.Linear(config.hidden_size * config.chunk_size, config.head_hidden_size),
            nn.LayerNorm(config.head_hidden_size),
            nn.GELU(),
        )

        # Delta generator: proposes changes
        layers = []
        for i in range(config.head_layers):
            layers.append(
                SwiGLU(
                    hidden_size=config.head_hidden_size,
                    expansion=2.0,
                )
            )
        self.delta_generator = nn.Sequential(*layers)

        # Delta projector: back to chunk space
        self.delta_projector = nn.Linear(config.head_hidden_size, config.hidden_size * config.chunk_size)

        # Confidence scorer: estimates quality of refinement
        self.confidence_scorer = nn.Sequential(
            nn.Linear(config.head_hidden_size, config.head_hidden_size // 2),
            nn.GELU(),
            nn.Linear(config.head_hidden_size // 2, 1),
            nn.Sigmoid(),
        )

        # Initialize delta projection to small values for stable training
        nn.init.normal_(self.delta_projector.weight, std=0.01)
        nn.init.zeros_(self.delta_projector.bias)

        # Convert all modules to the correct dtype
        self._convert_to_dtype()

    def _convert_to_dtype(self):
        """Convert all modules to the correct dtype to match backbone"""
        # Convert all Linear and LayerNorm layers to the target dtype
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.LayerNorm)):
                module.to(self.forward_dtype)

    def forward(
        self,
        chunk_emb: torch.Tensor,  # [B, chunk_size, hidden_dim]
        max_steps: int = 8,
        convergence_threshold: float = 0.01,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Recursively refine a chunk embedding.

        Returns:
            refined_chunk: [B, chunk_size, hidden_dim]
            confidence: [B] scalar confidence score
            num_steps: int, actual steps taken
        """
        B, L, D = chunk_emb.shape
        assert L == self.config.chunk_size
        assert D == self.config.hidden_size

        # Removed verbose debug output - see chunk-level debugging in selective_refinement instead

        # Ensure input dtype matches head dtype
        original_dtype = chunk_emb.dtype
        current_emb = chunk_emb.to(self.forward_dtype)
        prev_compressed = None

        for step in range(max_steps):
            # Compress chunk to fixed-size representation
            flat_emb = current_emb.reshape(B, -1)  # [B, chunk_size * hidden_dim]
            compressed = self.chunk_encoder(flat_emb)  # [B, head_hidden_size]

            # Check convergence via cosine similarity
            if prev_compressed is not None:
                similarity = F.cosine_similarity(compressed, prev_compressed, dim=-1)  # [B]
                if (similarity > 1.0 - convergence_threshold).all():
                    break

            prev_compressed = compressed

            # Generate delta
            delta_hidden = self.delta_generator(compressed)  # [B, head_hidden_size]
            delta_flat = self.delta_projector(delta_hidden)  # [B, chunk_size * hidden_dim]
            delta = delta_flat.reshape(B, L, D)  # [B, chunk_size, hidden_dim]

            # Apply delta with residual connection
            current_emb = current_emb + 0.1 * delta  # small learning rate for stability

        # Final confidence score
        final_compressed = self.chunk_encoder(current_emb.reshape(B, -1))
        confidence = self.confidence_scorer(final_compressed).squeeze(-1)  # [B]

        # Convert back to original dtype to match backbone
        current_emb = current_emb.to(original_dtype)

        return current_emb, confidence, step + 1


class SlotTRMRefinerAdapter(nn.Module):
    """
    Adapter to make SlotTRMRefiner compatible with RecursiveRefinementHead interface.

    This allows using SlotTRMRefiner (which has attention for cross-chunk reasoning)
    in place of the basic RecursiveRefinementHead.
    """

    def __init__(self, slot_refiner: SlotTRMRefiner, config: 'LLaDATRMConfig'):
        super().__init__()
        self.slot_refiner = slot_refiner
        self.config = config

    def forward(
        self,
        chunk_emb: torch.Tensor,  # [B, chunk_size, hidden_dim]
        max_steps: int = 8,
        convergence_threshold: float = 0.01,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Forward pass compatible with RecursiveRefinementHead interface.

        Returns:
            refined_chunk: [B, chunk_size, hidden_dim]
            confidence: [B] scalar confidence score
            num_steps: int, actual steps taken (always K for SlotTRMRefiner)
        """
        # SlotTRMRefiner always runs K steps (no early convergence)
        refined, stats = self.slot_refiner(chunk_emb, return_stats=True)

        # Extract per-batch confidence (mean across sequence)
        # stats['final_confidence'] is [B, L]
        confidence = stats['final_confidence'].mean(dim=-1)  # [B]

        # Number of steps is always K
        num_steps = stats['steps']

        return refined, confidence, num_steps


class LLaDATRMHybrid(nn.Module):
    """
    Hybrid LLaDA-TRM Model

    Architecture:
    1. LLaDA backbone (frozen or fine-tuned) for diffusion-based generation
    2. Recursive refinement head for chunk-level corrections
    3. Integrated forward pass with alternating diffusion and recursion
    """

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = LLaDATRMConfig(**config_dict)

        # Determine appropriate dtype for the device
        # Use float32 for CPU, bfloat16/float16 for CUDA
        if self.config.forward_dtype == "bfloat16" and not torch.cuda.is_available():
            print("Warning: BFloat16 not well supported on CPU, using Float32 for backbone")
            backbone_dtype = torch.float32
            self.config.forward_dtype = "float32"  # Update config for consistency
        else:
            backbone_dtype = getattr(torch, self.config.forward_dtype, torch.float32)

        # Load LLaDA backbone
        print(f"Loading LLaDA backbone: {self.config.llada_model_name}")
        print(f"Using dtype: {backbone_dtype}")
        self.llada_backbone = AutoModel.from_pretrained(
            self.config.llada_model_name,
            trust_remote_code=True,
            torch_dtype=backbone_dtype
        )

        # Freeze backbone if specified
        if self.config.freeze_llada_backbone:
            print("Freezing LLaDA backbone parameters")
            for param in self.llada_backbone.parameters():
                param.requires_grad = False

        # Get hidden size from backbone
        if hasattr(self.llada_backbone.config, 'hidden_size'):
            actual_hidden_size = self.llada_backbone.config.hidden_size
        elif hasattr(self.llada_backbone.config, 'd_model'):
            actual_hidden_size = self.llada_backbone.config.d_model
        else:
            actual_hidden_size = self.config.hidden_size

        self.config.hidden_size = actual_hidden_size
        print(f"Detected hidden size: {actual_hidden_size}")

        # Recursive refinement head - select based on refiner_type
        refiner_type = self.config.refiner_type.lower()
        if refiner_type in ["slot_based", "attn_refiner"]:
            print(f"Creating Attention-based Refiner with Slot Decomposition (size: {self.config.refiner_size})")
            slot_refiner = create_slot_trm_refiner(
                d_model=actual_hidden_size,
                size=self.config.refiner_size,
                K=self.config.max_recursive_steps,
                use_act=self.config.use_act,
                act_exploration_prob=self.config.act_exploration_prob,
            )
            # Convert to correct dtype to match backbone
            slot_refiner.to_dtype(backbone_dtype)
            self.refinement_head = SlotTRMRefinerAdapter(slot_refiner, self.config)
            print(f"  - {len(slot_refiner.blocks)} transformer blocks with attention for cross-chunk reasoning")
            print(f"  - Slot projections for semantic decomposition (context, reason, refine, confidence)")
            if self.config.use_act:
                print(f"  - ACT enabled: Adaptive halting with Q-learning (exploration_prob={self.config.act_exploration_prob})")
        else:
            print("Creating Basic Refinement Head (FFN-only, no attention)")
            self.refinement_head = RecursiveRefinementHead(self.config)

        # LM head for final predictions
        # Try to reuse LLaDA's LM head if available
        if hasattr(self.llada_backbone, 'lm_head'):
            self.lm_head = self.llada_backbone.lm_head
        else:
            vocab_size = self.llada_backbone.config.vocab_size
            self.lm_head = CastedLinear(self.config.hidden_size, vocab_size, bias=False)

        print(f"Hybrid model initialized: {self.count_parameters() / 1e6:.2f}M total params")
        print(f"Refinement head: {self.count_head_parameters() / 1e6:.2f}M params")

        # Store tokenizer reference for debug mode (loaded lazily)
        self._tokenizer = None

    def set_tokenizer(self, tokenizer):
        """Set tokenizer for debug text decoding"""
        self._tokenizer = tokenizer

    def _apply_dynamic_masking(
        self,
        input_ids: torch.Tensor,  # [B, seq_len]
    ) -> torch.Tensor:
        """
        Apply chunk-level dynamic masking during SFT training.

        IMPORTANT: Only masks reasoning/answer chunks AFTER the question.
        Question tokens are NEVER masked.

        Masks entire chunks (thoughts) in the reasoning portion.
        Each refinement step should unmask one chunk/thought.

        Returns:
            masked_input_ids: [B, seq_len] with entire chunks masked
        """
        if not self.training or self.config.mask_probability <= 0:
            return input_ids

        B, L = input_ids.shape
        masked_input_ids = input_ids.clone()

        # Calculate number of chunks
        num_chunks = (L + self.config.chunk_size - 1) // self.config.chunk_size

        for b in range(B):
            # Find where reasoning starts by looking for common patterns
            # Try to find "Let's solve this step by step:" or similar markers
            reasoning_start_pos = None

            # Try to decode and find the pattern
            if self._tokenizer is not None:
                try:
                    text = self._tokenizer.decode(input_ids[b], skip_special_tokens=False)
                    # Look for common reasoning markers (in order of priority)
                    markers = [
                        "Let's solve this step by step:",
                        "Step by step:",
                        "Solution:",
                        "<answer>",  # Can also mask answer portion
                        "Answer:"
                    ]
                    for marker in markers:
                        if marker in text:
                            # Find approximate token position
                            marker_text_pos = text.index(marker)
                            # Estimate token position (roughly 4 chars per token)
                            reasoning_start_pos = min(marker_text_pos // 4, L)
                            break
                except:
                    pass

            # Fallback: if we can't find marker, assume question is first 50% of sequence
            if reasoning_start_pos is None:
                reasoning_start_pos = L // 2

            # Find which chunk contains the reasoning start
            reasoning_start_chunk = reasoning_start_pos // self.config.chunk_size

            # Only mask chunks AFTER the question (starting from reasoning_start_chunk)
            for chunk_idx in range(reasoning_start_chunk, num_chunks):
                # Randomly decide whether to mask this entire chunk
                if torch.rand(1).item() < self.config.mask_probability:
                    start_pos = chunk_idx * self.config.chunk_size
                    end_pos = min((chunk_idx + 1) * self.config.chunk_size, L)

                    # Get tokens in this chunk
                    chunk_tokens = input_ids[b, start_pos:end_pos]

                    # Don't mask if chunk is all special tokens (padding)
                    is_special = chunk_tokens >= 150000
                    if not is_special.all():
                        # Mask the entire chunk
                        masked_input_ids[b, start_pos:end_pos] = self.config.mask_token_id

        return masked_input_ids

    def _decode_ids(self, input_ids, max_length=100, skip_special_tokens=True):
        """Decode input_ids to text for debug output"""
        if self._tokenizer is None:
            return "[tokenizer not set]"
        try:
            text = self._tokenizer.decode(input_ids[0], skip_special_tokens=skip_special_tokens)
            # if len(text) > max_length:
            #     return text[:max_length] + "..."
            return text
        except Exception as e:
            return f"[decode error: {e}]"

    def count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_head_parameters(self) -> int:
        """Count refinement head parameters"""
        return sum(p.numel() for p in self.refinement_head.parameters())

    def chunk_sequence(
        self,
        hidden_states: torch.Tensor,  # [B, seq_len, hidden_dim]
        chunk_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        Split sequence into chunks for refinement.

        Returns: [B, num_chunks, chunk_size, hidden_dim]
        """
        chunk_size = chunk_size or self.config.chunk_size
        B, L, D = hidden_states.shape

        # Pad to multiple of chunk_size
        pad_len = (chunk_size - L % chunk_size) % chunk_size
        if pad_len > 0:
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad_len))

        # Reshape to chunks
        L_padded = hidden_states.shape[1]
        num_chunks = L_padded // chunk_size
        chunks = hidden_states.reshape(B, num_chunks, chunk_size, D)

        return chunks

    def unchunk_sequence(
        self,
        chunks: torch.Tensor,  # [B, num_chunks, chunk_size, hidden_dim]
        original_length: int
    ) -> torch.Tensor:
        """Merge chunks back to sequence"""
        B, num_chunks, chunk_size, D = chunks.shape
        sequence = chunks.reshape(B, num_chunks * chunk_size, D)
        return sequence[:, :original_length, :]

    def diffusion_step(
        self,
        input_ids: torch.Tensor,
        mask_index: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Single diffusion step using LLaDA backbone.

        Returns: hidden_states [B, seq_len, hidden_dim]
        """
        outputs = self.llada_backbone(
            input_ids=input_ids,
            output_hidden_states=True,
            **kwargs
        )

        # Get last hidden state
        if hasattr(outputs, 'hidden_states'):
            hidden_states = outputs.hidden_states[-1]
        elif hasattr(outputs, 'last_hidden_state'):
            hidden_states = outputs.last_hidden_state
        else:
            raise ValueError("Cannot extract hidden states from LLaDA backbone")

        return hidden_states

    def selective_refinement(
        self,
        chunks: torch.Tensor,  # [B, num_chunks, chunk_size, hidden_dim]
        logits: torch.Tensor,  # [B, seq_len, vocab_size] for confidence estimation
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Selectively refine chunks based on confidence.

        Returns:
            refined_chunks: [B, num_chunks, chunk_size, hidden_dim]
            chunk_confidences: [B, num_chunks]
            refinement_steps: [B, num_chunks]
        """
        B, num_chunks, chunk_size, D = chunks.shape

        # Compute token-level confidence from logits
        probs = F.softmax(logits, dim=-1)
        max_probs = probs.max(dim=-1).values  # [B, seq_len]

        # Reshape to chunks
        max_probs_padded = F.pad(max_probs, (0, num_chunks * chunk_size - max_probs.shape[1]))
        chunk_probs = max_probs_padded.reshape(B, num_chunks, chunk_size)
        chunk_confidence = chunk_probs.mean(dim=-1)  # [B, num_chunks]

        refined_chunks = torch.zeros_like(chunks)
        refinement_steps = torch.zeros(B, num_chunks, dtype=torch.long, device=chunks.device)
        final_confidences = torch.zeros(B, num_chunks, device=chunks.device)

        # Get chunk-level predictions for debugging
        chunk_logits = logits[:, :num_chunks * chunk_size, :].reshape(B, num_chunks, chunk_size, -1)

        # Refine each chunk
        for i in range(num_chunks):
            chunk_i = chunks[:, i, :, :]  # [B, chunk_size, hidden_dim]

            # Skip chunk-by-chunk debug output during training (will show in deep supervision instead)

            if self.config.refine_low_confidence_only:
                # Only refine if confidence is low
                needs_refinement = chunk_confidence[:, i] < self.config.min_confidence

                if needs_refinement.any():
                    # Refine low-confidence samples
                    refined, conf, steps = self.refinement_head(
                        chunk_i,
                        max_steps=self.config.max_recursive_steps,
                        convergence_threshold=self.config.convergence_threshold
                    )

                    # Mix refined and original based on needs_refinement mask
                    needs_refinement_expanded = needs_refinement.view(B, 1, 1).expand_as(chunk_i)
                    refined_chunks[:, i, :, :] = torch.where(
                        needs_refinement_expanded,
                        refined,
                        chunk_i
                    )
                    final_confidences[:, i] = torch.where(needs_refinement, conf, chunk_confidence[:, i])
                    refinement_steps[:, i] = steps * needs_refinement.long()
                else:
                    refined_chunks[:, i, :, :] = chunk_i
                    final_confidences[:, i] = chunk_confidence[:, i]
            else:
                # Refine all chunks
                refined, conf, steps = self.refinement_head(
                    chunk_i,
                    max_steps=self.config.max_recursive_steps,
                    convergence_threshold=self.config.convergence_threshold
                )
                refined_chunks[:, i, :, :] = refined
                final_confidences[:, i] = conf
                refinement_steps[:, i] = steps

        return refined_chunks, final_confidences, refinement_steps

    def _compute_deep_supervision_loss(
        self,
        chunks: torch.Tensor,  # [B, num_chunks, chunk_size, hidden_dim]
        labels: torch.Tensor,  # [B, seq_len]
        refinement_steps: torch.Tensor,  # [B, num_chunks]
    ) -> torch.Tensor:
        """
        Compute deep supervision loss by supervising each chunk separately.

        For each chunk i:
        - Compute logits for chunks 0 to i (masking future chunks)
        - Compare against ground truth labels for chunk i
        - This teaches the model to generate thoughts step-by-step

        Returns:
            Combined loss across all chunks
        """
        B, num_chunks, chunk_size, D = chunks.shape

        # Prepare chunk-level labels
        seq_len = labels.shape[1]
        padded_labels = F.pad(labels, (0, num_chunks * chunk_size - seq_len), value=-100)
        chunk_labels = padded_labels.reshape(B, num_chunks, chunk_size)

        total_loss = 0.0
        num_supervised_chunks = 0
        loss_fct = nn.CrossEntropyLoss()

        if DEBUG:
            print(f"\n[Deep Supervision] Computing chunk-wise losses...")
            print(f"  Teaching model: for chunk i, only see chunks 0 to i-1 (causal masking)")
            print(f"  Each refinement step should reveal one thought/chunk progressively")
            if self._tokenizer is not None:
                print(f"\n  📚 Ground Truth Chunks (what model should learn to unmask):")
                # Show first few chunks of ground truth
                seq_len = labels.shape[1]
                padded_labels = F.pad(labels, (0, num_chunks * chunk_size - seq_len), value=-100)
                chunk_labels_preview = padded_labels.reshape(B, num_chunks, chunk_size)
                for preview_i in range(min(5, num_chunks)):  # Show first 5 chunks
                    if (chunk_labels_preview[:, preview_i, :] != -100).any():
                        try:
                            chunk_text = self._tokenizer.decode(chunk_labels_preview[0, preview_i, :], skip_special_tokens=False)
                            print_green(f"    Chunk {preview_i+1}: \"{chunk_text}\"", prefix="  ")
                        except:
                            pass
                if num_chunks > 5:
                    print(f"    ... ({num_chunks-5} more chunks)")

        # For each chunk, compute loss only on that chunk's predictions
        for i in range(num_chunks):
            # Only supervise chunks that were refined or are within the valid sequence
            chunk_has_labels = (chunk_labels[:, i, :] != -100).any()

            if not chunk_has_labels:
                continue

            # Get context embeddings: chunks 0 to i-1 (causal masking - don't see current chunk i)
            # For chunk 0, we have no context, so we include it to get at least one prediction
            if i == 0:
                # First chunk: use its own embeddings (no previous context)
                current_chunks = chunks[:, :1, :, :]  # [B, 1, chunk_size, hidden_dim]
            else:
                # Later chunks: use only previous chunks as context, plus current chunk for within-chunk predictions
                current_chunks = chunks[:, :i+1, :, :]  # [B, i+1, chunk_size, hidden_dim]

            # Flatten to sequence
            current_seq = current_chunks.reshape(B, -1, D)
            seq_len = current_seq.shape[1]

            # Compute logits for this partial sequence
            chunk_logits = self.lm_head(current_seq)  # [B, seq_len, vocab_size]

            # Get labels for chunk i
            chunk_i_labels = chunk_labels[:, i, :]  # [B, chunk_size]

            # For causal LM: position j predicts position j+1
            # We want to predict all tokens in chunk i
            if i == 0:
                # First chunk: standard causal shift within the chunk
                chunk_i_logits = chunk_logits[:, :chunk_size, :]
                shift_logits = chunk_i_logits[:, :-1, :].contiguous()
                shift_labels = chunk_i_labels[:, 1:].contiguous()
            else:
                # Later chunks: extract logits that predict chunk i
                # Logits from positions (i*chunk_size-1) to ((i+1)*chunk_size-2) predict positions i*chunk_size to (i+1)*chunk_size-1
                # But since current_seq has length (i+1)*chunk_size, positions are:
                # Last position of chunk i-1: i*chunk_size - 1
                # Positions in chunk i: i*chunk_size to (i+1)*chunk_size - 1
                # Extract logits from (i*chunk_size-1) to ((i+1)*chunk_size-2) inclusive
                start_logit = i * chunk_size - 1
                end_logit = (i + 1) * chunk_size - 1
                chunk_i_logits = chunk_logits[:, start_logit:end_logit, :]  # [B, chunk_size, vocab_size]
                # These logits predict chunk i directly (no further shift needed)
                shift_logits = chunk_i_logits.contiguous()
                shift_labels = chunk_i_labels.contiguous()

            chunk_loss = loss_fct(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1)
            )

            # Weight by refinement steps (chunks that needed more refinement get higher weight)
            weight = 1.0
            if self.config.deep_supervision_weight > 0:
                # Higher weight for chunks that were actually refined
                was_refined = (refinement_steps[:, i] > 0).float().mean()
                weight = 1.0 + self.config.deep_supervision_weight * was_refined

            total_loss += weight * chunk_loss
            num_supervised_chunks += 1

            if DEBUG and i < 5:  # Print first 5 chunks to show what model is learning
                try:
                    # Decode this chunk's prediction
                    chunk_pred_ids = chunk_i_logits.argmax(dim=-1)
                    chunk_text = self._tokenizer.decode(chunk_pred_ids[0], skip_special_tokens=False) if self._tokenizer else ""
                    label_text = self._tokenizer.decode(chunk_i_labels[0], skip_special_tokens=False) if self._tokenizer else ""

                    print(f"\n  Chunk {i+1}/{num_chunks}: loss={chunk_loss.item():.4f}, weight={weight:.2f}")
                    print_yellow(f"    Model predicts: \"{chunk_text}\"", prefix="  ")
                    print_green(f"    Should predict: \"{label_text}\"", prefix="  ")
                except:
                    pass

        # Average loss across supervised chunks
        if num_supervised_chunks > 0:
            total_loss = total_loss / num_supervised_chunks

        if DEBUG:
            print(f"  Supervised {num_supervised_chunks}/{num_chunks} chunks")
            print(f"  Average deep supervision loss: {total_loss.item():.4f}")

        return total_loss

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        enable_refinement: bool = True,
        return_dict: bool = True,
        **kwargs
    ) -> Union[Tuple, Dict]:
        """
        Forward pass with integrated diffusion and recursion.

        Process:
        1. Run diffusion step (LLaDA)
        2. Chunk hidden states
        3. Selectively refine chunks (TRM)
        4. Compute logits
        5. Return loss and outputs
        """
        B, L = input_ids.shape

        if DEBUG:
            print(f"\n{'='*70}")
            print(f"[DEBUG] LLaDATRMHybrid.forward()")
            print(f"{'='*70}")
            print(f"  Input IDs shape: {input_ids.shape}")
            print(f"  Input IDs dtype: {input_ids.dtype}")
            if attention_mask is not None:
                print(f"  Attention mask shape: {attention_mask.shape}")
            if labels is not None:
                print(f"  Labels shape: {labels.shape}")
            print(f"  Enable refinement: {enable_refinement}")

            # Decode and print actual input text in blue
            decoded_text = self._decode_ids(input_ids, max_length=150, skip_special_tokens=False)
            print(f"\n  📝 Input Text (with special tokens):")
            print_blue(f"  {decoded_text}", prefix="  ")

        # Step 1: Get hidden states (diffusion OR teacher forcing)
        if self.config.sft_mode:
            # SFT Mode: Teacher forcing with dynamic masking
            if DEBUG:
                print(f"\n[Step 1] Teacher forcing mode with dynamic masking...")

            # Apply dynamic masking to input_ids (reasoning tokens get masked)
            masked_input_ids = self._apply_dynamic_masking(input_ids)

            if DEBUG:
                # Show masked input
                num_masked = (masked_input_ids == self.config.mask_token_id).sum().item()
                total_tokens = masked_input_ids.numel()
                mask_pct = 100 * num_masked / total_tokens if total_tokens > 0 else 0
                print(f"  Masked {num_masked}/{total_tokens} tokens ({mask_pct:.1f}%)")

                # Show which chunks are masked
                num_chunks_temp = (L + self.config.chunk_size - 1) // self.config.chunk_size
                masked_chunks = []
                for chunk_idx in range(num_chunks_temp):
                    start_pos = chunk_idx * self.config.chunk_size
                    end_pos = min((chunk_idx + 1) * self.config.chunk_size, L)
                    chunk_tokens = masked_input_ids[0, start_pos:end_pos]
                    if (chunk_tokens == self.config.mask_token_id).any():
                        masked_chunks.append(chunk_idx + 1)

                if masked_chunks:
                    print(f"  Masked chunks (thoughts): {masked_chunks}")
                else:
                    print(f"  No chunks fully masked (token-level masking)")

                # Decode masked input
                masked_text = self._decode_ids(masked_input_ids, max_length=200, skip_special_tokens=False)
                print(f"\n  🎭 Masked Input (chunks to refine/unmask):")
                print_yellow(f"  {masked_text}", prefix="  ")

            # Get embeddings from masked input_ids
            embedding_layer = self.llada_backbone.get_input_embeddings()
            hidden_states = embedding_layer(masked_input_ids)

            if DEBUG:
                print(f"\n  Ground truth embeddings shape: {hidden_states.shape}")
                print(f"  Ground truth embeddings dtype: {hidden_states.dtype}")
                print(f"  Embeddings stats: min={hidden_states.min().item():.4f}, "
                      f"max={hidden_states.max().item():.4f}, "
                      f"mean={hidden_states.mean().item():.4f}")
        else:
            # Standard Mode: Diffusion step via LLaDA backbone
            if DEBUG:
                print(f"\n[Step 1] Running diffusion step...")

            hidden_states = self.diffusion_step(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs
            )

            if DEBUG:
                print(f"  Hidden states shape: {hidden_states.shape}")
                print(f"  Hidden states dtype: {hidden_states.dtype}")
                print(f"  Hidden states stats: min={hidden_states.min().item():.4f}, "
                      f"max={hidden_states.max().item():.4f}, "
                      f"mean={hidden_states.mean().item():.4f}")

        # Step 2: Get initial logits for confidence estimation
        if DEBUG:
            print(f"\n[Step 2] Computing initial logits...")

        initial_logits = self.lm_head(hidden_states)

        if DEBUG:
            print(f"  Initial logits shape: {initial_logits.shape}")
            print(f"  Initial logits dtype: {initial_logits.dtype}")

        if enable_refinement:
            if DEBUG:
                print(f"\n[Step 3] Chunking hidden states...")

            # Step 3: Chunk the hidden states
            chunks = self.chunk_sequence(hidden_states)  # [B, num_chunks, chunk_size, hidden_dim]

            if DEBUG:
                print(f"  Chunks shape: {chunks.shape}")
                print(f"  Num chunks: {chunks.shape[1]}")

            # Step 4: Selective refinement
            if DEBUG:
                print(f"\n[Step 4] Selective refinement (refining embeddings)...")

            refined_chunks, chunk_confidences, refinement_steps = self.selective_refinement(
                chunks, initial_logits
            )

            if DEBUG:
                print(f"\n[Step 4 Summary]")
                print(f"  Refined chunks shape: {refined_chunks.shape}")
                num_refined = (refinement_steps > 0).sum().item()
                total_chunks = refined_chunks.shape[1]
                print(f"  Chunks refined: {num_refined}/{total_chunks}")
                print(f"  Avg chunk confidence: {chunk_confidences.mean().item():.4f}")
                print(f"  Avg refinement steps: {refinement_steps.float().mean().item():.2f}")

            # Step 5: Unchunk back to sequence
            if DEBUG:
                print(f"\n[Step 5] Unchunking...")

            refined_hidden = self.unchunk_sequence(refined_chunks, L)

            if DEBUG:
                print(f"  Refined hidden shape: {refined_hidden.shape}")

            # Step 6: Final logits
            if DEBUG:
                print(f"\n[Step 6] Computing final logits...")

            logits = self.lm_head(refined_hidden)

            if DEBUG:
                print(f"  Final logits shape: {logits.shape}")

            # Track refinement stats
            avg_steps = refinement_steps.float().mean().item()
            avg_confidence = chunk_confidences.mean().item()
        else:
            if DEBUG:
                print(f"\n[Refinement disabled] Using initial logits")
            logits = initial_logits
            avg_steps = 0
            avg_confidence = 1.0

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            if DEBUG:
                print(f"\n[Computing loss]...")

            # Check if deep supervision is enabled
            if self.config.enable_deep_supervision and enable_refinement:
                # Deep supervision: supervise each chunk separately
                loss = self._compute_deep_supervision_loss(
                    chunks=refined_chunks,
                    labels=labels,
                    refinement_steps=refinement_steps
                )
                if DEBUG:
                    print(f"  Deep supervision loss: {loss.item():.4f}")
            else:
                # Standard causal LM loss
                # Shift for causal LM
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )

                if DEBUG:
                    print(f"  Standard loss: {loss.item():.4f}")

        if DEBUG:
            print(f"\n[Final Summary]")
            print(f"  Loss: {loss.item() if loss is not None else 'N/A'}")
            print(f"  Logits shape: {logits.shape}")
            print(f"  Refinement steps: {avg_steps:.2f}")
            print(f"  Chunk confidence: {avg_confidence:.4f}")

            # Only show final prediction if refinement was disabled
            # (otherwise we already showed it in step 6)
            if not enable_refinement and self._tokenizer is not None:
                try:
                    # Get predicted tokens from logits
                    predicted_ids = logits.argmax(dim=-1)
                    predicted_text = self._tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
                    print(f"\n  🎯 Final Prediction:")
                    print_green(f"  {predicted_text}", prefix="  ")
                except Exception as e:
                    print(f"  [Could not decode prediction: {e}]")

            print(f"{'='*70}\n")

        if return_dict:
            return {
                "loss": loss,
                "logits": logits,
                "hidden_states": refined_hidden if enable_refinement else hidden_states,
                "refinement_steps": avg_steps,
                "chunk_confidence": avg_confidence,
            }
        else:
            return (loss, logits) if loss is not None else logits

    def generate_with_refinement(
        self,
        prompt: torch.Tensor,
        max_length: int = 128,
        num_diffusion_steps: int = 8,
        temperature: float = 1.0,
        top_p: float = 0.9,
        mask_token_id: int = 156895,
    ) -> torch.Tensor:
        """
        Generate text with integrated diffusion and recursive refinement.

        This implements the hybrid inference pipeline:
        1. Initialize with masked tokens
        2. Run diffusion steps with refinement
        3. Progressively unmask tokens
        4. Return final sequence
        """
        self.eval()
        B = prompt.shape[0]
        device = prompt.device

        # Initialize sequence with prompt + masked tokens
        gen_length = max_length - prompt.shape[1]
        x = torch.full((B, max_length), mask_token_id, dtype=torch.long, device=device)
        x[:, :prompt.shape[1]] = prompt

        prompt_mask = torch.zeros(B, max_length, dtype=torch.bool, device=device)
        prompt_mask[:, :prompt.shape[1]] = True

        with torch.no_grad():
            for step in range(num_diffusion_steps):
                # Forward pass with refinement
                outputs = self.forward(
                    input_ids=x,
                    enable_refinement=True,
                    return_dict=True
                )
                logits = outputs["logits"]

                # Sample from logits
                if temperature > 0:
                    probs = F.softmax(logits / temperature, dim=-1)
                    next_tokens = torch.multinomial(probs.view(-1, probs.size(-1)), 1)
                    next_tokens = next_tokens.view(B, max_length)
                else:
                    next_tokens = logits.argmax(dim=-1)

                # Compute confidence for progressive unmasking
                max_probs = F.softmax(logits, dim=-1).max(dim=-1).values

                # Determine which positions to unmask (highest confidence)
                mask_positions = (x == mask_token_id) & ~prompt_mask
                num_to_unmask = max(1, mask_positions.sum().item() // (num_diffusion_steps - step))

                # Select top-k confident positions
                masked_confidences = torch.where(
                    mask_positions,
                    max_probs,
                    torch.tensor(-float('inf'), device=device)
                )
                _, top_indices = masked_confidences.topk(
                    min(num_to_unmask, mask_positions.sum().item()),
                    dim=-1
                )

                # Unmask selected positions
                for b in range(B):
                    x[b, top_indices[b]] = next_tokens[b, top_indices[b]]

        return x


def create_llada_trm_hybrid(
    llada_model_name: str = "inclusionAI/LLaDA-MoE-7B-A1B-Instruct",
    chunk_size: int = 16,
    max_recursive_steps: int = 8,
    freeze_backbone: bool = True,
    **kwargs
) -> LLaDATRMHybrid:
    """
    Factory function to create a LLaDA-TRM hybrid model.

    Args:
        llada_model_name: HuggingFace model name for LLaDA
        chunk_size: Number of tokens per chunk
        max_recursive_steps: Maximum recursion depth
        freeze_backbone: Whether to freeze LLaDA weights
        **kwargs: Additional config parameters

    Returns:
        LLaDATRMHybrid model
    """
    config = {
        "llada_model_name": llada_model_name,
        "freeze_llada_backbone": freeze_backbone,
        "chunk_size": chunk_size,
        "max_recursive_steps": max_recursive_steps,
        **kwargs
    }

    return LLaDATRMHybrid(config)


if __name__ == "__main__":
    # Test instantiation
    print("Testing LLaDA-TRM Hybrid Model...")

    config = {
        "llada_model_name": "inclusionAI/LLaDA-MoE-7B-A1B-Instruct",
        "freeze_llada_backbone": True,
        "chunk_size": 16,
        "max_recursive_steps": 8,
        "head_hidden_size": 512,
        "head_layers": 2,
    }

    model = LLaDATRMHybrid(config)
    print(f"✓ Model created successfully")
    print(f"✓ Total parameters: {model.count_parameters() / 1e6:.2f}M")
    print(f"✓ Head parameters: {model.count_head_parameters() / 1e6:.2f}M")
