#!/usr/bin/env python3
"""
SONAR Encoder-Decoder using HuggingFace PR models

Based on: https://github.com/huggingface/transformers/pull/29646
Uses the standalone M2M100 models from the PR.

This is the simplest approach - no fairseq2 needed!

Install:
    pip install torch transformers sentencepiece

Usage:
    python sonar_hf_example.py
"""

import sys
from pathlib import Path
import torch
from transformers.models.nllb.tokenization_nllb import NllbTokenizer
from transformers.modeling_outputs import BaseModelOutput

# # Add local m2m_100 module to path (from the PR)
# sys.path.insert(0, str(Path(__file__).parent))

from m2m_100.modeling_m2m_100 import M2M100EncoderModel, M2M100DecoderModel


def main():
    print("=" * 70)
    print("SONAR Encoder-Decoder with HuggingFace PR Models")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # ========================================================================
    # 1. Load Tokenizer
    # ========================================================================
    print("\n[1/4] Loading tokenizer...")
    tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    tokenizer.src_lang = "eng_Latn"
    print("✓ Loaded NLLB tokenizer")

    # ========================================================================
    # 2. Load Encoder
    # ========================================================================
    print("\n[2/4] Loading SONAR encoder...")
    try:
        encoder = M2M100EncoderModel.from_pretrained(
            "cointegrated/SONAR_200_text_encoder_hf"
        ).to(device)
        print("✓ Loaded encoder from HuggingFace Hub")
    except Exception as e:
        print(f"⚠ Could not load from Hub: {e}")
        print("  Using alternative: cointegrated/SONAR_200_text_encoder")
        from transformers import AutoModel
        encoder = AutoModel.from_pretrained(
            "cointegrated/SONAR_200_text_encoder"
        ).to(device)
        print("✓ Loaded encoder (alternative)")

    encoder.eval()

    # ========================================================================
    # 3. Load Decoder
    # ========================================================================
    print("\n[3/4] Loading SONAR decoder...")
    try:
        decoder = M2M100DecoderModel.from_pretrained(
            "cointegrated/SONAR_200_text_decoder_hf"
        ).to(device)
        print("✓ Loaded decoder from HuggingFace Hub")
    except Exception as e:
        print(f"⚠ Could not load decoder from Hub: {e}")
        print("  This model may not be publicly available yet.")
        print("  See: https://github.com/huggingface/transformers/pull/29646")
        return

    decoder.eval()

    # ========================================================================
    # 4. Test Full Pipeline
    # ========================================================================
    print("\n[4/4] Testing full pipeline...")

    # Input sentences (from the PR test)
    sentences = [
        "My name is SONAR.",
        "I can embed the sentences into vectorial space.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is fascinating!"
    ]

    print(f"\nInput sentences ({len(sentences)} total):")
    for i, sent in enumerate(sentences, 1):
        print(f"  {i}. {sent}")

    # Tokenize
    print("\n→ Tokenizing...")
    batch = tokenizer(sentences, padding=True, return_tensors="pt").to(device)
    print(f"  ✓ Tokenized to shape: {batch['input_ids'].shape}")

    # Encode to SONAR embeddings
    print("\n→ Encoding to SONAR embeddings...")
    with torch.inference_mode():
        # Use pool_last_hidden_state=True to get pooled embeddings
        enc_out = encoder(**batch, pool_last_hidden_state=True)

    embeddings = enc_out.last_hidden_state  # [batch, 1, 1024]
    print(f"  ✓ Encoded to shape: {embeddings.shape}")
    print(f"  Embedding stats:")
    print(f"    Mean: {embeddings.mean().item():.6f}")
    print(f"    Std:  {embeddings.std().item():.6f}")

    # Decode SONAR embeddings to text
    print("\n→ Decoding embeddings to text...")
    with torch.inference_mode():
        # Clone encoder outputs to avoid in-place modifications during beam search
        encoder_outputs = BaseModelOutput(
            last_hidden_state=enc_out.last_hidden_state.clone()
        )

        # Generate with beam search
        gen_out = decoder.generate(
            encoder_outputs=encoder_outputs,
            num_beams=5,
            max_length=100,
            use_cache=False,  # Disable caching to avoid issues
            forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
        )

    # Decode tokens to text
    decoded_texts = tokenizer.batch_decode(gen_out, skip_special_tokens=True)
    print(f"  ✓ Decoded {len(decoded_texts)} texts")

    # ========================================================================
    # Display Results
    # ========================================================================
    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)

    for i, (original, decoded) in enumerate(zip(sentences, decoded_texts), 1):
        print(f"\n[{i}]")
        print(f"  Original: {original}")
        print(f"  Decoded:  {decoded}")

    # ========================================================================
    # Analysis
    # ========================================================================
    print("\n" + "=" * 70)
    print("Analysis")
    print("=" * 70)

    print("\nWhat happened:")
    print("  1. Text → Tokenized")
    print("  2. Tokens → SONAR embeddings (1024-dim, pooled)")
    print("  3. SONAR embeddings → Decoded text")
    print("")
    print("Expected behavior:")
    print("  ✓ Decoded text should be meaningful")
    print("  ✓ May not exactly match original (semantic, not lexical)")
    print("  ✓ Meaning should be preserved")
    print("")
    print("Why close but not exact?")
    print("  - SONAR captures meaning, not exact words")
    print("  - The decoder generates based on semantics")
    print("  - Similar meanings = similar embeddings!")

    print("\n" + "=" * 70)
    print("Complete!")
    print("=" * 70)


# Convenience functions for standalone use
def encode_texts(texts, encoder, tokenizer, lang='eng_Latn', device='cuda'):
    """
    Encode texts to SONAR embeddings.

    Args:
        texts: List of strings
        encoder: M2M100EncoderModel
        tokenizer: Tokenizer
        lang: Language code
        device: Device

    Returns:
        Pooled SONAR embeddings [len(texts), 1024]
    """
    tokenizer.src_lang = lang

    with torch.inference_mode():
        batch = tokenizer(texts, padding=True, return_tensors='pt').to(device)
        enc_out = encoder(**batch, pool_last_hidden_state=True)
        # Remove the sequence dimension (it's already pooled to 1)
        embeddings = enc_out.last_hidden_state.squeeze(1)

    return embeddings


def decode_embeddings(embeddings, decoder, tokenizer, target_lang='eng_Latn', num_beams=5):
    """
    Decode SONAR embeddings to text.

    Args:
        embeddings: [batch, 1024] SONAR embeddings
        decoder: M2M100DecoderModel
        tokenizer: Tokenizer
        target_lang: Target language code
        num_beams: Beam size

    Returns:
        List of decoded text strings
    """
    # Ensure embeddings have shape [batch, 1, 1024]
    if embeddings.dim() == 2:
        embeddings = embeddings.unsqueeze(1)

    device = embeddings.device

    with torch.inference_mode():
        encoder_outputs = BaseModelOutput(last_hidden_state=embeddings)

        gen_out = decoder.generate(
            encoder_outputs=encoder_outputs,
            num_beams=num_beams,
            max_length=100,
            use_cache=False,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_lang),
        )

    texts = tokenizer.batch_decode(gen_out, skip_special_tokens=True)
    return texts


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("\n" + "=" * 70)
        print("Troubleshooting")
        print("=" * 70)
        print("\nMake sure you have:")
        print("  pip install torch transformers sentencepiece")
        print("")
        print("The decoder model may not be publicly available yet.")
        print("Check the PR: https://github.com/huggingface/transformers/pull/29646")
