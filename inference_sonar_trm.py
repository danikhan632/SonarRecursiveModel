#!/usr/bin/env python3
"""
Inference script for a trained TinyRecursiveModel (TRM) on SONAR embeddings.

This script performs the following steps:
1.  Loads a pre-trained RecursiveLLM model checkpoint.
2.  Loads the SONAR embedding dataset (`train_embeddings.pt` or `val_embeddings.pt`).
3.  Selects a random sample (sequence of embeddings) from the dataset.
4.  Feeds a prefix of the sequence to the model to predict the next embedding.
5.  Loads the SONAR text decoder.
6.  Decodes and prints the following for comparison:
    - The last sentence from the input prefix.
    - The ground truth next sentence.
    - The model's predicted next sentence.

Example usage:
    python inference_sonar_trm.py \
        --model_path "outputs/2025-10-10/some-run/checkpoints/model_step_1000.pt" \
        --config "outputs/2025-10-10/some-run/.hydra/config.yaml"
"""

import argparse
import random
import sys
from pathlib import Path
import os
import glob

import torch
from transformers.models.nllb.tokenization_nllb import NllbTokenizer
from transformers.modeling_outputs import BaseModelOutput

# Add local modules to path
sys.path.insert(0, str(Path(__file__).parent.absolute()))

from models.recursive_reasoning.recursive_llm import RecursiveLLM
from sonar_stuff.m2m_100.modeling_m2m_100 import M2M100DecoderModel

# --- Decoder Functions (from sonar_hf.py) ---

def get_decoder_and_tokenizer(device='cuda'):
    """Loads the SONAR decoder and NLLB tokenizer."""
    print("[1/4] Loading tokenizer...")
    tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    tokenizer.src_lang = "eng_Latn"
    print("✓ Loaded NLLB tokenizer")

    print("\n[2/4] Loading SONAR decoder...")
    try:
        decoder = M2M100DecoderModel.from_pretrained(
            "cointegrated/SONAR_200_text_decoder_hf"
        ).to(device)
        print("✓ Loaded decoder from HuggingFace Hub")
    except Exception as e:
        print(f"⚠ Could not load decoder from Hub: {e}")
        raise
    
    decoder.eval()
    return decoder, tokenizer

def decode_embeddings(embeddings, decoder, tokenizer, target_lang='eng_Latn', num_beams=5):
    """
    Decode SONAR embeddings to text.
    Args:
        embeddings: [batch, 1024] SONAR embeddings
    Returns:
        List of decoded text strings
    """
    if embeddings.dim() == 1:
        embeddings = embeddings.unsqueeze(0)
    if embeddings.dim() == 2:
        embeddings = embeddings.unsqueeze(1)

    device = embeddings.device

    with torch.inference_mode():
        encoder_outputs = BaseModelOutput(last_hidden_state=embeddings.to(device))
        gen_out = decoder.generate(
            encoder_outputs=encoder_outputs,
            num_beams=num_beams,
            max_length=100,
            use_cache=False,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_lang),
        )

    texts = tokenizer.batch_decode(gen_out, skip_special_tokens=True)
    return texts

# --- Main Inference Logic ---

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load SONAR Decoder
    decoder, tokenizer = get_decoder_and_tokenizer(device)

    # 2. Load TRM Model
    print("\n[3/4] Loading TRM model...")
    # Load the checkpoint which contains model args
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    train_args = checkpoint['args']

    # We need to know the sequence length from the data to build model config
    first_chunk_file = next(iter(sorted(glob.glob(os.path.join(args.data_folder, "*_embeddings_chunk_*.pt")))), None)
    if not first_chunk_file:
        raise FileNotFoundError(f"No chunk files found in {args.data_folder} to determine seq_len.")

    data_for_seq_len = torch.load(first_chunk_file)
    seq_len = data_for_seq_len[0].shape[0] - 1
    del data_for_seq_len # free memory

    # Re-build model config from saved training args
    cfg = dict(
        batch_size=train_args.batch_size,
        seq_len=seq_len,
        vocab_size=train_args.sonar_dim,
        H_cycles=train_args.H_cycles,
        L_cycles=train_args.L_cycles,
        L_layers=train_args.L_layers,
        hidden_size=train_args.sonar_dim,
        expansion=train_args.expansion,
        num_heads=train_args.num_heads,
        pos_encodings="none",
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        halt_max_steps=train_args.halt_max_steps,
        halt_exploration_prob=train_args.halt_exploration_prob,
        forward_dtype="bfloat16",
        no_ACT_continue=True,
        pretrained_model_name=None,
        freeze_embeddings=False
    )
    
    # Instantiate model from config
    model = RecursiveLLM(cfg).to(device)
    
    # Handle DataParallel or standard state_dict
    state_dict = checkpoint['model_state_dict']
    if any(key.startswith('module.') for key in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
    model.load_state_dict(state_dict)
    model.eval()
    print(f"✓ Loaded model from {args.model_path}")
    print(f"  Step: {checkpoint.get('step', 'N/A')}, Loss: {checkpoint.get('loss', 'N/A'):.4f}")

    # 3. Load Data
    print("\n[4/4] Loading data and running inference...")
    data = []
    chunk_files = sorted(glob.glob(os.path.join(args.data_folder, "*_embeddings_chunk_*.pt")))
    if not chunk_files:
        raise FileNotFoundError(f"No chunk files found in {args.data_folder}")

    print(f"Loading {len(chunk_files)} chunk files from {args.data_folder}...")
    for f in chunk_files:
        data.extend(torch.load(f))

    print(f"✓ Loaded {len(data)} total sequences from {args.data_folder}")

    # Select a sample
    sample_idx = args.sample_idx if args.sample_idx is not None else random.randint(0, len(data) - 1)
    print(f"→ Using sample index: {sample_idx}")
    sample_sequence = data[sample_idx].to(device).to(torch.bfloat16) # Match training dtype

    # Prepare for autoregressive prediction
    # Input is the sequence up to the second to last element
    # The model will predict the last element
    input_embeddings = sample_sequence[:-1].unsqueeze(0) # [1, seq_len, dim]
    
    # The last element of the input sequence (for context)
    last_input_embedding = sample_sequence[-2] # [dim]
    
    # The ground truth for the prediction
    ground_truth_embedding = sample_sequence[-1] # [dim]

    print(f"  Input shape: {input_embeddings.shape}")

    # 4. Run Inference
    with torch.inference_mode():
        # The model expects a carry state and a batch dictionary, like in training
        batch = {"input_embeddings": input_embeddings}
        carry = model.initial_carry(batch)
        carry, outputs = model(carry, batch)

        # The prediction is the last element of the logits sequence
        predicted_embedding = outputs['logits'][0, -1, :] # [dim]

    print("✓ Inference complete.")
    print(f"  Predicted embedding shape: {predicted_embedding.shape}")

    # 5. Decode and Compare
    print("\n" + "=" * 70)
    print("                DECODING RESULTS")
    print("=" * 70)

    # Batch decode for efficiency
    embeddings_to_decode = torch.stack([
        last_input_embedding,
        ground_truth_embedding,
        predicted_embedding
    ])
    
    # The decoder expects float32 inputs
    decoded_texts = decode_embeddings(embeddings_to_decode.to(torch.float32), decoder, tokenizer)

    last_input_text, ground_truth_text, predicted_text = decoded_texts

    print(f"\nContext (Last Input Sentence):")
    print(f"  '{last_input_text}'")
    
    print(f"\nGround Truth (Next Sentence):")
    print(f"  '{ground_truth_text}'")

    print(f"\nModel Prediction (Next Sentence):")
    print(f"  '{predicted_text}'")
    
    print("\n" + "=" * 70)
    print("Analysis:")
    print("The model was asked to predict the embedding for the 'Ground Truth' sentence,")
    print("given a sequence of sentences ending with the 'Context' sentence.")
    print("Compare the 'Model Prediction' with the 'Ground Truth'.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for TRM on SONAR embeddings.")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint (.pt file)."
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        default="data_chunks",
        help="Path to the folder containing data embeddings chunks."
    )
    parser.add_argument(
        "--sample_idx",
        type=int,
        default=None,
        help="Specific index of the sample to use. If not provided, a random sample is chosen."
    )

    args = parser.parse_args()
    
    try:
        main(args)
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()
