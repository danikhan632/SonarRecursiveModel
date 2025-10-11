#!/usr/bin/env python3
"""
Prepare SONAR embedding sequences for TRM LLM training.

Usage:
  python prepare_sonar_data.py \
    --dataset-name wikitext \
    --dataset-config wikitext-2-raw-v1 \
    --train-samples 1000 --val-samples 100 \
    --seq-length 10 --output-dir data_chunks \
    --lang eng_Latn
"""
import os
import torch
import argparse
from tqdm import tqdm
from datasets import load_dataset
from sonar_simple import SimpleSonarEncoder


def prepare_and_save_in_chunks(
    dataset, num_samples, seq_length, encoder, lang, batch_size, chunk_size, output_dir, split_name
):
    """
    Processes a dataset, generates embedding sequences, and saves them to disk
    incrementally in chunks to conserve memory.
    """
    texts = []
    print(f"Collecting text segments for {split_name} split...")
    for item in tqdm(dataset):
        text = item.get("text", "").strip()
        if text and len(text) > 10:
            texts.append(text)
            if len(texts) >= num_samples * (seq_length + 1):
                break
    print(f"Collected {len(texts)} text segments.")

    # Create all the sequences of texts first
    list_of_seq_texts = []
    for i in range(0, len(texts) - seq_length):
        if len(list_of_seq_texts) >= num_samples:
            break
        list_of_seq_texts.append(texts[i : i + seq_length + 1])

    if not list_of_seq_texts:
        print(f"No sequences could be created for {split_name} split.")
        return 0

    total_sequences_saved = 0
    chunk_num = 0

    print(f"Processing and saving {split_name} data in chunks of {chunk_size}...")
    # Outer progress bar for chunks
    for i in tqdm(range(0, len(list_of_seq_texts), chunk_size), desc=f"Processing {split_name} Chunks"):
        chunk_of_text_seqs = list_of_seq_texts[i : i + chunk_size]
        if not chunk_of_text_seqs:
            continue

        # 1. Flatten this chunk for batch encoding
        flat_texts_chunk = [text for seq in chunk_of_text_seqs for text in seq]

        # 2. Encode this chunk in batches with a nested progress bar
        all_embeddings_chunk = []
        for j in tqdm(range(0, len(flat_texts_chunk), batch_size), desc="Encoding Sentences", leave=False):
            batch_texts = flat_texts_chunk[j : j + batch_size]
            batch_embeddings = encoder.encode(batch_texts, lang=lang)
            all_embeddings_chunk.append(batch_embeddings)
        
        if not all_embeddings_chunk:
            continue
        flat_embeddings_chunk = torch.cat(all_embeddings_chunk, dim=0)
        del all_embeddings_chunk  # Free memory from the list of tensors

        # 3. Un-flatten the chunk back into sequences
        step = seq_length + 1
        num_encoded_in_chunk = flat_embeddings_chunk.shape[0]
        chunk_sequences = []
        for k in range(0, num_encoded_in_chunk, step):
            if k + step > num_encoded_in_chunk: continue
            seq_emb = flat_embeddings_chunk[k:k+step]
            chunk_sequences.append(seq_emb.cpu())

        del flat_embeddings_chunk # Free memory from the large GPU tensor

        # 4. Save this chunk of sequences to a file
        if chunk_sequences:
            out_path = os.path.join(
                output_dir, f"{split_name}_embeddings_chunk_{chunk_num}.pt"
            )
            torch.save(chunk_sequences, out_path)
            total_sequences_saved += len(chunk_sequences)
            chunk_num += 1
        
        # Explicitly clear CUDA cache to prevent OOM on next chunk
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"✓ Saved {total_sequences_saved} total sequences in {chunk_num} chunks for split '{split_name}'.")
    return total_sequences_saved


def main():
    parser = argparse.ArgumentParser(
        description="Prepare SONAR embeddings for TRM training"
    )
    parser.add_argument("--dataset-name",   type=str, required=True)
    parser.add_argument("--dataset-config", type=str, default=None)
    parser.add_argument("--train-samples",  type=int, default=1000)
    parser.add_argument("--val-samples",    type=int, default=100)
    parser.add_argument("--seq-length",     type=int, default=10,
                        help="Number of sentences per sequence (excluding target)")
    parser.add_argument("--output-dir",     type=str, default="data_chunks")
    parser.add_argument("--chunk-size",     type=int, default=1000,
                        help="Number of sequences to save per chunk file.")
    parser.add_argument("--batch-size",     type=int, default=256,
                        help="Batch size for the SONAR encoder.")
    parser.add_argument("--lang",           type=str, default="eng_Latn")
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Loading SONAR encoder on device {args.device}...")
    encoder = SimpleSonarEncoder(device=args.device)

    print("Loading dataset...")
    ds_train = load_dataset(
        args.dataset_name, args.dataset_config, split="train"
    )
    ds_val   = load_dataset(
        args.dataset_name, args.dataset_config, split="validation"
    )

    train_count = prepare_and_save_in_chunks(
        ds_train, args.train_samples, args.seq_length, encoder, args.lang, 
        args.batch_size, args.chunk_size, args.output_dir, "train"
    )
    val_count = prepare_and_save_in_chunks(
        ds_val, args.val_samples, args.seq_length, encoder, args.lang, 
        args.batch_size, args.chunk_size, args.output_dir, "val"
    )

    print("\nDATA PREPARATION COMPLETE")
    if train_count > 0:
        print(f"Total train sequences: {train_count}")
        print(f"Total validation sequences: {val_count}")
        print("You can now train a model using these chunks.")
    else:
        print("No data was generated. Check dataset and parameters.")


if __name__ == "__main__":
    main()