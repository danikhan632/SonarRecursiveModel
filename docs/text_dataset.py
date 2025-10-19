import torch
from torch.utils.data import Dataset
from pydantic import BaseModel
from datasets import load_dataset
from transformers import AutoTokenizer

class TextDatasetConfig(BaseModel):
    dataset_name: str = "wikitext"
    dataset_config_name: str = "wikitext-2-raw-v1"
    tokenizer_name: str = "hf-internal-testing/llama-tokenizer"
    seq_len: int
    # Compatibility fields
    rank: int = 0
    num_replicas: int = 1
    test_set_mode: bool = False
    epochs_per_iter: int = 1
    global_batch_size: int = 1

class TextDataset(Dataset):
    """
    A dataset that loads text from Hugging Face's `datasets` library,
    tokenizes it with a `transformers` tokenizer, and prepares it for pre-training.
    """
    def __init__(self, config: TextDatasetConfig, split: str = "train"):
        self.config = config
        self.seq_len = config.seq_len

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        # Load dataset from Hugging Face
        raw_dataset = load_dataset(config.dataset_name, config.dataset_config_name, split=split)

        # Tokenize the dataset
        def tokenize_function(examples):
            return self.tokenizer(examples["text"], add_special_tokens=False)

        tokenized_dataset = raw_dataset.map(
            tokenize_function, 
            batched=True, 
            remove_columns=raw_dataset.column_names
        )

        # Concatenate all tokens and create chunks
        all_input_ids = [id for ids in tokenized_dataset['input_ids'] for id in ids]

        self.examples = []
        for i in range(0, len(all_input_ids) - self.seq_len, self.seq_len):
            self.examples.append(torch.tensor(all_input_ids[i:i + self.seq_len], dtype=torch.long))

        # Create a dummy metadata object for compatibility
        self.metadata = self._create_metadata()

    def _create_metadata(self):
        class DummyMetadata:
            def __init__(self, vocab_size, seq_len, num_examples):
                self.vocab_size = vocab_size
                self.seq_len = seq_len
                self.total_groups = 1
                self.mean_puzzle_examples = num_examples
                self.num_puzzle_identifiers = 0
                self.sets = []

        return DummyMetadata(self.tokenizer.vocab_size, self.seq_len, len(self.examples))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        x = self.examples[idx]
        # The loss function will handle the shifting of labels internally
        return {"inputs": x, "labels": x.clone()}