import os, json, datetime
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from model import SonarAutoregressiveModel
from muon import SingleDeviceMuonWithAuxAdam
from sonar_simple import SimpleSonarEncoder
from sonar_normalizer import fit_sonar_normalizer_on_dataset, REASONING_SAMPLE_TEXTS

# ----------------------------
# Simple SONAR Embedding Dataset
# ----------------------------
class SonarEmbeddingDataset(Dataset):
    """
    Dataset that loads SONAR embeddings from preprocessed data.
    Each example is a sequence of SONAR embeddings.
    """
    def __init__(self, embeddings_file):
        # Load preprocessed embeddings
        # Expected format: list of tensors [num_examples, seq_len, sonar_dim]
        self.embeddings = torch.load(embeddings_file)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        # Return input and target (shifted by 1)
        emb_seq = self.embeddings[idx]

        # For autoregressive training: input is [:-1], target is [1:]
        input_emb = emb_seq[:-1]
        target_emb = emb_seq[1:]

        return {
            "input_embeddings": input_emb,
            "target_embeddings": target_emb
        }

# ----------------------------
# Configuration
# ----------------------------
def create_config():
    return {
        "model_config": {
            "d_model": 2048,
            "n_heads": 16,
            "d_ff": 8192,
            "dropout": 0.1,
            "sonar_dim": 1024,
            "n_layers": 6,
            "max_seq_len": 32
        },
        "training_config": {
            "batch_size": 8,
            "learning_rate": 1e-4,
            "num_epochs": 10,
            "grad_accum_steps": 4,
            "warmup_steps_ratio": 0.1,
            "mixed_precision": True,
        },
        "optimizer_config": {
            "muon_lr": 0.02,
            "adam_lr": 1e-4,
            "weight_decay": 0.01,
            "muon_momentum": 0.95
        },
        "paths": {
            "checkpoint_path": "sonar_model_checkpoint.pt",
            "weights_path": "sonar_model_weights.pt",
            "train_embeddings": "./data_chunks/train_embeddings.pt",
            "val_embeddings": "./data_chunks/val_embeddings.pt"
        }
    }

config = create_config()

# ----------------------------
# Setup
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load SONAR encoder for normalization fitting
print("Loading SONAR encoder...")
text2vec = SimpleSonarEncoder(device=device)

# ----------------------------
# Load/Create Datasets
# ----------------------------
print("Loading datasets...")

# Check if preprocessed embeddings exist
if not os.path.exists(config["paths"]["train_embeddings"]):
    print("ERROR: No preprocessed embeddings found!")
    print("You need to create training data first.")
    print("The dataset should contain sequences of SONAR embeddings.")
    exit(1)

train_dataset = SonarEmbeddingDataset(config["paths"]["train_embeddings"])
val_dataset = SonarEmbeddingDataset(config["paths"]["val_embeddings"])

train_loader = DataLoader(
    train_dataset,
    batch_size=config["training_config"]["batch_size"],
    shuffle=True,
    num_workers=2
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config["training_config"]["batch_size"],
    shuffle=False,
    num_workers=2
)

print(f"Train dataset: {len(train_dataset)} examples")
print(f"Val dataset: {len(val_dataset)} examples")

# ----------------------------
# Model
# ----------------------------
print("Initializing model...")
model_config = config["model_config"]
model = SonarAutoregressiveModel(model_config).to(device)

# Fit SONAR normalizer
print("Fitting SONAR normalizer...")
try:
    fit_sonar_normalizer_on_dataset(
        text2vec_pipeline=text2vec,
        sample_texts=REASONING_SAMPLE_TEXTS,
        normalizer=model.sonar_normalizer,
        batch_size=20
    )
    print("SONAR normalizer fitted successfully!")
except Exception as e:
    print(f"Warning: Could not fit SONAR normalizer: {e}")

# Count parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model parameters: {total_params:,}")

# ----------------------------
# Optimizer
# ----------------------------
print("Setting up optimizer...")

# Separate parameters for Muon (2D+ matrices) and Adam (biases, norms)
muon_params = []
adam_params = []

for n, p in model.named_parameters():
    if not p.requires_grad:
        continue

    if p.ndim >= 2 and not any(k in n.lower() for k in ["bias", "norm", "embedding"]):
        muon_params.append(p)
    else:
        adam_params.append(p)

print(f"Muon parameters: {len(muon_params)}, Adam parameters: {len(adam_params)}")

param_groups = [
    {"params": adam_params, "lr": config["optimizer_config"]["adam_lr"],
     "weight_decay": config["optimizer_config"]["weight_decay"], "use_muon": False},
    {"params": muon_params, "lr": config["optimizer_config"]["muon_lr"],
     "weight_decay": config["optimizer_config"]["weight_decay"], "use_muon": True,
     "momentum": config["optimizer_config"]["muon_momentum"]}
]

optimizer = SingleDeviceMuonWithAuxAdam(param_groups)

# ----------------------------
# Training Loop
# ----------------------------
scaler = torch.cuda.amp.GradScaler(enabled=(config["training_config"]["mixed_precision"] and device.type == "cuda"))

print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60)

for epoch in range(config["training_config"]["num_epochs"]):
    print(f"\nEpoch {epoch + 1}/{config['training_config']['num_epochs']}")

    # Training
    model.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    progress = tqdm(train_loader, desc=f"Training")
    for step, batch in enumerate(progress):
        input_emb = batch["input_embeddings"].to(device)
        target_emb = batch["target_embeddings"].to(device)

        with torch.amp.autocast("cuda", enabled=(config["training_config"]["mixed_precision"] and device.type == "cuda")):
            outputs = model(input_emb, target_embeddings=target_emb)
            loss = outputs["loss"]

        if loss is None or not torch.isfinite(loss):
            print(f"Warning: Invalid loss at step {step}")
            continue

        # Gradient accumulation
        loss_scaled = loss / config["training_config"]["grad_accum_steps"]
        scaler.scale(loss_scaled).backward()

        if (step + 1) % config["training_config"]["grad_accum_steps"] == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()
        progress.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_train_loss = total_loss / len(train_loader)
    print(f"Average Training Loss: {avg_train_loss:.4f}")

    # Validation
    model.eval()
    val_loss = 0.0

    with torch.inference_mode():
        for batch in tqdm(val_loader, desc="Validation"):
            input_emb = batch["input_embeddings"].to(device)
            target_emb = batch["target_embeddings"].to(device)

            outputs = model(input_emb, target_embeddings=target_emb)
            if outputs["loss"] is not None:
                val_loss += outputs["loss"].item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Average Validation Loss: {avg_val_loss:.4f}")

    # Save checkpoint
    torch.save(model.state_dict(), config["paths"]["weights_path"])
    torch.save({
        "epoch": epoch,
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss
    }, config["paths"]["checkpoint_path"])
    print(f"Checkpoint saved!")

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
