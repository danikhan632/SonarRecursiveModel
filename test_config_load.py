#!/usr/bin/env python3
"""
Quick test to validate config loading
"""
import yaml

def safe_float(value, default=0.0):
    """Safely convert value to float"""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int(value, default=0):
    """Safely convert value to int"""
    if value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

# Load config
config_path = "config/slot_trm_projection_warmup.yaml"
print(f"Loading config from: {config_path}")

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Test critical values
train_cfg = config['training']

print("\nCritical values from config:")
print(f"  lr: {train_cfg['lr']} (type: {type(train_cfg['lr']).__name__})")
print(f"  weight_decay: {train_cfg['weight_decay']} (type: {type(train_cfg['weight_decay']).__name__})")
print(f"  batch_size: {train_cfg['batch_size']} (type: {type(train_cfg['batch_size']).__name__})")
print(f"  num_epochs: {train_cfg['num_epochs']} (type: {type(train_cfg['num_epochs']).__name__})")

# Test safe conversion
print("\nAfter safe conversion:")
print(f"  lr: {safe_float(train_cfg['lr'], 1e-4)}")
print(f"  weight_decay: {safe_float(train_cfg['weight_decay'], 0.01)}")
print(f"  batch_size: {safe_int(train_cfg['batch_size'], 4)}")
print(f"  num_epochs: {safe_int(train_cfg['num_epochs'], 5)}")

print("\n✓ Config loading test passed!")
