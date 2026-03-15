"""
Fixed evaluation for fine-tuning experiments.
Computes eval loss on the held-out validation set (assistant tokens only).

DO NOT MODIFY — this is the ground truth metric.

Usage (called by finetune.py, not directly):
    from evaluate import evaluate_model, get_config
    eval_loss = evaluate_model(model, tokenizer, batch_size=4)
"""

import json
from pathlib import Path

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader

CACHE_DIR = Path.home() / ".cache" / "autoexperiments" / "llm-finetune"
DATA_DIR = CACHE_DIR / "data"


def get_val_dataset():
    """Load the pre-tokenized validation dataset."""
    val_path = DATA_DIR / "val"
    if not val_path.exists():
        raise RuntimeError(f"Validation data not found at {val_path}. Run prepare_data.py first.")
    return load_from_disk(str(val_path))


def get_config():
    """Load the data preparation config."""
    config_path = CACHE_DIR / "config.json"
    return json.loads(config_path.read_text())


@torch.no_grad()
def evaluate_model(model, tokenizer, batch_size=4, max_batches=None):
    """
    Evaluate model on the validation set.
    Loss is computed only on assistant tokens (labels != -100),
    matching the training objective.

    Args:
        model: The fine-tuned model.
        tokenizer: The tokenizer (for pad_token_id).
        batch_size: Evaluation batch size.
        max_batches: Cap on number of batches (None = full eval set).

    Returns:
        float: Average eval loss (assistant tokens only).
    """
    model.eval()
    val_ds = get_val_dataset()
    val_ds.set_format("torch")

    dataloader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    total_loss = 0.0
    total_batches = 0
    device = next(model.parameters()).device

    for i, batch in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Labels already have -100 for non-assistant tokens from prepare_data.py
        # No additional masking needed
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        total_loss += outputs.loss.item()
        total_batches += 1

    avg_loss = total_loss / max(total_batches, 1)
    return avg_loss
