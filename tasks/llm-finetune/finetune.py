"""
Fine-tuning script — THIS IS THE FILE THE AGENT MODIFIES.

Baseline: LoRA fine-tuning of Qwen3.5-2B with thinking traces on a blend of
Magpie-Reasoning-V2, R1-Distill-SFT (thinking), and UltraChat (non-thinking).

Labels are pre-masked to -100 for non-assistant tokens by prepare_data.py,
so the model only learns to generate assistant responses (including <think> traces).

Everything here is fair game: LoRA config, learning rate, scheduler,
batch size, training strategy, loss function, etc.

Usage: python3 finetune.py
"""

import math
import time
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate import evaluate_model, get_config

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = Path.home() / ".cache" / "autoexperiments" / "llm-finetune"
DATA_DIR = CACHE_DIR / "data"
TIME_BUDGET = 300  # 5 minutes of training time

# Hyperparameters (edit these!)
LEARNING_RATE = 1e-4
BATCH_SIZE = 1 if (torch.backends.mps.is_available() and not torch.cuda.is_available()) else 4
GRADIENT_ACCUMULATION_STEPS = 4
WARMUP_RATIO = 0.01
WEIGHT_DECAY = 0.05
MAX_GRAD_NORM = 1.0

# LoRA config (edit these!)
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Eval
EVAL_BATCH_SIZE = 1 if (torch.backends.mps.is_available() and not torch.cuda.is_available()) else 4

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)

config = get_config()
model_name = config["model"]
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = get_device()
if device.type == "cuda":
    model_dtype = torch.bfloat16
else:
    # MPS: float16 causes NaN in Qwen3.5's gated delta rule attention fallback
    model_dtype = torch.float32

print(f"Device: {device}")
print(f"Dtype: {model_dtype}")
print(f"Model: {model_name}")
print(f"Data: {config['train_size']} train, {config['val_size']} val")

# Load model
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=model_dtype,
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Apply LoRA
print("Applying LoRA...")
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=LORA_TARGET_MODULES,
)
model = get_peft_model(model, lora_config)
model.gradient_checkpointing_enable()
model.print_trainable_parameters()
model.to(device)

# Load pre-tokenized data (labels already have -100 masking from prepare_data.py)
print("Loading data...")
train_ds = load_from_disk(str(DATA_DIR / "train"))
train_ds.set_format("torch")
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

# LR scheduler — estimate total optimizer steps from time budget
# Each micro-step takes ~2.5s on MPS, ~0.5s on CUDA
est_micro_step_time = 2.5 if device.type == "mps" else 0.5
total_micro_steps = int(TIME_BUDGET / est_micro_step_time)
total_optim_steps = max(1, total_micro_steps // GRADIENT_ACCUMULATION_STEPS)
warmup_optim_steps = max(1, int(WARMUP_RATIO * total_optim_steps))

optim_step_count = 0

def get_lr(_unused_step):
    if optim_step_count < warmup_optim_steps:
        return optim_step_count / max(warmup_optim_steps, 1)
    progress = (optim_step_count - warmup_optim_steps) / max(total_optim_steps - warmup_optim_steps, 1)
    return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

print(f"\nStarting training (time budget: {TIME_BUDGET}s)...")
t_train_start = time.time()
step = 0
epoch = 0
total_training_time = 0
accumulated_loss = 0.0
log_interval = 10

model.train()
while True:
    epoch += 1
    for batch in train_loader:
        t0 = time.time()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        # Labels already have -100 for non-assistant tokens — no extra masking needed

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS
        loss.backward()

        accumulated_loss += loss.item()

        if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            optim_step_count += 1
            scheduler.step()
            optimizer.zero_grad()

        step += 1
        dt = time.time() - t0
        total_training_time += dt

        if step % log_interval == 0:
            avg_loss = accumulated_loss / log_interval
            lr = scheduler.get_last_lr()[0]
            remaining = max(0, TIME_BUDGET - total_training_time)
            print(f"\rstep {step:05d} | loss: {avg_loss:.4f} | lr: {lr:.2e} | dt: {dt*1000:.0f}ms | remaining: {remaining:.0f}s    ", end="", flush=True)
            accumulated_loss = 0.0

            # Fast fail
            if math.isnan(avg_loss) or avg_loss > 100:
                print("\nFAIL: loss diverged")
                exit(1)

        if total_training_time >= TIME_BUDGET:
            break

    if total_training_time >= TIME_BUDGET:
        break

print(f"\n\nTraining complete: {step} steps, {epoch} epochs")

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

print("Evaluating...")
eval_loss = evaluate_model(model, tokenizer, batch_size=EVAL_BATCH_SIZE)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

t_end = time.time()
if device.type == "cuda":
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
elif device.type == "mps":
    peak_vram_mb = torch.mps.driver_allocated_memory() / 1024 / 1024
else:
    peak_vram_mb = 0
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())

print("---")
print(f"eval_loss:        {eval_loss:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"num_steps:        {step}")
print(f"epochs:           {epoch}")
print(f"trainable_params: {trainable_params:,}")
print(f"total_params:     {total_params:,}")
print(f"lora_r:           {LORA_R}")
print(f"learning_rate:    {LEARNING_RATE}")
print(f"batch_size:       {BATCH_SIZE}")
