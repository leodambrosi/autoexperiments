"""Run multiple fine-tuning experiments with different configs."""

import math
import os
import time
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate import evaluate_model, get_config

CACHE_DIR = Path.home() / ".cache" / "autoexperiments" / "llm-finetune"
DATA_DIR = CACHE_DIR / "data"
TIME_BUDGET = 300

EXPERIMENTS = {
    "baseline": {
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "lr": 1e-4,
        "grad_accum": 8,
    },
    "mlp_targets": {
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "lr": 1e-4,
        "grad_accum": 8,
    },
    "rank32_mlp_accum4": {
        "lora_r": 32,
        "lora_alpha": 64,
        "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "lr": 1e-4,
        "grad_accum": 4,
    },
    "rank32_mlp_accum4_lr5e5": {
        "lora_r": 32,
        "lora_alpha": 64,
        "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "lr": 5e-5,
        "grad_accum": 4,
    },
}


def run_experiment(name, cfg):
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {name}")
    print(f"  LoRA r={cfg['lora_r']}, alpha={cfg['lora_alpha']}")
    print(f"  targets={cfg['lora_targets']}")
    print(f"  lr={cfg['lr']}, grad_accum={cfg['grad_accum']}")
    print(f"{'='*60}\n")

    t_start = time.time()
    torch.manual_seed(42)

    config = get_config()
    model_name = config["model"]
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=model_dtype, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=0.05,
        target_modules=cfg["lora_targets"],
    )
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}")
    model.to(device)

    train_ds = load_from_disk(str(DATA_DIR / "train"))
    train_ds.set_format("torch")
    batch_size = 1
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    grad_accum = cfg["grad_accum"]
    lr = cfg["lr"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    est_micro_step_time = 2.5
    total_micro_steps = int(TIME_BUDGET / est_micro_step_time)
    total_optim_steps = max(1, total_micro_steps // grad_accum)
    warmup_optim_steps = max(1, int(0.06 * total_optim_steps))
    optim_step_count = 0

    def get_lr(_):
        if optim_step_count < warmup_optim_steps:
            return optim_step_count / max(warmup_optim_steps, 1)
        progress = (optim_step_count - warmup_optim_steps) / max(total_optim_steps - warmup_optim_steps, 1)
        return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

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

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / grad_accum
            loss.backward()
            accumulated_loss += loss.item()

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optim_step_count += 1
                scheduler.step()
                optimizer.zero_grad()

            step += 1
            dt = time.time() - t0
            total_training_time += dt

            if step % log_interval == 0:
                avg_loss = accumulated_loss / log_interval
                current_lr = scheduler.get_last_lr()[0]
                remaining = max(0, TIME_BUDGET - total_training_time)
                print(f"\r  step {step:05d} | loss: {avg_loss:.4f} | lr: {current_lr:.2e} | dt: {dt*1000:.0f}ms | remaining: {remaining:.0f}s    ", end="", flush=True)
                accumulated_loss = 0.0
                if math.isnan(avg_loss) or avg_loss > 100:
                    print(f"\n  FAIL: loss diverged in {name}")
                    return name, float("nan"), trainable, 0

            if total_training_time >= TIME_BUDGET:
                break
        if total_training_time >= TIME_BUDGET:
            break

    print(f"\n  Training done: {step} steps, {optim_step_count} optim steps, {epoch} epochs")

    print("  Evaluating...")
    eval_loss = evaluate_model(model, tokenizer, batch_size=1)
    t_end = time.time()
    print(f"  eval_loss: {eval_loss:.6f} | total: {t_end - t_start:.0f}s")

    # Free memory
    del model, optimizer, scheduler, train_loader, train_ds
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return name, eval_loss, trainable, optim_step_count


if __name__ == "__main__":
    results = []
    for name, cfg in EXPERIMENTS.items():
        result = run_experiment(name, cfg)
        results.append(result)

    print(f"\n\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Experiment':<30} {'eval_loss':>10} {'params':>12} {'optim_steps':>12}")
    print(f"{'-'*30} {'-'*10} {'-'*12} {'-'*12}")
    for name, loss, params, steps in results:
        print(f"{name:<30} {loss:>10.6f} {params:>12,} {steps:>12}")

    best = min(results, key=lambda x: x[1])
    print(f"\nBest: {best[0]} (eval_loss={best[1]:.6f})")
