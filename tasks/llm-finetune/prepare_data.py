"""
One-time data preparation for Qwen3.5-2B fine-tuning with thinking traces.

Downloads 3 datasets, normalizes to Qwen ChatML format with <think>...</think>,
tokenizes with assistant-only label masking, and saves train/val splits.

DO NOT MODIFY — this is the fixed data pipeline.

Usage:
    python3 prepare_data.py
    python3 prepare_data.py --total-samples 50000
    python3 prepare_data.py --thinking-ratio 0.8
"""

import argparse
import json
import re
import os
from pathlib import Path
from dotenv import load_dotenv

# Load HF_TOKEN from .env if present
load_dotenv()

from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = Path.home() / ".cache" / "autoexperiments" / "llm-finetune"
DATA_DIR = CACHE_DIR / "data"

DEFAULT_MODEL = "Qwen/Qwen3.5-2B"
DEFAULT_MAX_SEQ_LEN = 1024
DEFAULT_TOTAL_SAMPLES = 30000
DEFAULT_THINKING_RATIO = 0.75  # 75% thinking, 25% non-thinking
DEFAULT_VAL_RATIO = 0.05

# Dataset sources
THINKING_DATASETS = [
    {
        "name": "Magpie-Align/Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B",
        "weight": 2,  # 2/3 of thinking data
    },
    {
        "name": "open-thoughts/OpenThoughts-114k",
        "weight": 1,  # 1/3 of thinking data
    },
]
NON_THINKING_DATASET = "HuggingFaceH4/ultrachat_200k"

# Markers to detect/convert thinking traces from various source formats
THINK_PATTERNS = [
    # DeepSeek R1 style
    (re.compile(r"<think>(.*?)</think>", re.DOTALL), None),
    # OpenThoughts style
    (re.compile(r"<\|begin_of_thought\|>(.*?)<\|end_of_thought\|>", re.DOTALL),
     re.compile(r"<\|begin_of_solution\|>(.*?)<\|end_of_solution\|>", re.DOTALL)),
]

SYSTEM_PROMPT = "You are a helpful assistant. Think carefully before answering."

# ---------------------------------------------------------------------------
# Dataset loading and normalization
# ---------------------------------------------------------------------------

def extract_thinking_and_response(text: str) -> tuple[str | None, str]:
    """
    Extract thinking trace and final response from assistant text.
    Handles multiple source formats. Returns (thinking, response).
    """
    # Try each pattern
    for think_pat, solution_pat in THINK_PATTERNS:
        think_match = think_pat.search(text)
        if think_match:
            thinking = think_match.group(1).strip()
            if solution_pat:
                sol_match = solution_pat.search(text)
                response = sol_match.group(1).strip() if sol_match else text[think_match.end():].strip()
            else:
                response = text[think_match.end():].strip()
            return thinking, response

    # No thinking markers found — treat entire text as thinking + response
    # (for datasets where the whole response IS the reasoning chain)
    return None, text


def format_assistant_with_thinking(text: str) -> str:
    """Format assistant response with <think>...</think> block for Qwen."""
    thinking, response = extract_thinking_and_response(text)
    if thinking:
        return f"<think>\n{thinking}\n</think>\n\n{response}"
    else:
        # Raw text — wrap entire thing as thinking + repeat last part as response
        # This handles datasets where reasoning IS the response
        return f"<think>\n{text}\n</think>\n\n{response}"


def format_assistant_no_thinking(text: str) -> str:
    """Format assistant response with empty <think> block for Qwen non-thinking mode."""
    # Strip any existing thinking traces
    clean = text
    for think_pat, solution_pat in THINK_PATTERNS:
        clean = think_pat.sub("", clean)
        if solution_pat:
            match = solution_pat.search(clean)
            if match:
                clean = match.group(1)
    clean = clean.strip()
    return f"<think>\n\n</think>\n\n{clean}"


def normalize_messages(example: dict) -> list[dict] | None:
    """
    Convert a dataset example to a list of {"role": ..., "content": ...} messages.
    Handles ShareGPT, OpenAI, and other common formats. Returns None if unparseable.
    """
    # OpenAI / HuggingFace messages format
    if "messages" in example:
        msgs = example["messages"]
        if isinstance(msgs, list) and len(msgs) > 0:
            if isinstance(msgs[0], dict) and "role" in msgs[0]:
                return msgs

    # ShareGPT format
    if "conversations" in example:
        convs = example["conversations"]
        if isinstance(convs, list) and len(convs) > 0:
            role_map = {"human": "user", "gpt": "assistant", "user": "user", "assistant": "assistant", "system": "system"}
            messages = []
            for turn in convs:
                role = role_map.get(turn.get("from", ""), turn.get("role", ""))
                content = turn.get("value", turn.get("content", ""))
                if role and content:
                    messages.append({"role": role, "content": content})
            if messages:
                return messages

    # Instruction/input/output format
    if "instruction" in example:
        messages = [{"role": "user", "content": example["instruction"]}]
        if example.get("input", "").strip():
            messages[0]["content"] += f"\n\n{example['input']}"
        if "output" in example:
            messages.append({"role": "assistant", "content": example["output"]})
        elif "response" in example:
            messages.append({"role": "assistant", "content": example["response"]})
        return messages

    # prompt/response format
    if "prompt" in example and ("response" in example or "completion" in example):
        resp = example.get("response", example.get("completion", ""))
        return [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": resp},
        ]

    return None


def _load_dataset_slice(dataset_name: str, n_samples: int, split: str = "train", config: str | None = None) -> Dataset:
    """Load a dataset slice. Uses split indexing to avoid downloading the entire dataset."""
    args = [dataset_name]
    if config:
        args.append(config)
    # Try with sample limit in split string (avoids full download)
    try:
        return load_dataset(*args, split=f"{split}[:{n_samples}]")
    except Exception as e1:
        print(f"    Slice load failed ({e1}), trying full split...")
    # Fallback: load full split
    return load_dataset(*args, split=split)


def load_thinking_data(dataset_name: str, n_samples: int, config: str | None = None) -> list[list[dict]]:
    """Load a thinking dataset and return normalized conversations with thinking traces."""
    print(f"  Loading {dataset_name} ({n_samples} samples)...")
    ds = _load_dataset_slice(dataset_name, n_samples * 2, config=config)  # load extra to account for skips

    conversations = []
    for i, example in enumerate(ds):
        if len(conversations) >= n_samples:
            break
        messages = normalize_messages(example)
        if not messages:
            continue

        # Ensure there's at least one user and one assistant message
        has_user = any(m["role"] == "user" for m in messages)
        has_assistant = any(m["role"] == "assistant" for m in messages)
        if not has_user or not has_assistant:
            continue

        # Format assistant messages with thinking
        formatted = []
        for msg in messages:
            if msg["role"] == "assistant":
                formatted.append({
                    "role": "assistant",
                    "content": format_assistant_with_thinking(msg["content"]),
                })
            else:
                formatted.append(msg)
        conversations.append(formatted)

    print(f"    Got {len(conversations)} conversations")
    return conversations


def load_non_thinking_data(dataset_name: str, n_samples: int) -> list[list[dict]]:
    """Load a non-thinking dataset and return normalized conversations with empty think blocks."""
    print(f"  Loading {dataset_name} ({n_samples} samples)...")
    try:
        ds = _load_dataset_slice(dataset_name, n_samples * 2, split="train_sft")
    except Exception:
        ds = _load_dataset_slice(dataset_name, n_samples * 2, split="train")

    conversations = []
    for i, example in enumerate(ds):
        if len(conversations) >= n_samples:
            break
        messages = normalize_messages(example)
        if not messages:
            continue

        has_user = any(m["role"] == "user" for m in messages)
        has_assistant = any(m["role"] == "assistant" for m in messages)
        if not has_user or not has_assistant:
            continue

        # Format assistant messages without thinking
        formatted = []
        for msg in messages:
            if msg["role"] == "assistant":
                formatted.append({
                    "role": "assistant",
                    "content": format_assistant_no_thinking(msg["content"]),
                })
            else:
                formatted.append(msg)
        conversations.append(formatted)

    print(f"    Got {len(conversations)} conversations")
    return conversations


# ---------------------------------------------------------------------------
# Tokenization with assistant-only label masking
# ---------------------------------------------------------------------------

def tokenize_conversation(
    messages: list[dict],
    tokenizer,
    max_seq_len: int,
    system_prompt: str = SYSTEM_PROMPT,
) -> dict | None:
    """
    Tokenize a conversation using the Qwen chat template.
    Returns input_ids, attention_mask, and labels (with -100 for non-assistant tokens).
    """
    # Prepend system message if not present
    if not messages or messages[0]["role"] != "system":
        messages = [{"role": "system", "content": system_prompt}] + messages

    # Tokenize full conversation
    try:
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=True,
        )
    except Exception:
        # Fallback if enable_thinking not supported
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    full_tokens = tokenizer(
        full_text,
        truncation=True,
        max_length=max_seq_len,
        padding="max_length",
        return_tensors=None,
    )

    input_ids = full_tokens["input_ids"]
    attention_mask = full_tokens["attention_mask"]

    # Build labels: -100 for everything except assistant responses
    labels = [-100] * len(input_ids)

    # Find assistant response spans by tokenizing incrementally
    # Strategy: tokenize up to each assistant turn start, then up to turn end
    prefix = ""
    for i, msg in enumerate(messages):
        if msg["role"] == "assistant":
            # Tokenize everything before this assistant's content
            pre_messages = messages[:i] + [{"role": "assistant", "content": ""}]
            try:
                pre_text = tokenizer.apply_chat_template(
                    pre_messages,
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=True,
                )
            except Exception:
                pre_text = tokenizer.apply_chat_template(
                    pre_messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            # Remove the empty assistant ending to get the prefix before content
            # The template adds <|im_start|>assistant\n...<|im_end|>
            # We want the position right after <|im_start|>assistant\n
            pre_tokens = tokenizer(pre_text, truncation=False, return_tensors=None)["input_ids"]

            # Tokenize up to and including this assistant turn
            post_messages = messages[:i + 1]
            try:
                post_text = tokenizer.apply_chat_template(
                    post_messages,
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=True,
                )
            except Exception:
                post_text = tokenizer.apply_chat_template(
                    post_messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            post_tokens = tokenizer(post_text, truncation=False, return_tensors=None)["input_ids"]

            # The assistant content spans from len(pre_tokens) to len(post_tokens)
            start = min(len(pre_tokens), max_seq_len)
            end = min(len(post_tokens), max_seq_len)

            for j in range(start, end):
                if j < len(labels):
                    labels[j] = input_ids[j]

    # Check we have some non-masked labels
    real_labels = sum(1 for l in labels if l != -100)
    if real_labels == 0:
        return None

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# Main preparation
# ---------------------------------------------------------------------------

def prepare(
    model_name: str = DEFAULT_MODEL,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    total_samples: int = DEFAULT_TOTAL_SAMPLES,
    thinking_ratio: float = DEFAULT_THINKING_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    config_path = CACHE_DIR / "config.json"

    # Check if already prepared with same settings
    if config_path.exists():
        existing = json.loads(config_path.read_text())
        if (existing.get("model") == model_name
                and existing.get("max_seq_len") == max_seq_len
                and existing.get("total_samples") == total_samples
                and existing.get("thinking_ratio") == thinking_ratio):
            print(f"Data already prepared at {DATA_DIR}")
            return
        print("Config changed, re-preparing...")

    # Calculate sample counts
    n_thinking = int(total_samples * thinking_ratio)
    n_non_thinking = total_samples - n_thinking

    # Split thinking samples across datasets by weight
    total_weight = sum(d["weight"] for d in THINKING_DATASETS)
    thinking_counts = []
    for d in THINKING_DATASETS:
        count = int(n_thinking * d["weight"] / total_weight)
        thinking_counts.append(count)

    print(f"Data blend plan:")
    for d, count in zip(THINKING_DATASETS, thinking_counts):
        print(f"  {d['name']}: {count} (thinking)")
    print(f"  {NON_THINKING_DATASET}: {n_non_thinking} (non-thinking)")
    print()

    # Load datasets
    print("Loading thinking datasets...")
    all_conversations = []
    for d, count in zip(THINKING_DATASETS, thinking_counts):
        convs = load_thinking_data(d["name"], count, config=d.get("config"))
        all_conversations.extend(convs)

    print("\nLoading non-thinking dataset...")
    non_thinking_convs = load_non_thinking_data(NON_THINKING_DATASET, n_non_thinking)
    all_conversations.extend(non_thinking_convs)

    print(f"\nTotal conversations: {len(all_conversations)}")

    # Tokenize
    print(f"\nLoading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Tokenizing {len(all_conversations)} conversations (max_seq_len={max_seq_len})...")
    tokenized = []
    skipped = 0
    for i, conv in enumerate(all_conversations):
        result = tokenize_conversation(conv, tokenizer, max_seq_len)
        if result is not None:
            tokenized.append(result)
        else:
            skipped += 1
        if (i + 1) % 5000 == 0:
            print(f"  Processed {i + 1}/{len(all_conversations)} ({skipped} skipped)")

    print(f"  Tokenized: {len(tokenized)}, Skipped: {skipped}")

    # Split train/val
    import random
    random.seed(42)
    random.shuffle(tokenized)
    val_size = max(1, int(len(tokenized) * val_ratio))
    val_data = tokenized[:val_size]
    train_data = tokenized[val_size:]

    print(f"\nTrain: {len(train_data)}, Val: {len(val_data)}")

    # Save as HF datasets
    train_ds = Dataset.from_list(train_data)
    val_ds = Dataset.from_list(val_data)

    train_ds.save_to_disk(str(DATA_DIR / "train"))
    val_ds.save_to_disk(str(DATA_DIR / "val"))

    # Save config
    config = {
        "model": model_name,
        "max_seq_len": max_seq_len,
        "total_samples": total_samples,
        "thinking_ratio": thinking_ratio,
        "train_size": len(train_data),
        "val_size": len(val_data),
        "datasets": {
            "thinking": [{"name": d["name"], "count": c} for d, c in zip(THINKING_DATASETS, thinking_counts)],
            "non_thinking": {"name": NON_THINKING_DATASET, "count": n_non_thinking},
        },
    }
    config_path.write_text(json.dumps(config, indent=2))

    print(f"\nDone! Data saved to {DATA_DIR}")
    print(f"Config: {json.dumps(config, indent=2)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare fine-tuning data with thinking traces")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name for tokenizer")
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument("--total-samples", type=int, default=DEFAULT_TOTAL_SAMPLES)
    parser.add_argument("--thinking-ratio", type=float, default=DEFAULT_THINKING_RATIO)
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    args = parser.parse_args()

    prepare(
        model_name=args.model,
        max_seq_len=args.max_seq_len,
        total_samples=args.total_samples,
        thinking_ratio=args.thinking_ratio,
        val_ratio=args.val_ratio,
    )
