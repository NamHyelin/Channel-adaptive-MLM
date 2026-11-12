from typing import Tuple

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from config import DataConfig


def get_dataset_and_tokenizer(cfg: DataConfig):
    dataset = load_dataset(
        cfg.dataset_name,
        name=cfg.dataset_config_name,
        cache_dir=cfg.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    return dataset, tokenizer


def load_token_ids(dataset, tokenizer, split: str = "train") -> torch.Tensor:
    if split in dataset:
        ds = dataset[split]
    else:
        ds = dataset

    all_ids = []
    for text in ds["text"]:
        if not text or not text.strip():
            continue
        all_ids.extend(tokenizer.encode(text, add_special_tokens=False))

    if len(all_ids) == 0:
        raise ValueError("Tokenized dataset is empty.")

    return torch.tensor(all_ids, dtype=torch.long)


def sample_token_batch(token_ids: torch.Tensor, batch_size: int, seq_len: int, device: str) -> torch.Tensor:
    total = token_ids.numel()
    if total <= seq_len:
        raise ValueError("Tokenized dataset is too short for the requested seq_len.")

    max_start = total - seq_len
    starts = torch.randint(0, max_start, (batch_size,))
    batch = torch.stack([token_ids[s : s + seq_len] for s in starts], dim=0)
    return batch.to(device)
