"""
Receiver-aligned baseline inspired by:
    J. Hao et al., "Short Wins Long: Short Codes with Language Model
    Semantic Correction Outperform Long Codes," arXiv:2505.08536, 2025.

Purpose
-------
This script keeps the user's evaluation environment fixed:
  * the same COCO-caption loader (`data_coco.iter_coco_captions`),
  * the same tokenizer-to-bit mapping and AWGN/LLR channel
    (`channel_corrupt.transmit_ids_awgn_llr`),
  * the same SNR sweep, modulation, semantic cosine metric, and `.npy` format.

Only the receiver reconstruction method is changed. The hard-decoded token
sequence is converted to corrupted text and restored by a BART denoising
sequence-to-sequence model, corresponding to the SEC idea in the paper.

This is intentionally a receiver-level *aligned baseline*, not an exact
reproduction of the paper's ASCII + BCH/OSD transmitter. Exact transmitter
reproduction would change payload length and channel coding, making comparison
with the user's token-bit receiver confounded.

Modes
-----
1) Train a BART SEC checkpoint on synthetic clean/corrupted COCO pairs:
   python baseline_sec_bart.py --mode train --sec-output-dir checkpoints/sec_bart

2) Evaluate the checkpoint over SNR:
   python baseline_sec_bart.py --mode test \
       --sec-model checkpoints/sec_bart \
       --output-dir results/baselines \
       --output-name baseline_sec_bart

The saved numeric array has columns:
    [snr_db, mean_f1, mean_cosine_similarity, mean_bert_score]
This matches the user's existing result layout.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cosine as scipy_cosine
from sklearn.metrics import f1_score
from torch.optim import AdamW
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

try:
    from data_coco import iter_coco_captions
except ImportError as exc:  # pragma: no cover - project-local dependency
    iter_coco_captions = None
    _DATA_IMPORT_ERROR = exc
else:
    _DATA_IMPORT_ERROR = None

try:
    from channel_corrupt import transmit_ids_awgn_llr
except ImportError:  # permit the attachment's alternate filename
    try:
        from corrupt import transmit_ids_awgn_llr
    except ImportError as exc:  # pragma: no cover - project-local dependency
        transmit_ids_awgn_llr = None
        _CHANNEL_IMPORT_ERROR = exc
    else:
        _CHANNEL_IMPORT_ERROR = None
else:
    _CHANNEL_IMPORT_ERROR = None


DEFAULT_SNR = tuple(range(-5, 11))


@dataclass
class MetricRow:
    snr_db: float
    f1: float
    cosine: float
    bert_score: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BART semantic error-correction baseline")
    p.add_argument("--mode", choices=("train", "test"), default="test")

    # Shared user environment.
    p.add_argument("--model-name", default="google/bert_uncased_L-4_H-512_A-8")
    p.add_argument(
        "--model-revision",
        default="refs/pr/1",
        help="Revision used by the current user code; set to main/None-equivalent when using another model",
    )
    p.add_argument(
        "--dataset-name", default="HuggingFaceM4/COCO"
    )  # when-wang/coco_captions
    p.add_argument("--split", default="train")
    p.add_argument("--text-field", default="caption")
    p.add_argument(
        "--num-samples", "--test-num-samples", dest="num_samples", type=int, default=500
    )
    p.add_argument("--max-len", type=int, default=64)
    p.add_argument(
        "--modulation", choices=("bpsk", "qpsk", "16qam", "64qam"), default="bpsk"
    )
    p.add_argument("--snr-values", nargs="+", type=float, default=list(DEFAULT_SNR))
    p.add_argument("--iterations", type=int, default=1)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--test-text", "--test-texts", dest="test_text", action="append", default=[]
    )

    # BART SEC.
    p.add_argument(
        "--sec-model",
        default="facebook/bart-base",
        help="HF model name or a locally fine-tuned SEC checkpoint",
    )
    p.add_argument("--sec-output-dir", default="checkpoints/sec_bart")
    p.add_argument("--num-beams", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--length-penalty", type=float, default=1.0)

    # Training options. The paper uses SNR -2..2 dB, Adam, lr 3e-5, batch 128.
    p.add_argument("--train-snr-min", type=float, default=-2.0)
    p.add_argument("--train-snr-max", type=float, default=2.0)
    p.add_argument(
        "--train-repeats",
        type=int,
        default=2,
        help="Synthetic channel realizations generated per clean caption per epoch",
    )
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Optional differentiable structural regularizer; 0 reproduces standard seq2seq CE",
    )

    # Output / metrics.
    p.add_argument("--output-dir", default="results/comparison")
    p.add_argument("--output-name", default="baseline_sec_bart")
    p.add_argument("--save-examples-per-snr", type=int, default=2)
    p.add_argument("--compute-bert-score", action="store_true")
    return p.parse_args()


def require_project_modules() -> None:
    if iter_coco_captions is None:
        raise ImportError(
            "Could not import data_coco.iter_coco_captions. Run this file from "
            "the user's project root or provide --test-text."
        ) from _DATA_IMPORT_ERROR
    if transmit_ids_awgn_llr is None:
        raise ImportError(
            "Could not import transmit_ids_awgn_llr from channel_corrupt.py (or corrupt.py)."
        ) from _CHANNEL_IMPORT_ERROR


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_seed(base: int, snr_index: int, iteration: int, sample_index: int) -> int:
    """Identical deterministic channel realization across all baseline scripts."""
    return int(base + 1_000_003 * snr_index + 10_007 * iteration + sample_index)


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def load_texts(args: argparse.Namespace) -> list[str]:
    if args.test_text:
        return [x.strip() for x in args.test_text if x.strip()]

    require_project_modules()
    texts = list(
        iter_coco_captions(
            args.dataset_name,
            args.split,
            args.text_field,
            args.num_samples,
        )
    )
    texts = [str(x).strip() for x in texts if str(x).strip()]
    if not texts:
        raise ValueError("No evaluation texts were loaded.")

    return texts[: args.num_samples]


def collate_channel_text(
    tokenizer,
    text: str,
    max_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = tokenizer(
        [text],
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    return encoded["input_ids"].to(device), encoded["attention_mask"].to(device)


def sanitize_ids(ids: torch.Tensor, vocab_size: int, unk_id: int) -> list[int]:
    return [int(x) if 0 <= int(x) < vocab_size else int(unk_id) for x in ids.tolist()]


def channel_corrupt_text(
    text: str,
    channel_tokenizer,
    evaluator_model,
    args: argparse.Namespace,
    snr_db: float,
    device: torch.device,
) -> tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids, attention_mask = collate_channel_text(
        channel_tokenizer, text, args.max_len, device
    )
    m_bits = int(math.ceil(math.log2(evaluator_model.config.vocab_size)))
    xbar_ids, llr_tok = transmit_ids_awgn_llr(
        input_ids,
        float(snr_db),
        args.modulation,
        m_bits,
        channel_tokenizer,
    )
    active = attention_mask[0].bool()
    unk_id = channel_tokenizer.unk_token_id or 100
    corrupted_ids = sanitize_ids(
        xbar_ids[0][active], evaluator_model.config.vocab_size, unk_id
    )
    corrupted_text = channel_tokenizer.decode(
        corrupted_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    ).strip()
    if not corrupted_text:
        corrupted_text = channel_tokenizer.unk_token or "[UNK]"
    return corrupted_text, input_ids, attention_mask, llr_tok


def bart_restore(
    corrupted_texts: list[str],
    sec_tokenizer,
    sec_model,
    args: argparse.Namespace,
    device: torch.device,
) -> list[str]:
    encoded = sec_tokenizer(
        corrupted_texts,
        padding=True,
        truncation=True,
        max_length=args.max_len * 2,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        generated = sec_model.generate(
            **encoded,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            length_penalty=args.length_penalty,
            early_stopping=True,
        )
    outputs = sec_tokenizer.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    return [x.strip() for x in outputs]


def macro_token_f1(clean_text: str, restored_text: str, tokenizer) -> float:
    try:
        clean_tokens = tokenizer.tokenize(clean_text)
        restored_tokens = tokenizer.tokenize(restored_text)
        length = max(len(clean_tokens), len(restored_tokens))
        if length == 0:
            return 0.0
        clean_ids = tokenizer.convert_tokens_to_ids(clean_tokens)
        restored_ids = tokenizer.convert_tokens_to_ids(restored_tokens)
        pad_id = tokenizer.pad_token_id or 0
        clean_ids += [pad_id] * (length - len(clean_ids))
        restored_ids += [pad_id] * (length - len(restored_ids))
        return float(
            f1_score(clean_ids, restored_ids, average="macro", zero_division=0)
        )
    except Exception:
        return 0.0


def sentence_cls_embedding(
    text: str,
    tokenizer,
    model,
    device: torch.device,
) -> np.ndarray:
    encoded = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)
    with torch.no_grad():
        out = model(**encoded, output_hidden_states=True, return_dict=True)
    return out.hidden_states[-1][:, 0, :].squeeze(0).float().cpu().numpy()


def cosine_similarity_metric(
    clean_text: str,
    restored_text: str,
    tokenizer,
    model,
    device: torch.device,
) -> float:
    try:
        clean = sentence_cls_embedding(clean_text, tokenizer, model, device)
        restored = sentence_cls_embedding(restored_text, tokenizer, model, device)
        value = float(1.0 - scipy_cosine(clean, restored))
        return value if np.isfinite(value) else 0.0
    except Exception:
        return 0.0


def bert_score_metric(clean_text: str, restored_text: str, enabled: bool) -> float:
    if not enabled:
        return float("nan")
    try:
        from bert_score import score as bert_score

        _, _, f1 = bert_score([restored_text], [clean_text], lang="en", verbose=False)
        return float(f1.mean().item())
    except Exception:
        return float("nan")


def aggregate_and_save(
    rows: list[MetricRow],
    snr_values: Iterable[float],
    npy_path: Path,
) -> np.ndarray:
    grouped: list[list[float]] = []
    for snr in snr_values:
        subset = [r for r in rows if float(r.snr_db) == float(snr)]
        if not subset:
            continue
        grouped.append(
            [
                float(snr),
                float(np.mean([r.f1 for r in subset])),
                float(np.mean([r.cosine for r in subset])),
                float(np.nanmean([r.bert_score for r in subset]))
                if any(np.isfinite(r.bert_score) for r in subset)
                else float("nan"),
            ]
        )
    arr = np.asarray(grouped, dtype=np.float32)
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, arr)
    return arr


def train_sec(args: argparse.Namespace) -> None:
    require_project_modules()
    set_seed(args.seed)
    device = resolve_device(args.device)
    texts = load_texts(args)

    channel_tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    evaluator_model = AutoModelForMaskedLM.from_pretrained(
        args.model_name, revision=args.model_revision, use_safetensors=True
    ).to(device)
    evaluator_model.eval()

    sec_tokenizer = AutoTokenizer.from_pretrained(args.sec_model)
    sec_model = AutoModelForSeq2SeqLM.from_pretrained(
        args.sec_model, use_safetensors=True
    ).to(device)
    optimizer = AdamW(
        sec_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    sec_model.train()
    global_step = 0
    for epoch in range(args.epochs):
        order = list(range(len(texts)))
        random.Random(args.seed + epoch).shuffle(order)
        batch_corrupted: list[str] = []
        batch_clean: list[str] = []

        def run_batch() -> None:
            nonlocal global_step, batch_corrupted, batch_clean
            if not batch_clean:
                return
            model_inputs = sec_tokenizer(
                batch_corrupted,
                padding=True,
                truncation=True,
                max_length=args.max_len * 2,
                return_tensors="pt",
            ).to(device)
            targets = sec_tokenizer(
                text_target=batch_clean,
                padding=True,
                truncation=True,
                max_length=args.max_len * 2,
                return_tensors="pt",
            )
            labels = targets["input_ids"].to(device)
            labels = labels.masked_fill(labels == sec_tokenizer.pad_token_id, -100)
            outputs = sec_model(**model_inputs, labels=labels)
            loss = outputs.loss
            if args.label_smoothing > 0:
                # `outputs.loss` is the exact seq2seq CE objective. Label smoothing
                # is an optional differentiable stand-in for the paper's reported
                # non-differentiable edit-distance regularization.
                logits = outputs.logits.contiguous()
                target = labels.contiguous()
                smooth = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    target.view(-1),
                    ignore_index=-100,
                    label_smoothing=args.label_smoothing,
                )
                loss = 0.5 * loss + 0.5 * smooth
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sec_model.parameters(), args.grad_clip)
            optimizer.step()
            global_step += 1
            if global_step % 25 == 0:
                print(f"epoch={epoch + 1} step={global_step} loss={loss.item():.4f}")
            batch_corrupted = []
            batch_clean = []

        for repeat in range(args.train_repeats):
            for rank, idx in enumerate(order):
                clean_text = texts[idx]
                local_seed = sample_seed(args.seed, epoch, repeat, rank)
                set_seed(local_seed)
                snr = random.uniform(args.train_snr_min, args.train_snr_max)
                corrupted_text, _, _, _ = channel_corrupt_text(
                    clean_text,
                    channel_tokenizer,
                    evaluator_model,
                    args,
                    snr,
                    device,
                )
                batch_corrupted.append(corrupted_text)
                batch_clean.append(clean_text)
                if len(batch_clean) >= args.batch_size:
                    run_batch()
        run_batch()

        epoch_dir = Path(args.sec_output_dir) / f"epoch_{epoch + 1}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        sec_model.save_pretrained(epoch_dir, safe_serialization=True)
        sec_tokenizer.save_pretrained(epoch_dir)
        print(f"Saved SEC checkpoint: {epoch_dir}")

    final_dir = Path(args.sec_output_dir)
    final_dir.mkdir(parents=True, exist_ok=True)
    sec_model.save_pretrained(final_dir, safe_serialization=True)
    sec_tokenizer.save_pretrained(final_dir)
    print(f"Saved final SEC checkpoint: {final_dir}")


def test_sec(args: argparse.Namespace) -> None:
    require_project_modules()
    set_seed(args.seed)
    device = resolve_device(args.device)
    texts = load_texts(args)

    channel_tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    evaluator_model = AutoModelForMaskedLM.from_pretrained(
        args.model_name, revision=args.model_revision, use_safetensors=True
    ).to(device)
    evaluator_model.eval()

    sec_tokenizer = AutoTokenizer.from_pretrained(args.sec_model)
    sec_model = AutoModelForSeq2SeqLM.from_pretrained(
        args.sec_model, use_safetensors=True
    ).to(device)
    sec_model.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    npy_path = output_dir / f"{args.output_name}.npy"
    csv_path = output_dir / f"{args.output_name}.csv"

    rows: list[MetricRow] = []
    examples: list[dict[str, object]] = []
    for snr_index, snr in enumerate(args.snr_values):
        print(f"\n=== SEC-BART | SNR {snr:g} dB ===")
        for iteration in range(args.iterations):
            for sample_index, clean_text in enumerate(texts):
                set_seed(sample_seed(args.seed, snr_index, iteration, sample_index))
                corrupted_text, _, _, _ = channel_corrupt_text(
                    clean_text,
                    channel_tokenizer,
                    evaluator_model,
                    args,
                    snr,
                    device,
                )
                restored_text = bart_restore(
                    [corrupted_text], sec_tokenizer, sec_model, args, device
                )[0]

                f1 = macro_token_f1(clean_text, restored_text, channel_tokenizer)
                cosine = cosine_similarity_metric(
                    clean_text,
                    restored_text,
                    channel_tokenizer,
                    evaluator_model,
                    device,
                )
                bert = bert_score_metric(
                    clean_text, restored_text, args.compute_bert_score
                )
                rows.append(MetricRow(float(snr), f1, cosine, bert))

                already = sum(1 for x in examples if float(x["snr_db"]) == float(snr))
                if already < args.save_examples_per_snr:
                    examples.append(
                        {
                            "snr_db": float(snr),
                            "iteration": iteration,
                            "sample_idx": sample_index,
                            "clean_text": clean_text,
                            "corrupted_text": corrupted_text,
                            "restored_text": restored_text,
                            "f1": f1,
                            "cosine_similarity": cosine,
                            "bert_score": bert,
                        }
                    )
            arr = aggregate_and_save(rows, args.snr_values, npy_path)
            print(
                f"iteration {iteration + 1}/{args.iterations}: "
                f"mean cosine={arr[-1, 2]:.4f}"
            )

    aggregate_and_save(rows, args.snr_values, npy_path)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "snr_db",
            "iteration",
            "sample_idx",
            "clean_text",
            "corrupted_text",
            "restored_text",
            "f1",
            "cosine_similarity",
            "bert_score",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(examples)

    metadata = {
        "baseline": "SEC-BART aligned receiver baseline",
        "paper": "arXiv:2505.08536",
        "sec_model": args.sec_model,
        "channel_model": args.model_name,
        "modulation": args.modulation,
        "snr_values": args.snr_values,
        "num_samples": len(texts),
        "iterations": args.iterations,
        "npy_columns": ["snr_db", "mean_f1", "mean_cosine", "mean_bert_score"],
    }
    (output_dir / f"{args.output_name}.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved: {npy_path}")
    print(f"Saved: {csv_path}")


def main() -> None:
    args = parse_args()
    if args.mode == "train":
        train_sec(args)
    else:
        test_sec(args)


if __name__ == "__main__":
    main()
