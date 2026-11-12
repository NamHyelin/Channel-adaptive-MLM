"""
Autoregressive LLM-SC baseline inspired by:
    Z. Wang et al., "Large Language Model Enabled Semantic Communication
    Systems," arXiv:2407.14112, 2024.

Core receiver rule
------------------
At token position i, retain the highest-scoring beam paths under

    cumulative_score
      += language_weight * log P_LM(token_i | decoded_prefix)
       + channel_weight  * log P_channel(observation_i | token_i)

The default decoder uses the parameter-matched autoregressive model
EleutherAI/pythia-31m and evaluates the full beam x valid-tokenizer-vocabulary
score matrix before retaining the global top-K paths. Pythia pads its model
output vocabulary to 50,304 entries while its tokenizer has 50,277 usable
entries; the padded output-only entries are explicitly excluded.

Important scope
---------------
This is an autoregressive implementation of the paper's decoding principle,
not an exact reproduction of its Vicuna-7B tokenizer or high-dimensional
8-QAM/16-QAM setup. It keeps the user's existing token-ID-to-bit channel
interface and COCO evaluation pipeline. A separate BERT encoder is used only
for cosine evaluation; it does not participate in decoding.

Example
-------
    python baseline_llmsc_pythia31m.py \
        --beam-size 15 \
        --modulation bpsk \
        --num-samples 500 \
        --output-dir results/baselines \
        --output-name baseline_llmsc_pythia31m

For a quick smoke test:
    python baseline_llmsc_pythia31m.py \
        --test-text "A man is riding a bicycle." \
        --snr-values 0 5 \
        --beam-size 4

Saved .npy columns
------------------
    [snr_db, mean_f1, mean_cosine_similarity, mean_bert_score]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cosine as scipy_cosine
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

try:
    from data_coco import iter_coco_captions
except ImportError as exc:  # pragma: no cover
    iter_coco_captions = None
    _DATA_IMPORT_ERROR = exc
else:
    _DATA_IMPORT_ERROR = None

try:
    from channel_corrupt import transmit_ids_awgn_llr
except ImportError as exc:  # pragma: no cover
    transmit_ids_awgn_llr = None
    _CHANNEL_IMPORT_ERROR = exc
else:
    _CHANNEL_IMPORT_ERROR = None


DEFAULT_SNR = tuple(range(-5, 11))


@dataclass
class CausalBeam:
    ids: torch.Tensor
    score: float


@dataclass
class MetricRow:
    snr_db: float
    f1: float
    cosine: float
    bert_score: float


@dataclass
class SampleResult:
    snr_db: float
    iteration: int
    sample_idx: int
    clean_text: str
    corrupted_text: str
    restored_text: str
    f1: float
    cosine_similarity: float
    bert_score: float
    exact_token_match: int
    hard_token_error_rate: float
    restored_token_error_rate: float
    diagnostics: str
    error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Autoregressive LLM-SC MAP beam-search baseline"
    )

    # Causal language model used by the communication receiver.
    parser.add_argument(
        "--model-name",
        default="EleutherAI/pythia-31m",
        help="Causal language model and tokenizer used for transmission/decoding.",
    )
    parser.add_argument(
        "--model-revision",
        default="main",
        help="Hugging Face revision for the causal model.",
    )

    # Separate metric model. refs/pr/1 supplies safetensors for this BERT model.
    parser.add_argument(
        "--metric-model-name",
        default="google/bert_uncased_L-4_H-512_A-8",
    )
    parser.add_argument(
        "--metric-model-revision",
        default="refs/pr/1",
    )

    parser.add_argument("--dataset-name", default="HuggingFaceM4/COCO")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-field", default="caption")
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--max-len", type=int, default=16)
    parser.add_argument(
        "--modulation",
        choices=("bpsk", "qpsk", "16qam", "64qam"),
        default="bpsk",
    )
    parser.add_argument(
        "--snr-values", nargs="+", type=float, default=list(DEFAULT_SNR)
    )
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--test-text", action="append", default=[])

    # The paper reports gains saturating around beam widths 10--20.
    parser.add_argument("--beam-size", type=int, default=2)
    parser.add_argument("--language-weight", type=float, default=0.5)
    parser.add_argument("--channel-weight", type=float, default=0.5)

    # Full is the default and evaluates all K x V combinations. Pruned mode is
    # provided only as a speed/memory fallback and is not the default baseline.
    parser.add_argument(
        "--candidate-mode",
        choices=("full", "pruned"),
        default="full",
        help="full: evaluate all beam x vocabulary paths; pruned: use LM/channel top-k union.",
    )
    parser.add_argument("--lm-topk", type=int, default=2)
    parser.add_argument("--channel-topk", type=int, default=2)
    parser.add_argument(
        "--keep-hard-token",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include hard-demodulated token in pruned candidate mode.",
    )

    parser.add_argument("--output-dir", default="results/comparison")
    parser.add_argument("--output-name", default="baseline_llmsc_pythia31m")
    parser.add_argument("--save-examples-per-snr", type=int, default=2)
    parser.add_argument("--compute-bert-score", action="store_true")
    parser.add_argument(
        "--save-sample-csv",
        action="store_true",
        help="Save every sample immediately to CSV. NPY saving is independent.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every N samples. Use 0 to disable progress output.",
    )
    parser.add_argument(
        "--show-progress-text",
        action="store_true",
        help="Also print clean/restored text in progress messages.",
    )
    return parser.parse_args()


def require_project_modules(need_dataset: bool = True) -> None:
    if need_dataset and iter_coco_captions is None:
        raise ImportError(
            "Could not import data_coco.iter_coco_captions. "
            "Run this script from the project root."
        ) from _DATA_IMPORT_ERROR
    if transmit_ids_awgn_llr is None:
        raise ImportError(
            "Could not import transmit_ids_awgn_llr from channel_corrupt.py."
        ) from _CHANNEL_IMPORT_ERROR


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_seed(base: int, snr_index: int, iteration: int, sample_index: int) -> int:
    return int(base + 1_000_003 * snr_index + 10_007 * iteration + sample_index)


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("[warning] CUDA is unavailable; using CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def load_texts(args: argparse.Namespace) -> list[str]:
    if args.test_text:
        texts = [text.strip() for text in args.test_text if text.strip()]
        if not texts:
            raise ValueError("--test-text was given, but no non-empty text was found.")
        return texts

    require_project_modules(need_dataset=True)
    texts = list(
        iter_coco_captions(
            args.dataset_name,
            args.split,
            args.text_field,
            args.num_samples,
        )
    )
    texts = [str(text).strip() for text in texts if str(text).strip()]
    if not texts:
        raise ValueError("No evaluation texts were loaded.")
    return texts[: args.num_samples]


def load_causal_model(args: argparse.Namespace, device: torch.device):
    """Load a causal LM while allowing padded model-output vocabularies.

    Pythia-31M has 50,277 tokenizer entries but a 50,304-row embedding/output
    matrix. The extra 27 rows are implementation padding, not usable tokens.
    """
    print(f"Loading causal decoder model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        revision=args.model_revision,
    )

    # Keep the vocabulary unchanged. Only reuse EOS as PAD when a causal
    # tokenizer truly lacks a PAD token.
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("The causal tokenizer has neither PAD nor EOS token.")
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        revision=args.model_revision,
        use_safetensors=True,
    ).to(device)
    model.eval()

    tokenizer_vocab_size = len(tokenizer)
    model_output_vocab_size = int(model.config.vocab_size)
    if tokenizer_vocab_size > model_output_vocab_size:
        raise ValueError(
            "Tokenizer vocabulary is larger than the model output vocabulary: "
            f"{tokenizer_vocab_size} > {model_output_vocab_size}."
        )

    print(
        "[vocabulary] "
        f"usable tokenizer tokens={tokenizer_vocab_size}, "
        f"model output rows={model_output_vocab_size}, "
        f"excluded padded outputs={model_output_vocab_size - tokenizer_vocab_size}"
    )
    return tokenizer, model


def load_metric_model(args: argparse.Namespace, device: torch.device):
    print(f"Loading metric model: {args.metric_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.metric_model_name)
    model = AutoModel.from_pretrained(
        args.metric_model_name,
        revision=args.metric_model_revision,
        use_safetensors=True,
    ).to(device)
    model.eval()
    return tokenizer, model


def tokenize_text_causal(
    tokenizer,
    text: str,
    max_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    if input_ids.numel() == 0:
        raise ValueError(f"Tokenizer produced an empty sequence for text: {text!r}")
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    return input_ids, attention_mask


def sanitize_tensor(ids: torch.Tensor, vocab_size: int, fallback_id: int) -> torch.Tensor:
    output = ids.clone().long()
    invalid = (output < 0) | (output >= vocab_size)
    output[invalid] = int(fallback_id)
    return output


def ids_to_bits_table(
    vocab_size: int,
    bits_per_token: int,
    device: torch.device,
) -> torch.Tensor:
    """Return little-endian token-ID bit patterns with shape [V, m]."""
    token_ids = torch.arange(vocab_size, device=device, dtype=torch.long)
    shifts = torch.arange(bits_per_token, device=device, dtype=torch.long)
    return ((token_ids.unsqueeze(1) >> shifts.unsqueeze(0)) & 1).bool()


def token_channel_log_likelihoods(
    llr_bits: torch.Tensor,
    vocab_bits: torch.Tensor,
) -> torch.Tensor:
    """
    Compute exact independent-bit token log likelihoods from bit LLRs.

    This code follows the pre-existing channel interface convention

        LLR = log P(bit=1 | y) / P(bit=0 | y).

    Therefore
        log P(bit=1 | y) = -softplus(-LLR)
        log P(bit=0 | y) = -softplus( LLR).
    """
    if llr_bits.ndim != 1:
        raise ValueError(f"Expected one token's LLRs [m], got {tuple(llr_bits.shape)}")
    if llr_bits.numel() != vocab_bits.size(1):
        raise ValueError(
            "LLR width does not match token bit width: "
            f"{llr_bits.numel()} vs {vocab_bits.size(1)}"
        )

    log_p1 = -F.softplus(-llr_bits.float())
    log_p0 = -F.softplus(llr_bits.float())
    return torch.where(
        vocab_bits,
        log_p1.unsqueeze(0),
        log_p0.unsqueeze(0),
    ).sum(dim=1)


def _causal_prefix_batch(
    beams: list[CausalBeam],
    start_token_id: int,
    device: torch.device,
) -> torch.Tensor:
    """Stack [start token + decoded prefix] for every current beam."""
    start = torch.tensor([start_token_id], dtype=torch.long, device=device)
    prefixes = [torch.cat((start, beam.ids), dim=0) for beam in beams]
    return torch.stack(prefixes, dim=0)


def _full_vocabulary_step(
    beams: list[CausalBeam],
    lm_log_probs: torch.Tensor,
    channel_scores: torch.Tensor,
    beam_size: int,
    language_weight: float,
    channel_weight: float,
) -> list[CausalBeam]:
    """Evaluate the full beam x vocabulary MAP score matrix."""
    device = lm_log_probs.device
    vocab_size = lm_log_probs.size(1)
    previous_scores = torch.tensor(
        [beam.score for beam in beams],
        dtype=torch.float32,
        device=device,
    )

    total_scores = (
        previous_scores[:, None]
        + language_weight * lm_log_probs
        + channel_weight * channel_scores[None, :]
    )

    keep = min(beam_size, total_scores.numel())
    values, flat_indices = torch.topk(total_scores.reshape(-1), k=keep)
    parent_indices = torch.div(flat_indices, vocab_size, rounding_mode="floor")
    token_indices = flat_indices.remainder(vocab_size)

    next_beams: list[CausalBeam] = []
    for rank in range(keep):
        parent_index = int(parent_indices[rank].item())
        token_id = int(token_indices[rank].item())
        next_token = torch.tensor([token_id], dtype=torch.long, device=device)
        next_beams.append(
            CausalBeam(
                ids=torch.cat((beams[parent_index].ids, next_token), dim=0),
                score=float(values[rank].item()),
            )
        )
    return next_beams


def _pruned_candidate_step(
    beams: list[CausalBeam],
    lm_log_probs: torch.Tensor,
    channel_scores: torch.Tensor,
    hard_token: int,
    args: argparse.Namespace,
) -> list[CausalBeam]:
    """Optional speed fallback using LM/channel top-k candidate unions."""
    vocab_size = lm_log_probs.size(1)
    k_lm = min(args.lm_topk, vocab_size)
    k_channel = min(args.channel_topk, vocab_size)
    channel_candidates = torch.topk(channel_scores, k=k_channel).indices.tolist()

    expansions: list[CausalBeam] = []
    for beam_index, beam in enumerate(beams):
        lm_candidates = torch.topk(
            lm_log_probs[beam_index], k=k_lm
        ).indices.tolist()
        candidate_ids = set(int(token_id) for token_id in lm_candidates)
        candidate_ids.update(int(token_id) for token_id in channel_candidates)
        if args.keep_hard_token and 0 <= hard_token < vocab_size:
            candidate_ids.add(int(hard_token))

        for token_id in candidate_ids:
            score_increment = (
                args.language_weight
                * float(lm_log_probs[beam_index, token_id].item())
                + args.channel_weight * float(channel_scores[token_id].item())
            )
            next_token = torch.tensor(
                [token_id], dtype=torch.long, device=lm_log_probs.device
            )
            expansions.append(
                CausalBeam(
                    ids=torch.cat((beam.ids, next_token), dim=0),
                    score=beam.score + score_increment,
                )
            )

    expansions.sort(key=lambda beam: beam.score, reverse=True)
    return expansions[: args.beam_size]


def llmsc_causal_beam_decode(
    hard_ids: torch.Tensor,
    llr_tok: torch.Tensor,
    tokenizer,
    model,
    vocab_bits: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, object]]:
    """
    Autoregressive LLM-SC MAP beam decoder.

    The transmitted token count is assumed to be known from packet framing.
    Only previously decoded token candidates are supplied to the causal LM.
    """
    if hard_ids.ndim != 1:
        raise ValueError(f"Expected hard_ids [T], got {tuple(hard_ids.shape)}")
    if llr_tok.ndim != 2:
        raise ValueError(f"Expected llr_tok [T,m], got {tuple(llr_tok.shape)}")
    if hard_ids.size(0) != llr_tok.size(0):
        raise ValueError(
            f"Token/LLR length mismatch: {hard_ids.size(0)} vs {llr_tok.size(0)}"
        )

    device = llr_tok.device
    # Use only token IDs that the tokenizer can encode/decode. Pythia's model
    # output dimension is padded beyond this valid vocabulary size.
    vocab_size = len(tokenizer)
    model_output_vocab_size = int(model.config.vocab_size)
    if vocab_size > model_output_vocab_size:
        raise ValueError(
            f"Tokenizer vocabulary {vocab_size} exceeds model output size "
            f"{model_output_vocab_size}."
        )
    if vocab_bits.size(0) != vocab_size:
        raise ValueError(
            f"vocab_bits/tokenizer mismatch: {vocab_bits.size(0)} vs {vocab_size}."
        )

    fallback_id = tokenizer.unk_token_id
    if fallback_id is None:
        fallback_id = tokenizer.eos_token_id
    if fallback_id is None:
        fallback_id = tokenizer.pad_token_id
    if fallback_id is None or not (0 <= int(fallback_id) < vocab_size):
        fallback_id = 0
    hard_valid = sanitize_tensor(hard_ids, vocab_size, int(fallback_id))

    start_token_id = tokenizer.bos_token_id
    if start_token_id is None:
        start_token_id = tokenizer.eos_token_id
    if start_token_id is None:
        start_token_id = tokenizer.pad_token_id
    if start_token_id is None:
        start_token_id = 0
    if not (0 <= int(start_token_id) < model_output_vocab_size):
        raise ValueError(
            f"Invalid start token ID {start_token_id} for model output size "
            f"{model_output_vocab_size}."
        )

    beams = [
        CausalBeam(
            ids=torch.empty(0, dtype=torch.long, device=device),
            score=0.0,
        )
    ]

    with torch.inference_mode():
        for position in range(llr_tok.size(0)):
            prefix_batch = _causal_prefix_batch(beams, int(start_token_id), device)
            attention_mask = torch.ones_like(prefix_batch, dtype=torch.long)
            outputs = model(
                input_ids=prefix_batch,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            # The model emits 50,304 logits for Pythia-31M, but only the
            # first len(tokenizer)=50,277 correspond to real tokenizer IDs.
            lm_logits = outputs.logits[:, -1, :vocab_size].float()
            lm_log_probs = torch.log_softmax(lm_logits, dim=-1)
            channel_scores = token_channel_log_likelihoods(
                llr_tok[position],
                vocab_bits,
            )

            if args.candidate_mode == "full":
                beams = _full_vocabulary_step(
                    beams=beams,
                    lm_log_probs=lm_log_probs,
                    channel_scores=channel_scores,
                    beam_size=args.beam_size,
                    language_weight=args.language_weight,
                    channel_weight=args.channel_weight,
                )
            else:
                beams = _pruned_candidate_step(
                    beams=beams,
                    lm_log_probs=lm_log_probs,
                    channel_scores=channel_scores,
                    hard_token=int(hard_valid[position].item()),
                    args=args,
                )

            if not beams:
                raise RuntimeError(
                    f"Beam search produced no candidates at position {position}."
                )

    best = max(beams, key=lambda beam: beam.score)
    diagnostics = {
        "model_type": "causal_lm",
        "candidate_mode": args.candidate_mode,
        "beam_size": args.beam_size,
        "decoded_positions": int(llr_tok.size(0)),
        "valid_tokenizer_vocab_size": vocab_size,
        "model_output_vocab_size": model_output_vocab_size,
        "excluded_padded_output_rows": model_output_vocab_size - vocab_size,
        "bits_per_token": int(vocab_bits.size(1)),
        "final_score": best.score,
        "runner_up_gap": (
            best.score - beams[1].score if len(beams) > 1 else float("nan")
        ),
        "language_weight": args.language_weight,
        "channel_weight": args.channel_weight,
    }
    return best.ids, diagnostics


def decode_token_ids(ids: torch.Tensor, tokenizer) -> str:
    return tokenizer.decode(
        ids.tolist(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    ).strip()


def macro_token_f1(clean_text: str, restored_text: str, tokenizer) -> float:
    """Position-aligned multiclass macro F1 without sklearn warnings.

    This is numerically equivalent to sklearn.metrics.f1_score(...,
    average="macro", zero_division=0) for the padded token-ID sequences,
    but avoids sklearn's regression-target heuristic warning when most token
    IDs occur only once.
    """
    try:
        clean_tokens = tokenizer.tokenize(clean_text)
        restored_tokens = tokenizer.tokenize(restored_text)
        length = max(len(clean_tokens), len(restored_tokens))
        if length == 0:
            return 0.0

        clean_ids = list(tokenizer.convert_tokens_to_ids(clean_tokens))
        restored_ids = list(tokenizer.convert_tokens_to_ids(restored_tokens))
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id or 0
        clean_ids += [pad_id] * (length - len(clean_ids))
        restored_ids += [pad_id] * (length - len(restored_ids))

        true = np.asarray(clean_ids, dtype=np.int64)
        pred = np.asarray(restored_ids, dtype=np.int64)
        labels = np.union1d(true, pred)
        if labels.size == 0:
            return 0.0

        per_label_f1: list[float] = []
        for label in labels:
            true_positive = int(np.sum((true == label) & (pred == label)))
            false_positive = int(np.sum((true != label) & (pred == label)))
            false_negative = int(np.sum((true == label) & (pred != label)))
            denominator = 2 * true_positive + false_positive + false_negative
            per_label_f1.append(
                0.0 if denominator == 0 else (2.0 * true_positive / denominator)
            )

        return float(np.mean(per_label_f1))
    except Exception:
        return 0.0


def cls_embedding(
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
    with torch.inference_mode():
        outputs = model(**encoded, return_dict=True)
    hidden = outputs.last_hidden_state
    return hidden[:, 0, :].squeeze(0).float().cpu().numpy()


def cosine_metric(
    clean: str,
    restored: str,
    tokenizer,
    model,
    device: torch.device,
) -> float:
    if not restored.strip():
        return 0.0
    try:
        clean_embedding = cls_embedding(clean, tokenizer, model, device)
        restored_embedding = cls_embedding(restored, tokenizer, model, device)
        value = float(1.0 - scipy_cosine(clean_embedding, restored_embedding))
        return value if np.isfinite(value) else 0.0
    except Exception:
        return 0.0


def bert_score_metric(clean: str, restored: str, enabled: bool) -> float:
    if not enabled:
        return float("nan")
    try:
        from bert_score import score as bert_score

        _, _, f1 = bert_score([restored], [clean], lang="en", verbose=False)
        return float(f1.mean().item())
    except Exception:
        return float("nan")


def _csv_safe_value(value):
    """Convert values into CSV-safe Python scalars/strings."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            value = value.detach().cpu().item()
        else:
            value = value.detach().cpu().tolist()
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        value = json.dumps(value, ensure_ascii=False)
    if isinstance(value, str):
        # Keep one physical CSV record per sample.
        value = value.replace("\r\n", "\\n").replace("\n", "\\n").replace("\r", "\\n")
    return value


def sample_result_to_csv_dict(row: SampleResult) -> dict[str, object]:
    return {
        field.name: _csv_safe_value(getattr(row, field.name))
        for field in fields(SampleResult)
    }


def _make_csv_writer(file, fieldnames: list[str]) -> csv.DictWriter:
    return csv.DictWriter(
        file,
        fieldnames=fieldnames,
        extrasaction="ignore",
        quoting=csv.QUOTE_MINIMAL,
        quotechar='"',
        doublequote=True,
        escapechar="\\",
        lineterminator="\n",
    )


def append_sample_result(path: Path, row: SampleResult) -> None:
    """Persist one sample without allowing commas/quotes to break the run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    fieldnames = [field.name for field in fields(SampleResult)]
    with path.open("a", newline="", encoding="utf-8-sig") as file:
        writer = _make_csv_writer(file, fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(sample_result_to_csv_dict(row))

def token_error_rate(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    reference = reference.detach().reshape(-1).long()
    estimate = estimate.detach().reshape(-1).long()
    length = max(reference.numel(), estimate.numel())
    if length == 0:
        return 0.0
    mismatches = abs(reference.numel() - estimate.numel())
    overlap = min(reference.numel(), estimate.numel())
    if overlap:
        mismatches += int((reference[:overlap] != estimate[:overlap]).sum().item())
    return float(mismatches / length)


def aggregate_and_save(
    rows: list[MetricRow],
    snrs: Iterable[float],
    path: Path,
) -> np.ndarray:
    grouped: list[list[float]] = []
    for snr in snrs:
        subset = [row for row in rows if float(row.snr_db) == float(snr)]
        if not subset:
            continue
        bert_values = [row.bert_score for row in subset]
        grouped.append(
            [
                float(snr),
                float(np.mean([row.f1 for row in subset])),
                float(np.mean([row.cosine for row in subset])),
                (
                    float(np.nanmean(bert_values))
                    if any(np.isfinite(value) for value in bert_values)
                    else float("nan")
                ),
            ]
        )

    result = np.asarray(grouped, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, result)
    return result


def validate_args(args: argparse.Namespace) -> None:
    if args.progress_every < 0:
        raise ValueError("--progress-every must be zero or a positive integer")
    if args.beam_size < 1:
        raise ValueError("--beam-size must be positive.")
    if args.max_len < 1:
        raise ValueError("--max-len must be positive.")
    if args.num_samples < 1 and not args.test_text:
        raise ValueError("--num-samples must be positive.")
    if args.candidate_mode == "pruned":
        if args.lm_topk < 1 or args.channel_topk < 1:
            raise ValueError("--lm-topk and --channel-topk must be positive.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    require_project_modules(need_dataset=not bool(args.test_text))
    set_seed(args.seed)

    device = resolve_device(args.device)
    texts = load_texts(args)
    tokenizer, causal_model = load_causal_model(args, device)
    metric_tokenizer, metric_model = load_metric_model(args, device)

    # Communication vocabulary: actual tokenizer IDs only. Do not use the
    # padded 50,304-row Pythia output dimension as token IDs.
    vocab_size = len(tokenizer)
    model_output_vocab_size = int(causal_model.config.vocab_size)
    bits_per_token = int(math.ceil(math.log2(vocab_size)))
    vocab_bits = ids_to_bits_table(vocab_size, bits_per_token, device)

    print(
        "[configuration] "
        f"valid_vocab={vocab_size}, model_output_vocab={model_output_vocab_size}, "
        f"excluded_outputs={model_output_vocab_size - vocab_size}, "
        f"bits/token={bits_per_token}, "
        f"beam={args.beam_size}, candidate_mode={args.candidate_mode}, "
        f"modulation={args.modulation}, device={device}"
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    npy_path = output_dir / f"{args.output_name}.npy"
    csv_path = output_dir / f"{args.output_name}.csv"
    examples_path = output_dir / f"{args.output_name}_examples.csv"

    # Sample CSV is optional and independent of NPY aggregation.
    if args.save_sample_csv and csv_path.exists():
        csv_path.unlink()

    metric_rows: list[MetricRow] = []
    sample_results: list[SampleResult] = []
    saved_examples: list[SampleResult] = []

    total_jobs = len(args.snr_values) * args.iterations * len(texts)
    completed_jobs = 0
    experiment_start = time.perf_counter()

    for snr_index, snr in enumerate(args.snr_values):
        print(f"\n=== LLM-SC-AR | SNR {snr:g} dB ===")
        for iteration in range(args.iterations):
            iteration_cosines: list[float] = []
            iteration_exact_matches: list[int] = []
            iteration_hard_ters: list[float] = []
            iteration_restored_ters: list[float] = []
            for sample_index, clean_text in enumerate(texts):
                job_number = completed_jobs + 1
                should_report = (
                    args.progress_every > 0
                    and (
                        sample_index % args.progress_every == 1
                        or job_number == total_jobs
                    )
                )
                sample_start = time.perf_counter()
                if should_report:
                    print(
                        f"[start] total {job_number}/{total_jobs} | "
                        f"SNR {snr:g} dB ({snr_index + 1}/{len(args.snr_values)}) | "
                        f"iteration {iteration + 1}/{args.iterations} | "
                        f"sample {sample_index + 1}/{len(texts)}",
                        flush=True,
                    )
                    if args.show_progress_text:
                        print(f"        clean: {clean_text}", flush=True)

                set_seed(
                    sample_seed(
                        args.seed,
                        snr_index,
                        iteration,
                        sample_index,
                    )
                )

                try:
                    clean_batch, attention_batch = tokenize_text_causal(
                        tokenizer,
                        clean_text,
                        args.max_len,
                        device,
                    )
                    hard_batch, llr_batch = transmit_ids_awgn_llr(
                        clean_batch,
                        float(snr),
                        args.modulation,
                        bits_per_token,
                        tokenizer,
                    )

                    active_length = int(attention_batch[0].sum().item())
                    hard_ids = hard_batch[0, :active_length]
                    llr_tok = llr_batch[0, :active_length, :]
                    fallback_id = tokenizer.unk_token_id
                    if fallback_id is None:
                        fallback_id = tokenizer.eos_token_id or 0
                    hard_valid = sanitize_tensor(
                        hard_ids,
                        vocab_size,
                        int(fallback_id),
                    )
                    corrupted_text = decode_token_ids(hard_valid, tokenizer)

                    restored_ids, diagnostics = llmsc_causal_beam_decode(
                        hard_ids=hard_ids,
                        llr_tok=llr_tok,
                        tokenizer=tokenizer,
                        model=causal_model,
                        vocab_bits=vocab_bits,
                        args=args,
                    )
                    restored_text = decode_token_ids(restored_ids, tokenizer)

                    clean_active_ids = clean_batch[0, :active_length]
                    hard_token_error_rate = token_error_rate(
                        clean_active_ids, hard_valid
                    )
                    restored_token_error_rate = token_error_rate(
                        clean_active_ids, restored_ids
                    )
                    exact_token_match = int(
                        clean_active_ids.numel() == restored_ids.numel()
                        and torch.equal(clean_active_ids, restored_ids)
                    )

                    f1 = macro_token_f1(clean_text, restored_text, tokenizer)
                    cosine = cosine_metric(
                        clean_text,
                        restored_text,
                        metric_tokenizer,
                        metric_model,
                        device,
                    )
                    bert = bert_score_metric(
                        clean_text,
                        restored_text,
                        args.compute_bert_score,
                    )
                    error = ""
                except Exception as exc:  # Keep SNR aggregation reproducible.
                    corrupted_text = ""
                    restored_text = ""
                    diagnostics = {"failed": True}
                    f1 = 0.0
                    cosine = 0.0
                    bert = float("nan")
                    exact_token_match = 0
                    hard_token_error_rate = float("nan")
                    restored_token_error_rate = 1.0
                    error = f"{type(exc).__name__}: {exc}"
                    print(
                        f"[failed] SNR={snr:g}, iteration={iteration}, "
                        f"sample={sample_index}: {error}"
                    )

                metric_rows.append(
                    MetricRow(
                        snr_db=float(snr),
                        f1=float(f1),
                        cosine=float(cosine),
                        bert_score=float(bert),
                    )
                )
                iteration_cosines.append(float(cosine))
                iteration_exact_matches.append(int(exact_token_match))
                iteration_hard_ters.append(float(hard_token_error_rate))
                iteration_restored_ters.append(float(restored_token_error_rate))

                result = SampleResult(
                    snr_db=float(snr),
                    iteration=iteration,
                    sample_idx=sample_index,
                    clean_text=clean_text,
                    corrupted_text=corrupted_text,
                    restored_text=restored_text,
                    f1=float(f1),
                    cosine_similarity=float(cosine),
                    bert_score=float(bert),
                    exact_token_match=int(exact_token_match),
                    hard_token_error_rate=float(hard_token_error_rate),
                    restored_token_error_rate=float(restored_token_error_rate),
                    diagnostics=json.dumps(diagnostics, ensure_ascii=False),
                    error=error,
                )
                sample_results.append(result)
                if args.save_sample_csv:
                    try:
                        append_sample_result(csv_path, result)
                    except csv.Error as exc:
                        print(f"[CSV warning] sample CSV write failed: {exc}")

                example_count = sum(
                    1
                    for saved in saved_examples
                    if float(saved.snr_db) == float(snr)
                )
                if example_count < args.save_examples_per_snr:
                    saved_examples.append(result)

                completed_jobs += 1
                if should_report:
                    sample_elapsed = time.perf_counter() - sample_start
                    total_elapsed = time.perf_counter() - experiment_start
                    average_elapsed = total_elapsed / max(1, completed_jobs)
                    remaining_seconds = average_elapsed * (total_jobs - completed_jobs)
                    status = "failed" if error else "done"
                    print(
                        f"[{status}] total {completed_jobs}/{total_jobs} | "
                        f"sample time {sample_elapsed:.1f}s | "
                        f"cosine {cosine:.6f} | exact {exact_token_match} | "
                        f"hard TER {hard_token_error_rate:.4f} | "
                        f"restored TER {restored_token_error_rate:.4f} | "
                        f"estimated remaining {remaining_seconds / 60.0:.1f} min",
                        flush=True,
                    )
                    if args.show_progress_text:
                        print(f"        restored: {restored_text}", flush=True)

            aggregate_and_save(metric_rows, args.snr_values, npy_path)
            mean_cosine = float(np.mean(iteration_cosines))
            exact_rate = float(np.mean(iteration_exact_matches))
            mean_hard_ter = float(np.nanmean(iteration_hard_ters))
            mean_restored_ter = float(np.nanmean(iteration_restored_ters))
            print(
                f"SNR {snr:g} dB | iteration {iteration + 1}/{args.iterations}: "
                f"mean cosine={mean_cosine:.6f}, "
                f"exact-token rate={exact_rate:.4f}, "
                f"hard TER={mean_hard_ter:.4f}, "
                f"restored TER={mean_restored_ter:.4f}"
            )
            if mean_cosine >= 0.9999 and exact_rate < 0.95:
                print(
                    "[metric warning] Cosine is near 1 although token-exact "
                    "recovery is not. Raw BERT [CLS] cosine can be highly "
                    "anisotropic; compare exact-token rate/TER and consider "
                    "mean pooling consistently across all methods."
                )

    final_array = aggregate_and_save(metric_rows, args.snr_values, npy_path)

    fieldnames = [field.name for field in fields(SampleResult)]
    if saved_examples:
        with examples_path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = _make_csv_writer(file, fieldnames)
            writer.writeheader()
            for row_index, row in enumerate(saved_examples):
                try:
                    writer.writerow(sample_result_to_csv_dict(row))
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to write example CSV row {row_index}: {exc}"
                    ) from exc

    metadata = {
        "baseline": "LLM-SC-AR Pythia-31M causal MAP beam-search baseline",
        "paper": "arXiv:2407.14112",
        "scope": (
            "Autoregressive decoding-rule implementation using Pythia-31M; "
            "not an exact Vicuna/high-dimensional-constellation reproduction."
        ),
        "causal_model_name": args.model_name,
        "causal_model_revision": args.model_revision,
        "metric_model_name": args.metric_model_name,
        "metric_model_revision": args.metric_model_revision,
        "valid_tokenizer_vocab_size": vocab_size,
        "model_output_vocab_size": model_output_vocab_size,
        "excluded_padded_output_rows": model_output_vocab_size - vocab_size,
        "bits_per_token": bits_per_token,
        "modulation": args.modulation,
        "snr_values": args.snr_values,
        "num_samples": len(texts),
        "iterations": args.iterations,
        "beam_size": args.beam_size,
        "candidate_mode": args.candidate_mode,
        "language_weight": args.language_weight,
        "channel_weight": args.channel_weight,
        "known_token_length_from_framing": True,
        "npy_columns": [
            "snr_db",
            "mean_f1",
            "mean_cosine",
            "mean_bert_score",
        ],
    }
    (output_dir / f"{args.output_name}.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nFinal aggregate:")
    print(final_array)
    print(f"Saved: {npy_path}")
    if args.save_sample_csv and csv_path.exists():
        print(f"Saved: {csv_path}")
    if examples_path.exists():
        print(f"Saved: {examples_path}")


if __name__ == "__main__":
    main()
