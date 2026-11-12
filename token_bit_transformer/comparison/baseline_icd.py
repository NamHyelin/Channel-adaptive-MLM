#!/usr/bin/env python3
"""
Single-file implementation of the ECCT-aided ICD text communication pipeline.

Run only one of:
    python icd_full_train_test.py train
    python icd_full_train_test.py test

TRAIN
    Random information bits -> systematic LDPC(49,24) -> BPSK/noisy channel
    -> ECCT training.  The best checkpoint is selected by validation BER.

TEST
    Text -> GPT-2 arithmetic source coding -> LDPC(49,24) -> BPSK/channel
    -> ECCT -> context overwrite -> blockwise CCG -> CCS -> GPT-2 arithmetic
    decoding -> CLR -> SNR-wise cosine similarity.

The ICD paper does not publish its exact implementation/checkpoint or all
hyperparameters.  The code follows the published CCG/CCS/CLR equations and the
public ECCT architecture, while using blockwise beam combination and robust
fallbacks so multi-block sentences can be evaluated in practice.

Dependencies:
    pip install torch transformers pyldpc numpy datasets
"""

from __future__ import annotations

import bisect
import csv
import math
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# CONFIGURATION
# =============================================================================

# `python icd_full_train_test.py train/test` overrides this value.
RUN_MODE = "test"  # "train" or "test"

DEVICE = "cuda"
SEED = 42
LDPC_SEED = 42
CHANNEL = "awgn"  # "awgn" or "rayleigh"
CHECKPOINT_PATH = "C:/token_bit_transformer/comparison/icd_ecct_checkpoint.pt" #_last #49000이 제일좋은거긴함

# ECCT training
TRAIN_STEPS = 50_000
TRAIN_BATCH_SIZE = 128
LEARNING_RATE = 1e-4
TRAIN_SNR_MIN = -6.0
TRAIN_SNR_MAX = 6.0
ECCT_DROPOUT = 0.0
GRAD_CLIP = 1.0
LOG_EVERY = 100
VALIDATE_EVERY = 500
VALIDATION_BATCH_SIZE = 4096
VALIDATION_SNRS = [-5.0, 0.0, 5.0]

# Language source coder
LANGUAGE_MODEL_NAME = "gpt2"
AC_CDF_SCALE = 1 << 20
LM_CONTEXT_WINDOW = 1024

# ICD receiver
CONTEXT_WORDS = 3

# Candidate generation is performed independently for every 24-bit LDPC
# information block, then the block candidates are combined by beam search.
BLOCK_CANDIDATES = 4
GLOBAL_BEAM_WIDTH = 24
LC = 24
LS = 6

MH_STEPS = 50
MH_BETA = 1.0
DIVERSITY_LAMBDA = 0.10
CLR_ALPHA = 0.05
MAX_DECODE_TOKENS = 64

# A candidate that does not produce EOS is retained with a likelihood penalty
# instead of being discarded. Strict context is tried first; when every strict
# candidate fails, the same context-overwritten candidates are ranked with a
# mismatch penalty rather than aborting the sample.
EOS_MISSING_PENALTY = 20.0
CONTEXT_MISMATCH_PENALTY = 20.0
DEBUG_DIAGNOSTICS = False

# SNR evaluation
SNR_VALUES = list(range(10, -6,-1))
NUM_SAMPLES = 100
ITERATIONS = 1

DATASET_NAME = "HuggingFaceM4/COCO"
DATASET_SPLIT = "train"
TEXT_FIELD = "caption"

METRIC_MODEL_NAME = "google/bert_uncased_L-4_H-512_A-8"
METRIC_MODEL_REVISION = "refs/pr/1"
METRIC_BATCH_SIZE = 32

OUTPUT_DIR = "C:/token_bit_transformer/results/comparison"
OUTPUT_NAME = "baseline_icd_full"


def require_transformers():
    try:
        from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "transformers is required. Install it with: pip install transformers"
        ) from exc
    return AutoModel, AutoModelForCausalLM, AutoTokenizer


def require_pyldpc():
    try:
        from pyldpc import make_ldpc
    except ImportError as exc:
        raise ImportError(
            "pyldpc is required. Install it with: pip install pyldpc"
        ) from exc
    return make_ldpc


def require_data_module():
    try:
        from data_coco import iter_coco_captions
    except ImportError as exc:
        raise ImportError(
            "data_coco module required. Ensure you run this from the project root."
        ) from exc
    return iter_coco_captions


# -----------------------------------------------------------------------------
# General utilities
# -----------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_seed(base: int, snr_index: int, iteration: int, sample_index: int) -> int:
    """Identical deterministic channel realization across comparison baselines."""
    return int(base + 1_000_003 * snr_index + 10_007 * iteration + sample_index)


def resolve_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        print("[warning] CUDA was requested but is unavailable; using CPU.")
        return torch.device("cpu")
    return torch.device(name)


def dense_numpy(x) -> np.ndarray:
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x)


def sign_to_bin(x: torch.Tensor) -> torch.Tensor:
    """Maps +1 -> 0 and -1 -> 1, matching the ECCT reference code."""
    return 0.5 * (1.0 - x)


def bin_to_sign(x: torch.Tensor) -> torch.Tensor:
    """Maps 0 -> +1 and 1 -> -1."""
    return 1.0 - 2.0 * x


def nonzero_sign(x: torch.Tensor) -> torch.Tensor:
    """torch.sign with the measure-zero value 0 mapped to +1."""
    s = torch.sign(x)
    return torch.where(s == 0, torch.ones_like(s), s)


def snr_db_to_noise_std(snr_db: float) -> float:
    """Reference LLM-SSCC convention: sigma = 1/sqrt(2*10^(SNR/10))."""
    snr = 10.0 ** (float(snr_db) / 10.0)
    return 1.0 / math.sqrt(2.0 * snr)


def chunk_bits(bits: Sequence[int], k: int) -> tuple[torch.Tensor, int]:
    original_length = len(bits)
    n_blocks = max(1, math.ceil(original_length / k))
    padded = list(int(b) for b in bits) + [0] * (n_blocks * k - original_length)
    return torch.tensor(padded, dtype=torch.long).view(n_blocks, k), original_length


def first_n_word_boundary(text: str, n_words: int) -> int:
    if n_words <= 0:
        return 0
    matches = list(re.finditer(r"\S+", text))
    if not matches:
        return 0
    return matches[min(n_words, len(matches)) - 1].end()


# -----------------------------------------------------------------------------
# LDPC(49,24), generated exactly as in the public LLM-SSCC reference code.
# -----------------------------------------------------------------------------
@dataclass
class LDPCCode:
    n: int
    k: int
    H: torch.Tensor  # [n-k-ish, n]
    G: torch.Tensor  # [n, k]

    def to(self, device: torch.device) -> "LDPCCode":
        return LDPCCode(self.n, self.k, self.H.to(device), self.G.to(device))

    def encode(self, message_bits: torch.Tensor) -> torch.Tensor:
        """message_bits [..., k] -> binary codeword [..., n].

        CUDA does not implement matrix multiplication for torch.int64. Since
        both operands are binary and the block is small, float32 multiplication
        is exact here; the result is reduced modulo 2 and converted back to long.
        """
        products = message_bits.to(dtype=torch.float32) @ self.G.T.to(dtype=torch.float32)
        return torch.remainder(products.round(), 2).to(dtype=torch.long)

    def syndrome(self, codeword_bits: torch.Tensor) -> torch.Tensor:
        """codeword_bits [..., n] -> binary syndrome [..., rows(H)]."""
        products = codeword_bits.to(dtype=torch.float32) @ self.H.T.to(dtype=torch.float32)
        return torch.remainder(products.round(), 2).to(dtype=torch.long)


def make_paper_ldpc(seed: int = 42) -> LDPCCode:
    make_ldpc = require_pyldpc()
    n, requested_k = 49, 24
    d_c = int(math.sqrt(n))  # 7
    d_v = int(d_c + 1 - requested_k / (d_c - 1))  # 4
    H_np, G_np = make_ldpc(n, d_v, d_c, systematic=True, sparse=True, seed=seed)
    H_np = dense_numpy(H_np).astype(np.int64) % 2
    G_np = dense_numpy(G_np).astype(np.int64) % 2
    if G_np.shape[0] != n and G_np.shape[1] == n:
        G_np = G_np.T
    if G_np.shape[0] != n:
        raise RuntimeError(
            f"Unexpected pyldpc generator shape {G_np.shape}; expected ({n}, k)."
        )
    k = int(G_np.shape[1])
    if k != requested_k:
        raise RuntimeError(
            f"pyldpc generated LDPC({n},{k}), not the paper's LDPC({n},{requested_k}). "
            "Use a pyldpc version compatible with the public LLM-SSCC repository."
        )
    if np.any((H_np @ G_np) % 2):
        raise RuntimeError("Generated H and G do not satisfy H G = 0 over GF(2).")
    # The paper extracts the first K codeword positions, requiring systematic G.
    if not np.array_equal(G_np[:k, :], np.eye(k, dtype=np.int64)):
        raise RuntimeError(
            "pyldpc did not place the systematic information bits in the first K positions. "
            "The paper assumes this ordering."
        )
    return LDPCCode(
        n=n,
        k=k,
        H=torch.from_numpy(H_np),
        G=torch.from_numpy(G_np),
    )


# -----------------------------------------------------------------------------
# ECCT architecture, following the public ECCT Model.py.
# -----------------------------------------------------------------------------
def clones(module: nn.Module, n: int) -> nn.ModuleList:
    import copy

    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class SublayerConnection(nn.Module):
    def __init__(self, size: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class MultiHeadedAttention(nn.Module):
    def __init__(self, heads: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        if d_model % heads != 0:
            raise ValueError("d_model must be divisible by number of heads")
        self.d_k = d_model // heads
        self.heads = heads
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch = query.size(0)
        query, key, value = [
            layer(x).view(batch, -1, self.heads, self.d_k).transpose(1, 2)
            for layer, x in zip(self.linears, (query, key, value))
        ]
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        probs = F.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        x = probs @ value
        x = x.transpose(1, 2).contiguous().view(batch, -1, self.heads * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.dropout(F.gelu(self.w1(x))))


class EncoderLayer(nn.Module):
    def __init__(
        self,
        size: int,
        self_attn: MultiHeadedAttention,
        feed_forward: PositionwiseFeedForward,
        dropout: float,
    ):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayers = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        x = self.sublayers[0](x, lambda z: self.self_attn(z, z, z, mask))
        return self.sublayers[1](x, self.feed_forward)


class ECCTEncoder(nn.Module):
    def __init__(self, layer: EncoderLayer, n_layers: int):
        super().__init__()
        self.layers = clones(layer, n_layers)
        self.norm = nn.LayerNorm(layer.size)
        self.mid_norm = nn.LayerNorm(layer.size) if n_layers > 1 else None

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        for idx, layer in enumerate(self.layers, start=1):
            x = layer(x, mask)
            if self.mid_norm is not None and idx == len(self.layers) // 2:
                x = self.mid_norm(x)
        return self.norm(x)


class ECCT(nn.Module):
    def __init__(
        self,
        code: LDPCCode,
        n_layers: int = 6,
        d_model: int = 32,
        heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n = code.n
        self.k = code.k
        self.register_buffer("H", code.H.long())
        syndrome_len = code.H.size(0)
        sequence_len = code.n + syndrome_len

        attn = MultiHeadedAttention(heads, d_model)
        ff = PositionwiseFeedForward(d_model, d_model * 4, dropout)
        self.src_embed = nn.Parameter(torch.empty(sequence_len, d_model))
        self.decoder = ECCTEncoder(EncoderLayer(d_model, attn, ff, dropout), n_layers)
        self.scalar_head = nn.Linear(d_model, 1)
        self.out_fc = nn.Linear(sequence_len, code.n)
        self.register_buffer("src_mask", self._build_mask(code.H))
        # Allows automatic compatibility with checkpoints trained by the older
        # sign-equivalent implementation used in earlier drafts of this file.
        self.output_multiplier = 1.0

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def _build_mask(H: torch.Tensor) -> torch.Tensor:
        n = H.size(1)
        m = H.size(0)
        mask = torch.eye(n + m, n + m, dtype=torch.float32)
        for row in range(m):
            idx = torch.where(H[row] > 0)[0]
            for j in idx.tolist():
                for k in idx.tolist():
                    if j != k:
                        mask[j, k] = 1
                        mask[k, j] = 1
                mask[n + row, j] = 1
                mask[j, n + row] = 1
        # True entries are blocked in masked_fill, as in the public code.
        return ~(mask > 0).unsqueeze(0).unsqueeze(0)

    def forward(
        self, magnitude: torch.Tensor, syndrome_sign: torch.Tensor
    ) -> torch.Tensor:
        features = torch.cat([magnitude, syndrome_sign], dim=-1).unsqueeze(-1)
        embedded = self.src_embed.unsqueeze(0) * features
        hidden = self.decoder(embedded, self.src_mask)
        logits = self.out_fc(self.scalar_head(hidden).squeeze(-1))
        return float(self.output_multiplier) * logits

    def loss_and_codeword(
        self,
        z_pred: torch.Tensor,
        multiplicative_noise: torch.Tensor,
        received: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Official ECCT convention: positive logit means that the hard channel
        # decision should be flipped.  multiplicative_noise = y * x, whose
        # negative sign exactly identifies a hard-decision error.
        target_flip = sign_to_bin(nonzero_sign(multiplicative_noise))
        loss = F.binary_cross_entropy_with_logits(z_pred, target_flip)
        x_pred = sign_to_bin(
            nonzero_sign(-z_pred * nonzero_sign(received))
        ).long()
        return loss, x_pred


# -----------------------------------------------------------------------------
# GPT-2 arithmetic source coder.
# Uses a 31-bit arithmetic state, as specified by the ICD paper.
# -----------------------------------------------------------------------------
@dataclass
class SourcePacket:
    text: str
    token_ids: list[int]
    bits: list[int]
    emitted_bits_after_token: list[int]
    context_token_count: int
    context_bit_count: int
    context_token_ids: list[int]


class ArithmeticCodingError(RuntimeError):
    pass


class GPT2ArithmeticCoder:
    """31-bit integer arithmetic coder driven by frozen GPT-2 probabilities.

    GPT-2 KV caching is used during both encoding and decoding.  This preserves
    the probability model while avoiding recomputation of the full prefix at
    every token.
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: torch.device,
        precision: int = 31,
        cdf_scale: int = 1 << 20,
        max_context_tokens: int | None = None,
    ):
        if precision < 16:
            raise ValueError("Arithmetic precision is too small")
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.precision = int(precision)
        self.top = 1 << self.precision
        self.max_value = self.top - 1
        self.half = self.top >> 1
        self.quarter = self.half >> 1
        self.three_quarters = self.quarter * 3
        self.cdf_scale = int(cdf_scale)
        self.max_context_tokens = int(
            max_context_tokens or getattr(model.config, "n_positions", 1024) or 1024
        )
        self.bos_id = tokenizer.bos_token_id
        self.eos_id = tokenizer.eos_token_id
        if self.bos_id is None or self.eos_id is None:
            raise ValueError("The tokenizer must provide BOS and EOS token IDs")
        self.model.eval()

    @torch.inference_mode()
    def _start_lm(self):
        token = torch.tensor([[int(self.bos_id)]], dtype=torch.long, device=self.device)
        output = self.model(input_ids=token, use_cache=True, return_dict=True)
        probs = F.softmax(output.logits[0, -1].float(), dim=-1).double().cpu()
        return probs, output.past_key_values

    @torch.inference_mode()
    def _advance_lm(self, token_id: int, past_key_values):
        token = torch.tensor([[int(token_id)]], dtype=torch.long, device=self.device)
        output = self.model(
            input_ids=token,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        probs = F.softmax(output.logits[0, -1].float(), dim=-1).double().cpu()
        return probs, output.past_key_values

    def probs_to_cdf(self, probs: torch.Tensor) -> tuple[list[int], int]:
        freq = torch.floor(probs * self.cdf_scale).long().clamp_min(1)
        cdf_t = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(freq, 0)])
        total = int(cdf_t[-1].item())
        if total >= self.quarter:
            raise ArithmeticCodingError(
                f"Frequency total {total} is too large for precision {self.precision}"
            )
        return cdf_t.tolist(), total

    def tokenize_with_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        return (
            list(map(int, encoded["input_ids"])),
            [tuple(map(int, x)) for x in encoded["offset_mapping"]],
        )

    def encode_text(self, text: str, context_words: int = 0) -> SourcePacket:
        text_ids, offsets = self.tokenize_with_offsets(text)
        token_ids = text_ids + [int(self.eos_id)]
        bits, emitted = self.encode_tokens(token_ids)
        boundary = first_n_word_boundary(text, context_words)
        context_token_count = (
            sum(1 for _, end in offsets if end <= boundary) if boundary else 0
        )
        context_token_count = min(context_token_count, len(text_ids))
        context_bit_count = emitted[context_token_count - 1] if context_token_count else 0
        return SourcePacket(
            text=text,
            token_ids=token_ids,
            bits=bits,
            emitted_bits_after_token=emitted,
            context_token_count=context_token_count,
            context_bit_count=context_bit_count,
            context_token_ids=text_ids[:context_token_count],
        )

    def encode_tokens(self, token_ids: Sequence[int]) -> tuple[list[int], list[int]]:
        if not token_ids:
            raise ArithmeticCodingError("Cannot encode an empty token sequence")
        low, high = 0, self.max_value
        pending = 0
        output_bits: list[int] = []
        emitted_after: list[int] = []
        probs, past = self._start_lm()

        def emit(bit: int) -> None:
            nonlocal pending
            output_bits.append(int(bit))
            output_bits.extend([1 - int(bit)] * pending)
            pending = 0

        for index, target in enumerate(token_ids):
            cdf, total = self.probs_to_cdf(probs)
            if target < 0 or target + 1 >= len(cdf):
                raise ArithmeticCodingError(f"Token ID {target} is outside the vocabulary")
            current_range = high - low + 1
            new_high = low + (current_range * cdf[target + 1] // total) - 1
            new_low = low + (current_range * cdf[target] // total)
            if new_low > new_high:
                raise ArithmeticCodingError("Encoder reached a zero-width interval")
            low, high = new_low, new_high

            while True:
                if high < self.half:
                    emit(0)
                elif low >= self.half:
                    emit(1)
                    low -= self.half
                    high -= self.half
                elif low >= self.quarter and high < self.three_quarters:
                    pending += 1
                    low -= self.quarter
                    high -= self.quarter
                else:
                    break
                low = (low << 1) & self.max_value
                high = ((high << 1) & self.max_value) | 1

            emitted_after.append(len(output_bits))
            if index + 1 < len(token_ids):
                probs, past = self._advance_lm(int(target), past)

        pending += 1
        emit(0 if low < self.quarter else 1)
        return output_bits, emitted_after

    def decode_bits(
        self,
        bits: Sequence[int],
        max_tokens: int = 64,
        required_prefix_tokens: Sequence[int] = (),
        strict_context: bool = True,
        eos_missing_penalty: float = EOS_MISSING_PENALTY,
        context_mismatch_penalty: float = CONTEXT_MISMATCH_PENALTY,
    ) -> tuple[str, list[int], float]:
        bit_list = [int(b) for b in bits]
        read_index = 0

        def read_bit() -> int:
            nonlocal read_index
            value = bit_list[read_index] if read_index < len(bit_list) else 0
            read_index += 1
            return int(value)

        low, high = 0, self.max_value
        code_value = 0
        for _ in range(self.precision):
            code_value = ((code_value << 1) & self.max_value) | read_bit()

        decoded: list[int] = []
        log_likelihood = 0.0
        context_mismatches = 0
        interval_collapsed = False
        probs, past = self._start_lm()

        for token_index in range(int(max_tokens)):
            cdf, total = self.probs_to_cdf(probs)
            current_range = high - low + 1
            if current_range <= 0:
                interval_collapsed = True
                break
            scaled = ((code_value - low + 1) * total - 1) // current_range
            target = bisect.bisect_right(cdf, int(scaled)) - 1
            target = max(0, min(target, len(cdf) - 2))

            probability = max(float(probs[target].item()), 1e-300)
            log_likelihood += math.log(probability)
            new_high = low + (current_range * cdf[target + 1] // total) - 1
            new_low = low + (current_range * cdf[target] // total)
            if new_low > new_high:
                interval_collapsed = True
                break
            low, high = new_low, new_high
            decoded.append(int(target))

            if token_index < len(required_prefix_tokens):
                if target != int(required_prefix_tokens[token_index]):
                    if strict_context:
                        raise ArithmeticCodingError("Decoded sequence violates reliable context")
                    context_mismatches += 1

            if target == int(self.eos_id):
                break

            while True:
                if high < self.half:
                    pass
                elif low >= self.half:
                    low -= self.half
                    high -= self.half
                    code_value -= self.half
                elif low >= self.quarter and high < self.three_quarters:
                    low -= self.quarter
                    high -= self.quarter
                    code_value -= self.quarter
                else:
                    break
                low = (low << 1) & self.max_value
                high = ((high << 1) & self.max_value) | 1
                code_value = ((code_value << 1) & self.max_value) | read_bit()

            probs, past = self._advance_lm(int(target), past)

        terminated_by_eos = bool(decoded and decoded[-1] == int(self.eos_id))
        text_ids = decoded[:-1] if terminated_by_eos else decoded
        text = self.tokenizer.decode(text_ids, skip_special_tokens=True)
        if not terminated_by_eos:
            log_likelihood -= float(eos_missing_penalty)
        if context_mismatches:
            log_likelihood -= float(context_mismatch_penalty) * context_mismatches
        if interval_collapsed:
            log_likelihood -= 50.0
        if not decoded:
            log_likelihood -= 100.0
        return text, text_ids, float(log_likelihood)


# -----------------------------------------------------------------------------
# Channel and ECCT receiver front end
# -----------------------------------------------------------------------------
@dataclass
class ChannelOutput:
    transmitted_codeword: torch.Tensor
    received: torch.Tensor
    hard_information_bits: torch.Tensor
    information_reliability: torch.Tensor
    raw_codeword_prediction: torch.Tensor


def sample_rayleigh(shape: Sequence[int], device: torch.device) -> torch.Tensor:
    u = torch.rand(*shape, device=device).clamp_(1e-12, 1 - 1e-12)
    return torch.sqrt(-2.0 * torch.log1p(-u))


def transmit_and_ecct_decode(
    message_blocks: torch.Tensor,
    code: LDPCCode,
    ecct: ECCT,
    snr_db: float,
    channel: str,
    device: torch.device,
) -> ChannelOutput:
    message_blocks = message_blocks.to(device).long()
    code = code.to(device)
    codewords = code.encode(message_blocks)
    tx = bin_to_sign(codewords.float())
    sigma = snr_db_to_noise_std(snr_db)
    noise = torch.randn_like(tx) * sigma
    if channel == "awgn":
        h = torch.ones_like(tx)
    elif channel == "rayleigh":
        h = sample_rayleigh(tx.shape, device)
    else:
        raise ValueError("channel must be awgn or rayleigh")
    received = h * tx + noise

    hard_channel = sign_to_bin(nonzero_sign(received)).long()
    syndrome_bits = code.syndrome(hard_channel)
    syndrome_sign = bin_to_sign(syndrome_bits.float())
    magnitude = torch.abs(received)

    ecct.eval()
    with torch.no_grad():
        z_pred = ecct(magnitude, syndrome_sign)
        _, x_pred = ecct.loss_and_codeword(z_pred, received * tx, received)
        # Confidence of the ECCT-corrected hard decision. The sign selects
        # keep/flip; its magnitude represents certainty.
        reliability = torch.sigmoid(torch.abs(z_pred))

    # The paper uses systematic encoding and extracts the first K positions.
    return ChannelOutput(
        transmitted_codeword=codewords.detach().cpu(),
        received=received.detach().cpu(),
        hard_information_bits=x_pred[:, : code.k].detach().cpu().long(),
        information_reliability=reliability[:, : code.k].detach().cpu(),
        raw_codeword_prediction=x_pred.detach().cpu().long(),
    )


# -----------------------------------------------------------------------------
# ICD: CCG, CCS, and CLR
# -----------------------------------------------------------------------------
@dataclass
class ICDCandidate:
    bits: torch.Tensor
    confidence: float
    flip_positions: tuple[int, ...]
    decoded_text: str | None = None
    decoded_token_ids: list[int] | None = None
    log_likelihood: float = float("-inf")
    fused_score: float = float("-inf")


def context_overwrite(
    hard_bits: torch.Tensor,
    reliability: torch.Tensor,
    clean_bits: Sequence[int],
    context_bit_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hard = hard_bits.clone().long()
    rho = reliability.clone().float()
    locked = torch.zeros_like(hard, dtype=torch.bool)
    count = min(int(context_bit_count), hard.numel(), len(clean_bits))
    if count > 0:
        hard[:count] = torch.tensor(clean_bits[:count], dtype=torch.long)
        rho[:count] = 1.0
        locked[:count] = True
    return hard, rho, locked


def ccg_top_candidates(
    hard_bits: torch.Tensor,
    reliability: torch.Tensor,
    locked: torch.Tensor,
    lc: int,
) -> list[ICDCandidate]:
    """
    Exact top-Lc solution of Eqs. (20)-(23) without explicitly materializing
    all 2^K patterns. Because FConf is additive across bits, retaining the top
    Lc partial paths after every bit gives the exact top-Lc complete patterns.
    """
    if lc <= 0:
        raise ValueError("Lc must be positive")
    hard = hard_bits.flatten().long().cpu()
    rho = reliability.flatten().float().cpu().clamp(0.0, 1.0)
    locked = locked.flatten().bool().cpu()

    # state = (score, tuple_of_flipped_positions)
    states: list[tuple[float, tuple[int, ...]]] = [(0.0, ())]
    for p in range(hard.numel()):
        if bool(locked[p]):
            states = [(score + float(rho[p]), flips) for score, flips in states]
            continue
        no_flip_score = float(rho[p])
        flip_score = float(1.0 - rho[p])
        expanded: list[tuple[float, tuple[int, ...]]] = []
        for score, flips in states:
            expanded.append((score + no_flip_score, flips))
            expanded.append((score + flip_score, flips + (p,)))
        # Stable deterministic tie breaking: larger score, then fewer flips,
        # then lexicographic flip positions.
        expanded.sort(key=lambda x: (-x[0], len(x[1]), x[1]))
        states = expanded[:lc]

    candidates: list[ICDCandidate] = []
    for score, flips in states:
        bits = hard.clone()
        if flips:
            idx = torch.tensor(flips, dtype=torch.long)
            bits[idx] = 1 - bits[idx]
        candidates.append(
            ICDCandidate(bits=bits, confidence=float(score), flip_positions=flips)
        )
    return candidates


def blockwise_ccg_beam(
    hard_bits: torch.Tensor,
    reliability: torch.Tensor,
    locked: torch.Tensor,
    block_size: int,
    per_block_lc: int,
    beam_width: int,
    global_lc: int,
) -> list[ICDCandidate]:
    """Generate CCG candidates per LDPC information block and beam-combine them.

    This prevents a few residual errors distributed over many 24-bit LDPC
    blocks from being lost in one enormous whole-sentence flip search.
    """
    hard = hard_bits.flatten().long().cpu()
    rho = reliability.flatten().float().cpu()
    lock = locked.flatten().bool().cpu()
    if hard.numel() % block_size != 0:
        raise ValueError("Blockwise CCG input length must be divisible by block_size")

    beam: list[ICDCandidate] = [
        ICDCandidate(
            bits=torch.empty(0, dtype=torch.long),
            confidence=0.0,
            flip_positions=(),
        )
    ]

    for block_index, start in enumerate(range(0, hard.numel(), block_size)):
        end = start + block_size
        local_candidates = ccg_top_candidates(
            hard[start:end],
            rho[start:end],
            lock[start:end],
            lc=max(1, int(per_block_lc)),
        )

        expanded: list[ICDCandidate] = []
        for prefix_candidate in beam:
            for local in local_candidates:
                global_flips = prefix_candidate.flip_positions + tuple(
                    start + p for p in local.flip_positions
                )
                expanded.append(
                    ICDCandidate(
                        bits=torch.cat([prefix_candidate.bits, local.bits]),
                        confidence=(
                            prefix_candidate.confidence + local.confidence
                        ),
                        flip_positions=global_flips,
                    )
                )

        expanded.sort(
            key=lambda candidate: (
                -candidate.confidence,
                len(candidate.flip_positions),
                candidate.flip_positions,
            )
        )
        beam = expanded[: max(int(beam_width), int(global_lc))]

        if DEBUG_DIAGNOSTICS:
            print(
                f"[blockwise CCG] block={block_index + 1}, "
                f"local={len(local_candidates)}, beam={len(beam)}"
            )

    beam.sort(
        key=lambda candidate: (
            -candidate.confidence,
            len(candidate.flip_positions),
            candidate.flip_positions,
        )
    )
    return beam[: int(global_lc)]


def pairwise_hamming_sum(
    candidates: Sequence[ICDCandidate], indices: Sequence[int]
) -> float:
    if len(indices) < 2:
        return 0.0
    bits = torch.stack([candidates[i].bits for i in indices]).long()
    distances = (bits[:, None, :] != bits[None, :, :]).sum(dim=-1).float()
    # Eq. (25) writes m != m', so both directions are included.
    return float(distances.sum().item())


def ccs_sample(
    candidates: Sequence[ICDCandidate],
    ls: int,
    mh_steps: int,
    beta: float,
    diversity_lambda: float,
    rng: random.Random,
) -> list[ICDCandidate]:
    lc = len(candidates)
    if not (0 < ls <= lc - 2):
        raise ValueError("The paper requires 0 < Ls <= Lc - 2")

    def energy(indices: Sequence[int]) -> float:
        confidence = sum(candidates[i].confidence for i in indices)
        diversity = pairwise_hamming_sum(candidates, indices)
        return -confidence - diversity_lambda * diversity

    current = list(range(ls))
    current_energy = energy(current)
    all_indices = set(range(lc))

    for _ in range(max(0, int(mh_steps))):
        remove_slot = rng.randrange(ls)
        outside = sorted(all_indices.difference(current))
        incoming = rng.choice(outside)
        proposal = current.copy()
        proposal[remove_slot] = incoming
        proposal_energy = energy(proposal)
        exponent = -float(beta) * (proposal_energy - current_energy)
        acceptance = 1.0 if exponent >= 0 else math.exp(max(exponent, -700.0))
        if rng.random() <= acceptance:
            current = proposal
            current_energy = proposal_energy

    return [candidates[i] for i in current]


def clr_decode_and_rank(
    candidates: Sequence[ICDCandidate],
    coder: GPT2ArithmeticCoder,
    original_bit_length: int,
    context_token_ids: Sequence[int],
    max_decode_tokens: int,
    alpha: float,
    strict_context: bool = True,
) -> ICDCandidate:
    best: ICDCandidate | None = None
    failure_reasons: Counter[str] = Counter()

    for candidate in candidates:
        bits = candidate.bits[:original_bit_length].tolist()
        try:
            text, token_ids, ll = coder.decode_bits(
                bits,
                max_tokens=max_decode_tokens,
                required_prefix_tokens=context_token_ids,
                strict_context=strict_context,
            )
            candidate.decoded_text = text
            candidate.decoded_token_ids = token_ids
            candidate.log_likelihood = float(ll)
            candidate.fused_score = candidate.confidence + float(alpha) * float(ll)
        except (ArithmeticCodingError, ValueError, OverflowError) as exc:
            failure_reasons[str(exc)] += 1
            candidate.decoded_text = None
            candidate.decoded_token_ids = None
            candidate.log_likelihood = float("-inf")
            candidate.fused_score = float("-inf")

        if best is None or candidate.fused_score > best.fused_score:
            best = candidate

    if best is None or not math.isfinite(best.fused_score):
        raise ArithmeticCodingError(
            "No ICD candidate produced a valid arithmetic decoding. "
            f"Candidate failures: {dict(failure_reasons)}"
        )
    return best


# -----------------------------------------------------------------------------
# Complete end-to-end system
# -----------------------------------------------------------------------------
@dataclass
class EndToEndResult:
    source_text: str
    reconstructed_text: str
    snr_db: float
    channel: str
    source_bits: int
    ldpc_blocks: int
    context_words: int
    context_bits: int
    hard_bit_errors_before_icd: int
    selected_bit_errors_after_icd: int
    selected_confidence: float
    selected_log_likelihood: float
    selected_fused_score: float
    selected_flip_positions: list[int]
    selection_stage: str


def run_end_to_end(
    text: str,
    coder: GPT2ArithmeticCoder,
    code: LDPCCode,
    ecct: ECCT,
    device: torch.device,
    snr_db: float,
    channel: str,
    context_words: int,
    lc: int,
    ls: int,
    mh_steps: int,
    beta: float,
    diversity_lambda: float,
    alpha: float,
    max_decode_tokens: int,
    rng: random.Random,
) -> EndToEndResult:
    source = coder.encode_text(text, context_words=context_words)
    blocks, original_bit_length = chunk_bits(source.bits, code.k)

    if DEBUG_DIAGNOSTICS:
        clean_text, _, _ = coder.decode_bits(
            source.bits,
            max_tokens=max_decode_tokens,
            required_prefix_tokens=(),
            strict_context=False,
        )
        print(
            "[AC clean round-trip] "
            f"bits={original_bit_length}, exact={clean_text == text}, "
            f"original={text!r}, decoded={clean_text!r}"
        )

    front = transmit_and_ecct_decode(blocks, code, ecct, snr_db, channel, device)

    # Keep the complete padded block layout for blockwise candidate generation.
    hard_full = front.hard_information_bits.flatten().long()
    rho_full = front.information_reliability.flatten().float()
    clean_padded = blocks.flatten().long().cpu()
    clean = clean_padded[:original_bit_length]
    hard_errors = int(
        (hard_full[:original_bit_length] != clean).sum().item()
    )

    hard_ctx, rho_ctx, locked = context_overwrite(
        hard_full,
        rho_full,
        clean_padded.tolist(),
        source.context_bit_count,
    )

    # Framing provides the final source-bit length, so padded zeros are fixed and
    # excluded from the candidate flip space.
    if original_bit_length < hard_ctx.numel():
        hard_ctx[original_bit_length:] = clean_padded[original_bit_length:]
        rho_ctx[original_bit_length:] = 1.0
        locked[original_bit_length:] = True

    pool = blockwise_ccg_beam(
        hard_ctx,
        rho_ctx,
        locked,
        block_size=code.k,
        per_block_lc=BLOCK_CANDIDATES,
        beam_width=GLOBAL_BEAM_WIDTH,
        global_lc=lc,
    )
    if not pool:
        raise ArithmeticCodingError("Blockwise CCG produced no candidates")

    effective_ls = min(int(ls), max(1, len(pool) - 2))
    if len(pool) >= 3 and effective_ls <= len(pool) - 2:
        sampled = ccs_sample(
            pool,
            ls=effective_ls,
            mh_steps=mh_steps,
            beta=beta,
            diversity_lambda=diversity_lambda,
            rng=rng,
        )
    else:
        sampled = pool[:effective_ls]

    if DEBUG_DIAGNOSTICS:
        def clean_rank(candidates: Sequence[ICDCandidate]) -> int | None:
            for index, candidate in enumerate(candidates):
                if torch.equal(candidate.bits[:original_bit_length], clean):
                    return index
            return None

        print(
            "[ICD diagnostics] "
            f"source_bits={original_bit_length}, blocks={blocks.size(0)}, "
            f"hard_errors={hard_errors}, pool={len(pool)}, sampled={len(sampled)}, "
            f"clean_rank_pool={clean_rank(pool)}, "
            f"clean_rank_sampled={clean_rank(sampled)}"
        )

    selection_stage = "sampled_strict"
    try:
        selected = clr_decode_and_rank(
            sampled,
            coder,
            original_bit_length=original_bit_length,
            context_token_ids=source.context_token_ids,
            max_decode_tokens=max_decode_tokens,
            alpha=alpha,
            strict_context=True,
        )
    except ArithmeticCodingError as sampled_error:
        if DEBUG_DIAGNOSTICS:
            print(f"[CLR fallback] sampled strict failed: {sampled_error}")
        selection_stage = "pool_strict"
        try:
            selected = clr_decode_and_rank(
                pool,
                coder,
                original_bit_length=original_bit_length,
                context_token_ids=source.context_token_ids,
                max_decode_tokens=max_decode_tokens,
                alpha=alpha,
                strict_context=True,
            )
        except ArithmeticCodingError as pool_error:
            if DEBUG_DIAGNOSTICS:
                print(f"[CLR fallback] pool strict failed: {pool_error}")
            selection_stage = "pool_relaxed_context"
            selected = clr_decode_and_rank(
                pool,
                coder,
                original_bit_length=original_bit_length,
                context_token_ids=source.context_token_ids,
                max_decode_tokens=max_decode_tokens,
                alpha=alpha,
                strict_context=False,
            )

    selected_errors = int(
        (selected.bits[:original_bit_length] != clean).sum().item()
    )
    return EndToEndResult(
        source_text=text,
        reconstructed_text=str(selected.decoded_text or ""),
        snr_db=float(snr_db),
        channel=channel,
        source_bits=original_bit_length,
        ldpc_blocks=int(blocks.size(0)),
        context_words=int(context_words),
        context_bits=int(source.context_bit_count),
        hard_bit_errors_before_icd=hard_errors,
        selected_bit_errors_after_icd=selected_errors,
        selected_confidence=float(selected.confidence),
        selected_log_likelihood=float(selected.log_likelihood),
        selected_fused_score=float(selected.fused_score),
        selected_flip_positions=list(selected.flip_positions),
        selection_stage=selection_stage,
    )

# =============================================================================
# ECCT TRAINING
# =============================================================================

def random_channel_batch(
    batch_size: int,
    code: LDPCCode,
    snr_min: float,
    snr_max: float,
    channel: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    code = code.to(device)
    message = torch.randint(0, 2, (batch_size, code.k), device=device)
    codeword = code.encode(message)
    tx = bin_to_sign(codeword.float())

    snr_db = torch.empty(batch_size, 1, device=device).uniform_(snr_min, snr_max)
    sigma = 1.0 / torch.sqrt(2.0 * torch.pow(10.0, snr_db / 10.0))
    noise = torch.randn_like(tx) * sigma

    if channel == "awgn":
        h = torch.ones_like(tx)
    elif channel == "rayleigh":
        h = sample_rayleigh(tx.shape, device)
    else:
        raise ValueError("CHANNEL must be 'awgn' or 'rayleigh'")

    received = h * tx + noise
    hard = sign_to_bin(nonzero_sign(received)).long()
    syndrome = bin_to_sign(code.syndrome(hard).float())
    magnitude = torch.abs(received)
    multiplicative_noise = received * tx
    return magnitude, syndrome, multiplicative_noise, received


@torch.inference_mode()
def validate_ecct(
    model: ECCT,
    code: LDPCCode,
    device: torch.device,
) -> tuple[float, dict[float, tuple[float, float]]]:
    model.eval()
    details: dict[float, tuple[float, float]] = {}
    ecct_values: list[float] = []
    for snr in VALIDATION_SNRS:
        magnitude, syndrome, z_mul, received = random_channel_batch(
            VALIDATION_BATCH_SIZE,
            code,
            float(snr),
            float(snr),
            CHANNEL,
            device,
        )
        z_pred = model(magnitude, syndrome)
        _, x_pred = model.loss_and_codeword(z_pred, z_mul, received)
        hard = sign_to_bin(nonzero_sign(received)).long()
        target = sign_to_bin(nonzero_sign(received * z_mul)).long()
        hard_ber = float((hard != target).float().mean().item())
        ecct_ber = float((x_pred != target).float().mean().item())
        details[float(snr)] = (hard_ber, ecct_ber)
        ecct_values.append(ecct_ber)
    model.train()
    return float(np.mean(ecct_values)), details


def _checkpoint_payload(
    model: ECCT,
    code: LDPCCode,
    step: int,
    train_loss: float,
    validation_ber: float,
) -> dict:
    return {
        "model_state_dict": model.state_dict(),
        "n": code.n,
        "k": code.k,
        "H": code.H.cpu(),
        "G": code.G.cpu(),
        "architecture": {"n_layers": 6, "d_model": 32, "heads": 8},
        "channel": CHANNEL,
        "train_snr_min": TRAIN_SNR_MIN,
        "train_snr_max": TRAIN_SNR_MAX,
        "step": int(step),
        "train_loss": float(train_loss),
        "validation_ber": float(validation_ber),
        "ecct_sign_convention": "official_flip_logit",
        "seed": SEED,
        "ldpc_seed": LDPC_SEED,
    }


def train_ecct() -> None:
    set_seed(SEED)
    device = resolve_device(DEVICE)
    code = make_paper_ldpc(LDPC_SEED)
    model = ECCT(
        code,
        n_layers=6,
        d_model=32,
        heads=8,
        dropout=ECCT_DROPOUT,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    checkpoint = Path(CHECKPOINT_PATH)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    last_path = checkpoint.with_name(f"{checkpoint.stem}_last{checkpoint.suffix}")
    best_validation_ber = float("inf")
    model.train()

    print(f"Training ECCT on {device}")
    print(f"Channel={CHANNEL}, train SNR=[{TRAIN_SNR_MIN}, {TRAIN_SNR_MAX}] dB")
    print(f"Best checkpoint: {checkpoint}")

    for step in range(1, TRAIN_STEPS + 1):
        magnitude, syndrome, z_mul, received = random_channel_batch(
            TRAIN_BATCH_SIZE,
            code,
            TRAIN_SNR_MIN,
            TRAIN_SNR_MAX,
            CHANNEL,
            device,
        )
        z_pred = model(magnitude, syndrome)
        loss, x_pred = model.loss_and_codeword(z_pred, z_mul, received)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        if step == 1 or step % LOG_EVERY == 0:
            with torch.no_grad():
                hard = sign_to_bin(nonzero_sign(received)).long()
                target = sign_to_bin(nonzero_sign(received * z_mul)).long()
                hard_ber = float((hard != target).float().mean().item())
                ecct_ber = float((x_pred != target).float().mean().item())
            print(
                f"step={step:7d} loss={loss.item():.6f} "
                f"hard_BER={hard_ber:.4e} ECCT_BER={ecct_ber:.4e}"
            )

        should_validate = step == 1 or step % VALIDATE_EVERY == 0 or step == TRAIN_STEPS
        if should_validate:
            mean_ber, details = validate_ecct(model, code, device)
            detail_text = ", ".join(
                f"{snr:g}dB:{ecct:.3e}" for snr, (_, ecct) in details.items()
            )
            print(f"[validation] step={step} mean_ECCT_BER={mean_ber:.4e} ({detail_text})")
            payload = _checkpoint_payload(
                model, code, step, float(loss.item()), mean_ber
            )
            torch.save(payload, last_path)
            if mean_ber < best_validation_ber:
                best_validation_ber = mean_ber
                torch.save(payload, checkpoint)
                print(f"[saved best] validation_BER={best_validation_ber:.4e}")

    print(f"Saved best checkpoint: {checkpoint}")
    print(f"Saved final checkpoint: {last_path}")


def load_ecct(
    checkpoint_path: str,
    code: LDPCCode,
    device: torch.device,
) -> ECCT:
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        # Compatibility with older PyTorch releases.
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    print(
        "[checkpoint loaded] "
        f"path={checkpoint_path}, "
        f"step={checkpoint.get('step', 'unknown')}, "
        f"validation_ber={checkpoint.get('validation_ber', 'unknown')}, "
        f"channel={checkpoint.get('channel', 'unknown')}"
    )

    if (
        int(checkpoint.get("n", code.n)) != code.n
        or int(checkpoint.get("k", code.k)) != code.k
    ):
        raise ValueError("Checkpoint dimensions do not match LDPC(49,24)")

    if "H" in checkpoint and not torch.equal(checkpoint["H"].long(), code.H.long()):
        raise ValueError(
            "Checkpoint parity-check matrix differs from the generated matrix. "
            "Use the same LDPC_SEED for training and testing."
        )

    architecture = checkpoint.get(
        "architecture", {"n_layers": 6, "d_model": 32, "heads": 8}
    )
    model = ECCT(code, **architecture).to(device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    convention = checkpoint.get("ecct_sign_convention")
    if convention != "official_flip_logit":
        # Earlier generated checkpoints used the mathematically equivalent
        # opposite-logit convention.  Negating the loaded model output converts
        # them to the official ECCT interpretation without retraining.
        model.output_multiplier = -1.0
        print("[checkpoint compatibility] applying legacy ECCT sign adapter")
    model.eval()
    return model


# =============================================================================
# COSINE-SIMILARITY EVALUATION
# =============================================================================

def sentence_embedding(
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
        output = model(**encoded, output_hidden_states=True, return_dict=True)

    hidden = output.hidden_states[-1]
    if getattr(tokenizer, "cls_token_id", None) is not None:
        pooled = hidden[:, 0, :]
    else:
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    return pooled.squeeze(0).float().cpu().numpy()


@torch.inference_mode()
def sentence_embeddings_batch(
    texts: Sequence[str],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    vectors: list[np.ndarray] = []
    for start in range(0, len(texts), int(batch_size)):
        batch = list(texts[start : start + int(batch_size)])
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)
        output = model(**encoded, return_dict=True)
        hidden = output.last_hidden_state
        if getattr(tokenizer, "cls_token_id", None) is not None:
            pooled = hidden[:, 0, :]
        else:
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        vectors.append(pooled.float().cpu().numpy())
    return np.concatenate(vectors, axis=0)


def cosine_rows(clean: np.ndarray, restored: np.ndarray) -> np.ndarray:
    numerator = np.sum(clean * restored, axis=1)
    denominator = np.linalg.norm(clean, axis=1) * np.linalg.norm(restored, axis=1)
    result = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float32),
        where=denominator > 0,
    )
    result[~np.isfinite(result)] = 0.0
    return result.astype(np.float32)


def cosine_similarity_metric(
    clean_text: str,
    restored_text: str,
    tokenizer,
    model,
    device: torch.device,
) -> float:
    if not restored_text:
        return 0.0

    try:
        clean = sentence_embedding(clean_text, tokenizer, model, device)
        restored = sentence_embedding(restored_text, tokenizer, model, device)
        denominator = float(np.linalg.norm(clean) * np.linalg.norm(restored))
        if denominator == 0.0:
            return 0.0
        value = float(np.dot(clean, restored) / denominator)
        return value if np.isfinite(value) else 0.0
    except Exception:
        return 0.0


def _extract_caption_values(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        result: list[str] = []
        for key in ("caption", "captions", "text", "raw"):
            if key in value:
                result.extend(_extract_caption_values(value[key]))
        return result
    if isinstance(value, (list, tuple)):
        result: list[str] = []
        for item in value:
            result.extend(_extract_caption_values(item))
        return result
    return []


def load_coco_texts() -> list[str]:
    # Prefer the user's existing helper so the exact same sample pipeline can be
    # reused.  Fall back to Hugging Face datasets to keep this file standalone.
    try:
        iter_coco_captions = require_data_module()
        texts = list(
            iter_coco_captions(
                DATASET_NAME,
                DATASET_SPLIT,
                TEXT_FIELD,
                NUM_SAMPLES,
            )
        )
    except ImportError:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "Install datasets (`pip install datasets`) or provide data_coco.py"
            ) from exc
        dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
        texts: list[str] = []
        for row in dataset:
            candidates = _extract_caption_values(row.get(TEXT_FIELD, row))
            for caption in candidates:
                caption = str(caption).strip()
                if caption:
                    texts.append(caption)
                    if len(texts) >= NUM_SAMPLES:
                        break
            if len(texts) >= NUM_SAMPLES:
                break

    texts = [str(text).strip() for text in texts if str(text).strip()]
    if not texts:
        raise ValueError("No caption text was loaded")
    return texts[:NUM_SAMPLES]


def load_language_coder(device: torch.device) -> GPT2ArithmeticCoder:
    _, AutoModelForCausalLM, AutoTokenizer = require_transformers()
    print(f"Loading source language model: {LANGUAGE_MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(LANGUAGE_MODEL_NAME, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(LANGUAGE_MODEL_NAME).to(device)
    model.eval()

    return GPT2ArithmeticCoder(
        model,
        tokenizer,
        device,
        precision=31,
        cdf_scale=AC_CDF_SCALE,
        max_context_tokens=LM_CONTEXT_WINDOW,
    )


def load_metric_model(device: torch.device):
    AutoModel, _, AutoTokenizer = require_transformers()
    print(f"Loading cosine metric model: {METRIC_MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(METRIC_MODEL_NAME)
    kwargs = {}
    if METRIC_MODEL_REVISION not in {None, "", "none", "None"}:
        kwargs["revision"] = METRIC_MODEL_REVISION
    model = AutoModel.from_pretrained(METRIC_MODEL_NAME, **kwargs).to(device)
    model.eval()
    return tokenizer, model


def save_evaluation(
    per_sample_rows: list[dict[str, object]],
    cosine_by_snr: dict[float, list[float]],
) -> None:
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # NPY columns: [snr_db, mean_cosine_similarity]
    result_array = np.asarray(
        [
            [float(snr), float(np.mean(cosine_by_snr.get(float(snr), [0.0])))]
            for snr in SNR_VALUES
        ],
        dtype=np.float32,
    )
    npy_path = output_dir / f"{OUTPUT_NAME}.npy"
    np.save(npy_path, result_array)

    csv_path = output_dir / f"{OUTPUT_NAME}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "snr_db",
            "iteration",
            "sample_idx",
            "clean_text",
            "reconstructed_text",
            "cosine_similarity",
            "selection_stage",
            "error",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_sample_rows)

    print(f"Saved aggregate cosine results: {npy_path}")
    print(f"Saved per-sample results: {csv_path}")


def evaluate_snr_cosine() -> None:
    set_seed(SEED)
    device = resolve_device(DEVICE)
    print(f"Testing on {device}")

    code = make_paper_ldpc(LDPC_SEED)
    ecct = load_ecct(CHECKPOINT_PATH, code, device)
    coder = load_language_coder(device)
    metric_tokenizer, metric_model = load_metric_model(device)
    texts = load_coco_texts()

    # Fail early when the source coder itself is inconsistent.  This separates
    # arithmetic-coder problems from channel/ICD problems.
    preflight = coder.encode_text(texts[0], context_words=CONTEXT_WORDS)
    roundtrip_text, _, _ = coder.decode_bits(
        preflight.bits,
        max_tokens=MAX_DECODE_TOKENS,
        required_prefix_tokens=(),
        strict_context=False,
    )
    if roundtrip_text.strip() != texts[0].strip():
        raise RuntimeError(
            "Clean arithmetic-coding round-trip failed before channel testing. "
            f"original={texts[0]!r}, decoded={roundtrip_text!r}"
        )
    print("[preflight passed] clean GPT-2 arithmetic coding round-trip")

    print("Precomputing clean-text metric embeddings...")
    clean_embeddings = sentence_embeddings_batch(
        texts,
        metric_tokenizer,
        metric_model,
        device,
        METRIC_BATCH_SIZE,
    )

    per_sample_rows: list[dict[str, object]] = []
    cosine_by_snr: dict[float, list[float]] = {float(snr): [] for snr in SNR_VALUES}

    for snr_index, snr in enumerate(SNR_VALUES):
        print(f"\n=== SNR {snr:g} dB ===")
        for iteration in range(ITERATIONS):
            restored_texts: list[str] = []
            iteration_rows: list[dict[str, object]] = []

            for sample_index, clean_text in enumerate(texts):
                local_seed = sample_seed(SEED, snr_index, iteration, sample_index)
                set_seed(local_seed)
                try:
                    result = run_end_to_end(
                        text=clean_text,
                        coder=coder,
                        code=code,
                        ecct=ecct,
                        device=device,
                        snr_db=float(snr),
                        channel=CHANNEL,
                        context_words=CONTEXT_WORDS,
                        lc=LC,
                        ls=LS,
                        mh_steps=MH_STEPS,
                        beta=MH_BETA,
                        diversity_lambda=DIVERSITY_LAMBDA,
                        alpha=CLR_ALPHA,
                        max_decode_tokens=MAX_DECODE_TOKENS,
                        rng=random.Random(local_seed),
                    )
                    restored_text = result.reconstructed_text
                    selection_stage = result.selection_stage
                    error = ""
                except Exception as exc:
                    restored_text = ""
                    selection_stage = "failed"
                    error = f"{type(exc).__name__}: {exc}"
                    print(
                        f"[failed] SNR={snr:g}, iteration={iteration}, "
                        f"sample={sample_index}: {error}"
                    )
                restored_texts.append(restored_text)
                iteration_rows.append(
                    {
                        "snr_db": float(snr),
                        "iteration": iteration,
                        "sample_idx": sample_index,
                        "clean_text": clean_text,
                        "reconstructed_text": restored_text,
                        "selection_stage": selection_stage,
                        "error": error,
                    }
                )
                if (sample_index + 1) % 10 == 0 or sample_index + 1 == len(texts):
                    print(f"decoded {sample_index + 1}/{len(texts)}")

            restored_embeddings = sentence_embeddings_batch(
                restored_texts,
                metric_tokenizer,
                metric_model,
                device,
                METRIC_BATCH_SIZE,
            )
            cosine_values = cosine_rows(clean_embeddings, restored_embeddings)
            for row, value in zip(iteration_rows, cosine_values.tolist()):
                row["cosine_similarity"] = float(value)
                per_sample_rows.append(row)
                cosine_by_snr[float(snr)].append(float(value))

            mean_cosine = float(np.mean(cosine_by_snr[float(snr)]))
            print(
                f"iteration={iteration + 1}/{ITERATIONS}, "
                f"SNR={snr:g} dB, mean cosine={mean_cosine:.6f}"
            )
            save_evaluation(per_sample_rows, cosine_by_snr)

    print("\nFinal SNR-wise cosine similarity")
    for snr in SNR_VALUES:
        values = cosine_by_snr[float(snr)]
        mean_cosine = float(np.mean(values)) if values else 0.0
        print(f"SNR={snr:>3} dB: {mean_cosine:.6f}")


# =============================================================================
# SIMPLE MAIN
# =============================================================================

def main() -> None:
    mode = sys.argv[1].strip().lower() if len(sys.argv) > 1 else RUN_MODE

    if mode == "train":
        train_ecct()
    elif mode in {"test", "eval", "evaluate"}:
        evaluate_snr_cosine()
    else:
        raise ValueError(
            "Mode must be 'train' or 'test'.\n"
            "Example: python icd_full_train_test.py train"
        )


if __name__ == "__main__":
    main()
