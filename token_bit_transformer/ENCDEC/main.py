import csv
import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import AutoTokenizer, logging as hf_logging

from config import (
    ModelConfig,
    TrainConfig,
    DataConfig,
    ChannelConfig,
    LogConfig,
    WandbConfig,
)
from loader import get_dataset_and_tokenizer, load_token_ids, sample_token_batch
from models import SemanticBitEncoder
from models import BitToTokenDecoder
from utils import bsc_flip, ber_from_snr
from blocks import attention_in_degree, normalize_importance

try:
    from bert_score import score as bert_score
except Exception:
    bert_score = None

hf_logging.set_verbosity_error()

try:
    import wandb
except Exception:
    wandb = None
os.environ["WANDB_API_KEY"] = "4aed4fc9f49dd25450fbf91f2014ded537dcfb1a"


def token_ce_loss(logits, target, weight=None):
    # logits: (B,N,V), target: (B,N)
    B, N, V = logits.shape
    loss = F.cross_entropy(
        logits.reshape(B * N, V), target.reshape(B * N), reduction="none"
    ).reshape(B, N)
    if weight is not None:
        loss = loss * weight
        return loss.sum() / (weight.sum() + 1e-8)
    return loss.mean()


def sample_channel_state(batch_size: int, device: str, snr_db: float = 10.0):
    snr = torch.full((batch_size,), snr_db, device=device)
    return snr


def ensure_csv_header(csv_path: str, fieldnames: list):
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()


def compute_bert_f1(
    reference_text: str, candidate_text: str, device: str, tokenizer, max_length: int
) -> float:
    if bert_score is None:
        raise ImportError("bert-score package is required. Please install bert-score.")

    ref_ids = tokenizer.encode(
        reference_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )
    cand_ids = tokenizer.encode(
        candidate_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )
    ref_text = tokenizer.decode(ref_ids, skip_special_tokens=True)
    cand_text = tokenizer.decode(cand_ids, skip_special_tokens=True)

    _, _, f1 = bert_score(
        [cand_text], [ref_text], lang="en", device=device, verbose=False
    )
    return float(f1[0].item())


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mcfg = ModelConfig()
    tcfg = TrainConfig()
    dcfg = DataConfig()
    ccfg = ChannelConfig()
    lcfg = LogConfig()
    wcfg = WandbConfig()
    bert_tokenizer = AutoTokenizer.from_pretrained(lcfg.bert_score_model)
    dataset, tokenizer = get_dataset_and_tokenizer(dcfg)

    mcfg.vocab_size = tokenizer.vocab_size
    token_ids = load_token_ids(dataset, tokenizer, split="train")

    enc = SemanticBitEncoder(mcfg).to(device)
    dec = BitToTokenDecoder(mcfg).to(device)

    opt = optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=tcfg.lr)

    if wcfg.enabled:
        if wandb is None:
            raise ImportError("wandb package is required. Please install wandb.")
        wandb.init(project=wcfg.project, name=wcfg.run_name)

    log_fields = [
        "batch_idx",
        "epoch_num",
        "input_text",
        "snr",
        "ber",
        "encoder_output_bit",
        "received_errored_bit",
        "decoder_output_token",
        "output_text",
        "bert_score_f1",
    ]
    ensure_csv_header(lcfg.csv_name, log_fields)

    for step in range(tcfg.steps):
        tokens = sample_token_batch(token_ids, tcfg.batch_size, tcfg.seq_len, device)
        snr = sample_channel_state(tcfg.batch_size, device, snr_db=ccfg.snr_db)
        ber = ber_from_snr(snr, ccfg.modulation)

        # Encoder
        enc_out = enc(tokens, snr, ber)
        lengths_hard = enc_out["lengths_hard"]
        lengths_expect = enc_out["lengths_expect"]
        payload = enc_out["payload_bits"]  # (B,N,Lmax)

        # Build per-token masking for payload (truncate beyond length)
        B, N, Lmax = payload.shape
        pos = torch.arange(Lmax, device=device)[None, None, :].expand(B, N, Lmax)
        trunc_mask = (pos < lengths_hard[:, :, None]).float()
        payload_eff = payload * trunc_mask

        # Channel: bit flip on payload stream (token-wise here for simplicity)
        # (In your real system: you pack to stream, modulate, demod -> bits/LLR)
        payload_noisy = bsc_flip(
            payload_eff.reshape(B, -1), snr, ccfg.modulation
        ).reshape(B, N, Lmax)

        # Decoder
        logits = dec(payload_noisy, lengths_hard, snr, ber)

        # Importance from self-attention in-degree (detach)
        attn = enc_out["attn_weights"]  # (L,B,H,N,N)
        attn_mean = attn.mean(dim=0)  # (B,H,N,N)
        indeg = attention_in_degree(attn_mean)  # (B,N)
        weight = normalize_importance(indeg, alpha=tcfg.alpha_imp).detach()

        # Recon loss (token CE)
        L_rec = token_ce_loss(logits, tokens, weight=weight)

        # Rate loss (expected bits per token)
        L_rate = lengths_expect.mean()

        loss = L_rec + tcfg.lambda_rate * L_rate

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(list(enc.parameters()) + list(dec.parameters()), 1.0)
        opt.step()

        if wcfg.enabled:
            wandb.log(
                {
                    "L_rec": float(L_rec.item()),
                    "L_rate": float(L_rate.item()),
                },
                step=step,
            )

        if (step + 1) % 5 == 0:
            epoch_num = step // tcfg.steps_per_epoch
            batch_idx = step % tcfg.steps_per_epoch
            output_tokens = logits.argmax(dim=-1)

            bert_f1_values = []

            with open(lcfg.csv_name, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=log_fields)
                for i in range(tokens.size(0)):
                    input_ids = tokens[i].tolist()
                    output_ids = output_tokens[i].tolist()

                    input_text = tokenizer.decode(input_ids, skip_special_tokens=True)
                    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)

                    bert_f1 = compute_bert_f1(
                        input_text,
                        output_text,
                        device,
                        bert_tokenizer,
                        lcfg.bert_score_max_length,
                    )
                    bert_f1_values.append(bert_f1)

                    row = {
                        "batch_idx": int(batch_idx),
                        "epoch_num": int(epoch_num),
                        "input_text": input_text,
                        "snr": float(snr[i].item()),
                        "ber": float(ber[i].item()),
                        "encoder_output_bit": json.dumps(payload_eff[i].tolist()),
                        "received_errored_bit": json.dumps(payload_noisy[i].tolist()),
                        "decoder_output_token": json.dumps(output_ids),
                        "output_text": output_text,
                        "bert_score_f1": float(bert_f1),
                    }
                    writer.writerow(row)

            if wcfg.enabled and len(bert_f1_values) > 0:
                wandb.log(
                    {"bert_score_f1": float(sum(bert_f1_values) / len(bert_f1_values))},
                    step=step,
                )

        if step % 100 == 0:
            print(
                f"step={step} loss={loss.item():.4f} rec={L_rec.item():.4f} rate={L_rate.item():.2f}"
            )


if __name__ == "__main__":
    main()
