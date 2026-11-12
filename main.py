# main.py
import os
import csv
import json
import math
import random
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from transformers import AutoTokenizer, AutoModelForMaskedLM
from datasets import disable_caching

from config import parse_args, Config
from data_coco import iter_coco_captions
from channel_corrupt import transmit_ids_awgn_llr, make_masked_input
from q_model import popcount_u16_lut, q_full_vocab_from_ber
from guidance import (
    make_soft_inputs_embeds,
    build_key_bias_from_q,
    BertModelWithKeyBias,
)
from fusion import AdaptiveFusion

disable_caching()


try:
    import wandb
except Exception:
    wandb = None
os.environ["WANDB_API_KEY"] = "4aed4fc9f49dd25450fbf91f2014ded537dcfb1a"


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_texts(tokenizer, texts, max_len: int):
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    return enc["input_ids"], enc["attention_mask"]


@torch.no_grad()
def compute_validation_metrics(
    clean_ids, restored_ids, tokenizer, loss_fn, model, device
):
    """
    Compute CE loss and F1 score between clean and restored token sequences.
    clean_ids: [B,L] original token ids
    restored_ids: [B,L] restored token ids
    """
    # CE loss: compare logits from restored ids to clean ids as targets
    logits = model(
        input_ids=restored_ids.to(device), output_hidden_states=False, return_dict=True
    ).logits
    ce_loss = loss_fn(logits.view(-1, logits.size(-1)), clean_ids.view(-1).to(device))

    # F1 score: token-level accuracy (treating as multi-class classification)
    pred_tokens = logits.argmax(dim=-1).cpu()
    clean_tokens = clean_ids.cpu()

    # Flatten for F1 calculation
    pred_flat = pred_tokens.view(-1).numpy()
    clean_flat = clean_tokens.view(-1).numpy()

    f1 = f1_score(clean_flat, pred_flat, average="macro", zero_division=0)

    return float(ce_loss.item()), float(f1)


@torch.no_grad()
def build_q_list_for_masks(
    vocab_size: int,
    xbar_ids: torch.Tensor,  # [B,L] hard received token ids
    mask_positions: list[tuple[int, int]],
    ber: float,
    lut_u16: torch.Tensor,
):
    q_list = []
    for b, pos in mask_positions:  # mask 된 token 들에 대해 만듦
        xbar = int(xbar_ids[b, pos].item())
        if xbar < 0 or xbar >= vocab_size:
            q = torch.full(
                (vocab_size,),
                1.0 / vocab_size,
                device=xbar_ids.device,
                dtype=torch.float32,
            )
        else:
            q = q_full_vocab_from_ber(vocab_size, xbar, ber, xbar_ids.device, lut_u16)
        q_list.append(q)
    return q_list


def main(cfg: Config):
    # set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForMaskedLM.from_pretrained(
        cfg.model_name, revision="refs/pr/1", use_safetensors=True
    ).to(device)
    model.train()

    # 2-A wrapper (only used when cfg.guide == "attn_bias")
    bert_bias_wrapper = (
        BertModelWithKeyBias(model.bert).to(device)
        if cfg.guide == "attn_bias"
        else None
    )

    fusion = (
        AdaptiveFusion(model.config.hidden_size).to(device) if cfg.use_fusion else None
    )

    optim = torch.optim.AdamW(
        list(model.parameters())
        + (list(fusion.parameters()) if fusion is not None else []),
        lr=cfg.lr,
    )
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    # Data
    captions = list(
        iter_coco_captions(cfg.dataset_name, cfg.split, cfg.text_field, cfg.max_samples)
    )
    dl = DataLoader(captions, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(
        captions, batch_size=cfg.batch_size, shuffle=False, drop_last=False
    )

    # bit-length for token ids
    m_bits = int(math.ceil(math.log2(model.config.vocab_size)))
    lut = popcount_u16_lut(
        device
    )  # vocab_size 이상의 개수를 2진법으로 표현할때 1의 개수. popcount_u16_lut[i] = i의 2진 표현에서 1의 개수

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    snr_str = f"{cfg.snr_db:.1f}" if cfg.snr_db is not None else "random"
    csv_path = f"results/fusion_{cfg.use_fusion}__guide_{cfg.guide}__snr_{snr_str}__mod_{cfg.modulation}_{run_tag}.csv"
    model_path = f"results/models/fusion_{cfg.use_fusion}__guide_{cfg.guide}__snr_{snr_str}__mod_{cfg.modulation}_{run_tag}.pth"
    step = 0

    # Initialize wandb
    wandb.init(
        project="token-bit-transformer",
        config={
            "snr_db": cfg.snr_db,
            "modulation": cfg.modulation,
            "guide": cfg.guide,
            "use_fusion": cfg.use_fusion,
            "mask_strategy": cfg.mask_strategy,
            "rel_tau": cfg.rel_tau,
            "batch_size": cfg.batch_size,
            "epochs": cfg.epochs,
        },
        name=f"snr_{snr_str}_mod_{cfg.modulation}",
    )

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "sample_idx",
                "snr_db",
                "modulation",
                "ber",
                "clean_text",
                "clean_token_ids",
                "corrupted_token_ids",
                "corrupted_text",
                "masked_token_ids",
                "masked_text",
                "restored_token_ids",
                "restored_text",
            ],
        )
        writer.writeheader()

        for epoch in range(cfg.epochs):
            for batch_idx, batch_texts in enumerate(dl):
                input_ids, attn_mask = collate_texts(
                    tokenizer, list(batch_texts), cfg.max_len
                )
                input_ids = input_ids.to(device)
                attn_mask = attn_mask.to(device)

                # clean tokens = input_ids
                x_clean = input_ids

                # channel: transmit through AWGN and get LLRs
                snr_db_used = (
                    cfg.snr_db if cfg.snr_db is not None else random.uniform(-5.0, 10.0)
                )
                xbar, llr_tok = transmit_ids_awgn_llr(
                    x_clean, snr_db_used, cfg.modulation, m_bits, tokenizer
                )

                # make Y with [MASK] (based on LLR reliability)
                xhat_ids, rel = make_masked_input(
                    xbar,
                    llr_tok,
                    tokenizer,
                    strategy=cfg.mask_strategy,
                    mask_rate=cfg.mask_rate,
                    rel_tau=cfg.rel_tau,
                    rel_mode=cfg.real_mode,
                )

                # targets: only compute loss on masked positions
                labels = x_clean.clone()
                labels[xhat_ids != tokenizer.mask_token_id] = -100  # ignore non-mask

                # list mask positions (batch-wise)
                mask_positions = (
                    (xhat_ids == tokenizer.mask_token_id)
                    .nonzero(as_tuple=False)
                    .tolist()
                )
                if len(mask_positions) == 0:
                    if batch_idx == len(dl) - 1:
                        pred_ids = xhat_ids.clone()
                        sample_indices = [0, 1] if input_ids.size(0) > 1 else [0]
                        for sidx in sample_indices:
                            attn = attn_mask[sidx].bool()
                            clean_list = x_clean[sidx][attn].tolist()
                            corrupted_list = xbar[sidx][attn].tolist()
                            masked_list = xhat_ids[sidx][attn].tolist()
                            restored_list = pred_ids[sidx][attn].tolist()

                            clean_text = tokenizer.decode(
                                clean_list, skip_special_tokens=True
                            )
                            corrupted_text = tokenizer.decode(
                                corrupted_list, skip_special_tokens=True
                            )
                            masked_text = tokenizer.decode(
                                masked_list, skip_special_tokens=False
                            )
                            restored_text = tokenizer.decode(
                                restored_list, skip_special_tokens=True
                            )

                            from channel_corrupt import ber_from_snr

                            ber_estimate, _ = ber_from_snr(snr_db_used, cfg.modulation)
                            writer.writerow(
                                {
                                    "epoch": epoch,
                                    "sample_idx": sidx,
                                    "snr_db": round(snr_db_used, 6),
                                    "modulation": cfg.modulation,
                                    "ber": round(ber_estimate, 8),
                                    "clean_text": clean_text,
                                    "clean_token_ids": json.dumps(clean_list),
                                    "corrupted_token_ids": json.dumps(corrupted_list),
                                    "corrupted_text": corrupted_text,
                                    "masked_token_ids": json.dumps(masked_list),
                                    "masked_text": masked_text,
                                    "restored_token_ids": json.dumps(restored_list),
                                    "restored_text": restored_text,
                                }
                            )
                        f.flush()
                    continue

                # q for each mask (compute BER from SNR for BSC approximation)
                from channel_corrupt import ber_from_snr

                ber, _ = ber_from_snr(snr_db_used, cfg.modulation)
                q_list = build_q_list_for_masks(
                    vocab_size=model.config.vocab_size,
                    xbar_ids=xbar,
                    mask_positions=[(b, p) for b, p in mask_positions],
                    ber=ber,
                    lut_u16=lut,
                )

                # ---- Forward with optional guidance ----
                if cfg.guide == "soft_embed":
                    inputs_embeds = make_soft_inputs_embeds(
                        model, xhat_ids, [(b, p) for b, p in mask_positions], q_list
                    )
                    out = model(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attn_mask,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    logits = out.logits
                    hidden_last = out.hidden_states[-1]

                elif cfg.guide == "attn_bias":
                    # 2-A: key bias from q summaries
                    key_bias = build_key_bias_from_q(
                        input_ids=xhat_ids,
                        mask_positions=[(b, p) for b, p in mask_positions],
                        q_list=q_list,
                        ber=ber,
                        mode="rmax",
                        beta=cfg.attn_beta_init,
                    ).to(device)

                    enc_out = bert_bias_wrapper(
                        input_ids=xhat_ids,
                        attention_mask=attn_mask,
                        key_bias=key_bias,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    hidden_last = enc_out["last_hidden_state"]  # [B,L,D]
                    # MLM head: use model.cls
                    logits = model.cls(hidden_last)  # [B,L,V]

                else:
                    out = model(
                        input_ids=xhat_ids,
                        attention_mask=attn_mask,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    logits = out.logits
                    hidden_last = out.hidden_states[-1]

                # ---- (1-A) fusion at masked positions (full vocab) ----
                if fusion is not None:
                    # apply fusion only at masked positions
                    for idx, (b, p) in enumerate(mask_positions):
                        fused, _gate = fusion(
                            logits[b, p],  # [V]
                            hidden_last[b, p],  # [D]
                            q_list[idx].to(device),  # [V]
                            ber=ber,
                        )
                        logits[b, p] = fused

                # loss
                loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

                optim.zero_grad()
                loss.backward()
                optim.step()

                if step % cfg.log_every == 0:
                    print(
                        f"epoch={epoch} step={step} loss={loss.item():.4f} guide={cfg.guide} fusion={cfg.use_fusion}"
                    )
                wandb.log({"step": step, "train_loss": loss.item()})
                step += 1

                if batch_idx == len(dl) - 1:
                    pred_ids = xhat_ids.clone()
                    pred_tokens = logits.argmax(dim=-1)
                    for b, p in mask_positions:
                        pred_ids[b, p] = pred_tokens[b, p]

                    sample_indices = [0, 1] if input_ids.size(0) > 1 else [0]
                    for sidx in sample_indices:
                        attn = attn_mask[sidx].bool()
                        clean_list = x_clean[sidx][attn].tolist()
                        corrupted_list = xbar[sidx][attn].tolist()
                        masked_list = xhat_ids[sidx][attn].tolist()
                        restored_list = pred_ids[sidx][attn].tolist()

                        clean_text = tokenizer.decode(
                            clean_list, skip_special_tokens=True
                        )
                        corrupted_text = tokenizer.decode(
                            corrupted_list, skip_special_tokens=True
                        )
                        masked_text = tokenizer.decode(
                            masked_list, skip_special_tokens=False
                        )
                        restored_text = tokenizer.decode(
                            restored_list, skip_special_tokens=True
                        )

                        from channel_corrupt import ber_from_snr

                        ber_estimate, _ = ber_from_snr(snr_db_used, cfg.modulation)
                        writer.writerow(
                            {
                                "epoch": epoch,
                                "sample_idx": sidx,
                                "snr_db": round(snr_db_used, 6),
                                "modulation": cfg.modulation,
                                "ber": round(ber_estimate, 8),
                                "clean_text": clean_text,
                                "clean_token_ids": json.dumps(clean_list),
                                "corrupted_token_ids": json.dumps(corrupted_list),
                                "corrupted_text": corrupted_text,
                                "masked_token_ids": json.dumps(masked_list),
                                "masked_text": masked_text,
                                "restored_token_ids": json.dumps(restored_list),
                                "restored_text": restored_text,
                            }
                        )
                    f.flush()

            # Validation at end of epoch
            print(f"\nRunning validation for epoch {epoch}...")
            val_ce_losses = []
            val_f1_scores = []

            model.eval()
            with torch.no_grad():
                for val_batch_texts in val_dl:
                    val_input_ids, val_attn_mask = collate_texts(
                        tokenizer, list(val_batch_texts), cfg.max_len
                    )
                    val_input_ids = val_input_ids.to(device)
                    val_attn_mask = val_attn_mask.to(device)

                    # Channel transmission
                    val_snr = (
                        cfg.snr_db
                        if cfg.snr_db is not None
                        else random.uniform(-5.0, 10.0)
                    )
                    val_xbar, val_llr = transmit_ids_awgn_llr(
                        val_input_ids, val_snr, cfg.modulation, m_bits, tokenizer
                    )

                    # Masking
                    val_xhat, _ = make_masked_input(
                        val_xbar,
                        val_llr,
                        tokenizer,
                        strategy=cfg.mask_strategy,
                        mask_rate=cfg.mask_rate,
                        rel_tau=cfg.rel_tau,
                        rel_mode=cfg.real_mode,
                    )

                    mask_positions = (
                        (val_xhat == tokenizer.mask_token_id)
                        .nonzero(as_tuple=False)
                        .tolist()
                    )

                    if len(mask_positions) == 0:
                        val_out = model(
                            input_ids=val_xhat,
                            attention_mask=val_attn_mask,
                            output_hidden_states=False,
                            return_dict=True,
                        )
                        val_logits = val_out.logits
                    else:
                        from channel_corrupt import ber_from_snr

                        val_ber, _ = ber_from_snr(val_snr, cfg.modulation)
                        q_list = build_q_list_for_masks(
                            vocab_size=model.config.vocab_size,
                            xbar_ids=val_xbar,
                            mask_positions=[(b, p) for b, p in mask_positions],
                            ber=val_ber,
                            lut_u16=lut,
                        )

                        if cfg.guide == "soft_embed":
                            inputs_embeds = make_soft_inputs_embeds(
                                model,
                                val_xhat,
                                [(b, p) for b, p in mask_positions],
                                q_list,
                            )
                            val_out = model(
                                inputs_embeds=inputs_embeds,
                                attention_mask=val_attn_mask,
                                output_hidden_states=True,
                                return_dict=True,
                            )
                            val_logits = val_out.logits
                            val_hidden_last = val_out.hidden_states[-1]

                        elif cfg.guide == "attn_bias":
                            key_bias = build_key_bias_from_q(
                                input_ids=val_xhat,
                                mask_positions=[(b, p) for b, p in mask_positions],
                                q_list=q_list,
                                ber=val_ber,
                                mode="rmax",
                                beta=cfg.attn_beta_init,
                            ).to(device)

                            enc_out = bert_bias_wrapper(
                                input_ids=val_xhat,
                                attention_mask=val_attn_mask,
                                key_bias=key_bias,
                                output_hidden_states=True,
                                return_dict=True,
                            )
                            val_hidden_last = enc_out["last_hidden_state"]
                            val_logits = model.cls(val_hidden_last)

                        else:
                            val_out = model(
                                input_ids=val_xhat,
                                attention_mask=val_attn_mask,
                                output_hidden_states=True,
                                return_dict=True,
                            )
                            val_logits = val_out.logits
                            val_hidden_last = val_out.hidden_states[-1]

                        if fusion is not None:
                            for idx, (b, p) in enumerate(mask_positions):
                                fused, _gate = fusion(
                                    val_logits[b, p],
                                    val_hidden_last[b, p],
                                    q_list[idx].to(device),
                                    ber=val_ber,
                                )
                                val_logits[b, p] = fused

                    val_argmax = val_logits.argmax(dim=-1)
                    val_pred = val_xhat.clone()
                    mask = val_xhat == tokenizer.mask_token_id
                    val_pred[mask] = val_argmax[mask]

                    # Compute metrics for this batch
                    val_ce, val_f1 = compute_validation_metrics(
                        val_input_ids.cpu(),
                        val_pred.cpu(),
                        tokenizer,
                        loss_fn,
                        model,
                        device,
                    )
                    val_ce_losses.append(val_ce)
                    val_f1_scores.append(val_f1)

            model.train()

            # Average metrics across all validation batches
            avg_val_ce = sum(val_ce_losses) / len(val_ce_losses)
            avg_val_f1 = sum(val_f1_scores) / len(val_f1_scores)

            print(
                f"Epoch {epoch} - Val CE Loss: {avg_val_ce:.4f}, Val F1: {avg_val_f1:.4f}"
            )
            wandb.log(
                {
                    "epoch": epoch,
                    "val_ce_loss": avg_val_ce,
                    "val_f1_score": avg_val_f1,
                }
            )
            save_dict = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
            }
            if fusion is not None:
                save_dict["fusion_state_dict"] = fusion.state_dict()
            torch.save(save_dict, model_path)
            print(f"Epoch {epoch} checkpoint saved to {model_path}")

    print(f"Done. CSV saved to {csv_path}")
    wandb.finish()


if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)
