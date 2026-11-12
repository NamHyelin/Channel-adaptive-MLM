import argparse
import math
import os
import json
import random
import datetime
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
from bert_score import score as bert_score
from scipy.spatial.distance import cosine as scipy_cosine

from transformers import AutoTokenizer, AutoModelForMaskedLM

from config import Config
from data_coco import iter_coco_captions
from channel_corrupt import transmit_ids_awgn_llr, make_masked_input, ber_from_snr
from q_model import popcount_u16_lut, q_full_vocab_from_llr
from guidance import (
    make_soft_inputs_embeds,
)
from fusion import AdaptiveFusion


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_test_args() -> Config:
    p = argparse.ArgumentParser()
    for k, v in Config().__dict__.items():
        arg = "--" + k.replace("_", "-")
        if k == "snr_db":
            p.add_argument(arg, type=str, default=None)
        elif k == "test_texts":
            p.add_argument(arg, action="append", default=None)
        elif isinstance(v, bool):
            p.add_argument(arg, action="store_true" if not v else "store_false")
        else:
            p.add_argument(arg, type=type(v), default=v)
    ns = p.parse_args()
    data = vars(ns)
    raw = data.get("snr_db")
    if raw is None:
        data["snr_db"] = None
    else:
        s = str(raw).strip().lower()
        data["snr_db"] = None if s in {"none", "null", ""} else float(raw)
    data["test_texts"] = data.get("test_texts") or []
    return Config(**data)


def collate_texts(tokenizer, texts, max_len: int):
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    return enc["input_ids"], enc["attention_mask"]


def build_q_list_for_masks(
    llr: torch.Tensor,
    vocab_size: int,
    xbar_ids: torch.Tensor,
    mask_positions: list[tuple[int, int]],
    ber: float,
    lut_u16: torch.Tensor,
):
    q_list = []
    for b, pos in mask_positions:
        xbar = int(xbar_ids[b, pos].item())
        if xbar < 0 or xbar >= vocab_size:
            q = torch.full(
                (vocab_size,),
                1.0 / vocab_size,
                device=xbar_ids.device,
                dtype=torch.float32,
            )
        else:
            # q = q_full_vocab_from_ber(vocab_size, xbar, ber, xbar_ids.device, lut_u16)
            llr_at_pos = llr[b, pos]
            q = q_full_vocab_from_llr(
                llr_at_pos, vocab_size, xbar_ids.device, temp=1
            )  # (5/llr_at_pos.abs().mean())

        q_list.append(q)
    return q_list


def sanitize_for_decode(ids: torch.Tensor, vocab_size: int, unk_id: int) -> list[int]:
    arr = ids.tolist()
    return [t if 0 <= t < vocab_size else unk_id for t in arr]


def topk_index_value_pairs(values: torch.Tensor, k: int = 5) -> list[dict[str, float]]:
    topk_vals, topk_indices = torch.topk(values, k=min(k, values.numel()))
    return [
        {"index": int(idx.item()), "value": float(val.item())}
        for idx, val in zip(topk_indices, topk_vals)
    ]


def js_divergence_from_probs(
    p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12
) -> float:
    p_safe = torch.clamp(p, min=eps)
    q_safe = torch.clamp(q, min=eps)
    p_safe = p_safe / p_safe.sum()
    q_safe = q_safe / q_safe.sum()
    m = 0.5 * (p_safe + q_safe)
    kl_pm = torch.sum(p_safe * torch.log(p_safe / m))
    kl_qm = torch.sum(q_safe * torch.log(q_safe / m))
    return float(0.5 * (kl_pm + kl_qm))


def pearson_corr(v1: torch.Tensor, v2: torch.Tensor, eps: float = 1e-12) -> float:
    x = v1 - v1.mean()
    y = v2 - v2.mean()
    denom = torch.sqrt(torch.sum(x * x)) * torch.sqrt(torch.sum(y * y))
    if float(denom.item()) <= eps:
        return 0.0
    return float(torch.sum(x * y) / denom)


def mean_or_none(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def forward_with_guidance(
    model,
    fusion,
    tokenizer,
    xhat_ids,
    xbar_ids,
    attn_mask,
    snr_db_used,
    modulation,
    lut,
    guide,
    attn_beta_init,
):
    mask_positions = (
        (xhat_ids == tokenizer.mask_token_id).nonzero(as_tuple=False).tolist()
    )  # mask token 위치 [(batch_idx, seq_pos), ...] 리스트

    if len(mask_positions) == 0:
        out = model(
            input_ids=xhat_ids,
            attention_mask=attn_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        logits = out.logits
        hidden_last = out.hidden_states[-1]
        return logits, hidden_last

    ber, _ = ber_from_snr(snr_db_used, modulation)
    q_list = build_q_list_for_masks(
        vocab_size=model.config.vocab_size,
        xbar_ids=xbar_ids,
        mask_positions=[(b, p) for b, p in mask_positions],
        ber=ber,
        lut_u16=lut,
    )

    if guide == "soft_embed":
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

    # elif guide == "attn_bias":
    #     key_bias = build_key_bias_from_q(
    #         input_ids=xhat_ids,
    #         mask_positions=[(b, p) for b, p in mask_positions],
    #         q_list=q_list,
    #         ber=ber,
    #         mode="rmax",
    #         beta=attn_beta_init,
    #     ).to(xhat_ids.device)

    #     enc_out = bert_bias_wrapper(
    #         input_ids=xhat_ids,
    #         attention_mask=attn_mask,
    #         key_bias=key_bias,
    #         output_hidden_states=True,
    #         return_dict=True,
    #     )
    #     hidden_last = enc_out["last_hidden_state"]
    #     logits = model.cls(hidden_last)

    else:
        out = model(
            input_ids=xhat_ids,
            attention_mask=attn_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        logits = out.logits
        hidden_last = out.hidden_states[-1]

    if fusion is not None:
        eps = 1e-12

        # Hill g(ber) = gmin + (gmax - gmin) * ber^k / (ber^k + tau^k)
        gmin = 0.667621374130249
        gmax = 0.9978040158748627
        tau = 0.041547033935785294
        k = 2.0921876430511475

        ber_safe = max(float(ber), eps)
        num = ber_safe**k
        den = num + (tau**k)
        g = gmin + (gmax - gmin) * (num / den)

        for idx, (b, p) in enumerate(mask_positions):
            q = q_list[idx].to(xhat_ids.device)
            logits[b, p] = logits[b, p] + g * torch.log(q + eps)
            # print(f"Gate values: {_gate}, Scale: {_scale.item():.4f}, Temp: {_temp.item():.4f}")
    return logits, hidden_last


def test_snr():
    cfg = parse_test_args()
    import csv

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    test_modulation = (
        cfg.test_modulation if cfg.test_modulation is not None else cfg.modulation
    )
    test_mask_strategy = (
        cfg.test_mask_strategy
        if cfg.test_mask_strategy is not None
        else cfg.mask_strategy
    )
    test_mask_rate = (
        cfg.test_mask_rate if cfg.test_mask_rate is not None else cfg.mask_rate
    )
    test_rel_tau = cfg.test_rel_tau if cfg.test_rel_tau is not None else cfg.rel_tau
    test_guide = cfg.test_guide if cfg.test_guide is not None else cfg.guide
    test_use_fusion = (
        cfg.test_use_fusion if cfg.test_use_fusion is not None else cfg.use_fusion
    )
    test_rel_mode = (
        cfg.test_rel_mode if cfg.test_rel_mode is not None else cfg.real_mode
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForMaskedLM.from_pretrained(
        cfg.model_name, revision="refs/pr/1", use_safetensors=True
    ).to(device)

    fusion = test_use_fusion

    model_path = cfg.test_model_path
    if model_path:
        ckpt = torch.load(model_path, map_location=device)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))

    model.eval()

    snr_choices = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    texts = list(cfg.test_texts or [])
    if not texts:
        captions = list(
            iter_coco_captions(
                cfg.dataset_name, cfg.split, cfg.text_field, cfg.test_num_samples
            )
        )
        texts = captions[: cfg.test_num_samples]
    if not texts:
        raise ValueError("No test texts available for test_snr()")

    mmdd = datetime.datetime.now().strftime("%m%d")
    csv_path = os.path.join(
        "C:/token_bit_transformer/results/",
        mmdd,
        f"{cfg.test_csv_name}.csv",
    )
    csv_dir = os.path.dirname(csv_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)

    # path for aggregated per-SNR metrics (will be periodically checkpointed)
    snr_npy_path = os.path.join(
        "C:/token_bit_transformer/results/",
        mmdd,
        f"{cfg.test_csv_name}.npy",
    )

    csv_columns = [
        "snr_db",
        "sample_idx",
        "clean_text",
        "errored_text",
        "masked_text",
        "restored_text",
        "f1_score_text",
        "cosine_similarity",
        "bert_score",
        "q_top10",
        "pre_fusion_top10",
        "post_fusion_top10",
    ]
    csv_rows: list[dict[str, str | int]] = []
    # keep saved entries (max two per SNR) and overwrite CSV each batch
    saved_entries: list[dict[str, str | int]] = []

    def flush_csv_rows() -> None:
        nonlocal csv_rows, saved_entries
        # overwrite the CSV with the current saved entries (always write header)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_columns)
            writer.writeheader()
            if saved_entries:
                writer.writerows(saved_entries)
        csv_rows = []

    vocab_size = model.config.vocab_size
    unk_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else 100
    loss_fn = nn.CrossEntropyLoss()
    save_counts = {snr: 0 for snr in snr_choices}
    metric_rows = []
    # how often to checkpoint (in number of processed samples)
    checkpoint_every = int(getattr(cfg, "checkpoint_every", 100))

    def _save_snr_npy_atomic(path: str, rows: list[dict]):
        # compute grouped means and save atomically to avoid partial writes
        snr_metric_rows_grouped = []
        for snr in snr_choices:
            rows_for_snr = [r for r in rows if r["snr_db"] == snr]
            if rows_for_snr:
                snr_metric_rows_grouped.append(
                    {
                        "snr_db": snr,
                        "mean_f1_score_text": float(
                            np.mean([r["f1_score_text"] for r in rows_for_snr])
                        ),
                        "mean_cosine_similarity": float(
                            np.mean([r["cosine_similarity"] for r in rows_for_snr])
                        ),
                        "mean_bert_score": float(
                            np.mean([r["bert_score"] for r in rows_for_snr])
                        ),
                    }
                )

        if not snr_metric_rows_grouped:
            return

        arr = np.array(
            [
                [
                    r["snr_db"],
                    r["mean_f1_score_text"],
                    r["mean_cosine_similarity"],
                    r["mean_bert_score"],
                ]
                for r in snr_metric_rows_grouped
            ],
            dtype=np.float32,
        )
        tmp_path = path + ".tmp"
        np.save(tmp_path, arr)
        try:
            os.replace(tmp_path, path)
        except Exception:
            # best-effort: if replace fails, try a normal save
            np.save(path, arr)

    print(f"\nfusion: {test_use_fusion}, guide: {test_guide}")
    print(f"Running {cfg.test_iteration_num} iteration(s) through the data loader")

    sample_idx_global = 0
    # Run grouped by SNR: for each SNR in snr_choices, run cfg.test_iteration_num
    # passes over the dataset (so total runs per SNR = batch_size * iterations).
    total_runs_per_snr = len(texts) * cfg.test_iteration_num
    for snr_db_used in snr_choices:
        print(f"\n=== SNR {snr_db_used} dB ===")
        for iteration in range(cfg.test_iteration_num):
            print(
                f"\n--- Iteration {iteration + 1}/{cfg.test_iteration_num} (SNR {snr_db_used} dB) ---"
            )
            for sample_idx, clean_text_in in enumerate(texts):
                print(f"Using SNR (dB): {snr_db_used:.2f}")

                input_ids, attn_mask = collate_texts(
                    tokenizer, [clean_text_in], cfg.max_len
                )
                input_ids = input_ids.to(device)
                attn_mask = attn_mask.to(device)

                m_bits = int(math.ceil(math.log2(model.config.vocab_size)))
                lut = popcount_u16_lut(device)

                xbar, llr_tok = transmit_ids_awgn_llr(
                    input_ids, snr_db_used, test_modulation, m_bits, tokenizer
                )

                if cfg.test_mask:
                    xhat_ids, _ = make_masked_input(
                        xbar,
                        llr_tok,
                        tokenizer,
                        strategy=test_mask_strategy,
                        mask_rate=test_mask_rate,
                        rel_tau=test_rel_tau,
                        rel_mode=test_rel_mode,
                    )
                else:
                    xhat_ids = xbar.clone()

                mask_positions = (
                    (xhat_ids == tokenizer.mask_token_id)
                    .nonzero(as_tuple=False)
                    .tolist()
                )

                if len(mask_positions) == 0:
                    out = model(
                        input_ids=xhat_ids,
                        attention_mask=attn_mask,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    logits = out.logits
                else:
                    ber, _ = ber_from_snr(snr_db_used, test_modulation)
                    q_list = build_q_list_for_masks(
                        llr=llr_tok,
                        vocab_size=model.config.vocab_size,
                        xbar_ids=xbar,
                        mask_positions=[(b, p) for b, p in mask_positions],
                        ber=ber,
                        lut_u16=lut,
                    )

                    if test_guide == "soft_embed":
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
                    else:
                        out = model(
                            input_ids=xhat_ids,
                            attention_mask=attn_mask,
                            output_hidden_states=True,
                            return_dict=True,
                        )
                        logits = out.logits

                    if test_use_fusion is True:
                        eps = 1e-12
                        # Hill g(ber) = gmin + (gmax - gmin) * ber^k / (ber^k + tau^k)
                        gmin = 0.667621374130249
                        gmax = 0.9978040158748627
                        tau = 0.041547033935785294
                        k = 2.0921876430511475

                        ber_safe = max(float(ber), eps)
                        num = ber_safe**k
                        den = num + (tau**k)
                        g = gmin + (gmax - gmin) * (num / den)

                        for idx, (b, p) in enumerate(mask_positions):
                            q = q_list[idx].to(xhat_ids.device)
                            logits[b, p] = logits[b, p] + g * torch.log(q + eps)

                pred_ids = torch.softmax(logits, dim=-1).argmax(dim=-1)
                restored = xhat_ids.clone()
                restored[xhat_ids == tokenizer.mask_token_id] = pred_ids[
                    xhat_ids == tokenizer.mask_token_id
                ]

                attn = attn_mask[0].bool()
                clean_ids = input_ids[0][attn]
                corrupted_ids = xbar[0][attn]
                masked_ids = xhat_ids[0][attn]
                restored_ids = restored[0][attn]

                clean_list = clean_ids.tolist()
                corrupted_list = sanitize_for_decode(corrupted_ids, vocab_size, unk_id)
                masked_list = sanitize_for_decode(masked_ids, vocab_size, unk_id)
                restored_list = sanitize_for_decode(restored_ids, vocab_size, unk_id)

                clean_text = tokenizer.decode(clean_list, skip_special_tokens=True)
                corrupted_text = tokenizer.decode(
                    corrupted_list, skip_special_tokens=True
                )
                masked_text = tokenizer.decode(masked_list, skip_special_tokens=False)
                restored_text = tokenizer.decode(
                    restored_list, skip_special_tokens=True
                )

                _ = loss_fn(
                    logits[0].view(-1, logits.size(-1)), input_ids[0].view(-1)
                ).item()

                try:
                    clean_tokens = tokenizer.tokenize(clean_text)
                    restored_tokens = tokenizer.tokenize(restored_text)
                    max_len = max(len(clean_tokens), len(restored_tokens))
                    if max_len == 0:
                        f1_text = 0.0
                    else:
                        clean_token_ids = tokenizer.convert_tokens_to_ids(
                            clean_tokens[:max_len]
                        )
                        restored_token_ids = tokenizer.convert_tokens_to_ids(
                            restored_tokens[:max_len]
                        )
                        clean_token_ids += [tokenizer.pad_token_id] * (
                            max_len - len(clean_token_ids)
                        )
                        restored_token_ids += [tokenizer.pad_token_id] * (
                            max_len - len(restored_token_ids)
                        )
                        f1_text = float(
                            f1_score(
                                clean_token_ids,
                                restored_token_ids,
                                average="macro",
                                zero_division=0,
                            )
                        )
                except Exception:
                    f1_text = 0.0

                try:
                    with torch.no_grad():
                        clean_encoded = tokenizer(
                            clean_text,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=512,
                        ).to(device)
                        restored_encoded = tokenizer(
                            restored_text,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=512,
                        ).to(device)
                        clean_outputs = model(
                            **clean_encoded, output_hidden_states=True
                        )
                        restored_outputs = model(
                            **restored_encoded, output_hidden_states=True
                        )
                        clean_embed = (
                            clean_outputs.hidden_states[-1][:, 0, :]
                            .squeeze()
                            .cpu()
                            .numpy()
                        )
                        restored_embed = (
                            restored_outputs.hidden_states[-1][:, 0, :]
                            .squeeze()
                            .cpu()
                            .numpy()
                        )
                        cosine_sim = float(
                            1 - scipy_cosine(clean_embed, restored_embed)
                        )
                except Exception:
                    cosine_sim = 0.0

                try:
                    _, _, bert_f1 = bert_score([restored_text], [clean_text], lang="en")
                    bert_f1_val = float(bert_f1.mean().item())
                except Exception:
                    bert_f1_val = 0.0

                q_top10_rows = []
                pre_fusion_top10_rows = []
                post_fusion_top10_rows = []
                pre_fusion_logits = logits.detach().clone()
                post_fusion_logits = logits

                if len(mask_positions) > 0:
                    for idx, (b, p) in enumerate(mask_positions):
                        q = q_list[idx].detach().cpu() if "q_list" in locals() else None
                        if q is None:
                            continue
                        clean_token_id = int(input_ids[b, p].item())
                        clean_token = tokenizer.convert_ids_to_tokens(clean_token_id)
                        clean_q = float(q[clean_token_id].item())
                        clean_rank = int((q > q[clean_token_id]).sum().item()) + 1

                        topk_vals, topk_indices = torch.topk(q, k=min(10, q.numel()))
                        topk_tokens = [
                            tokenizer.convert_ids_to_tokens(int(token_id.item()))
                            for token_id in topk_indices
                        ]
                        topk_scores = [f"{float(v.item()):.3f}" for v in topk_vals]
                        q_top10_rows.append(
                            {
                                "mask_pos": int(p),
                                "clean_token": clean_token,
                                "clean_rank": clean_rank,
                                "clean_q": f"{clean_q:.3f}",
                                "top10": list(zip(topk_tokens, topk_scores)),
                            }
                        )

                        pre_fusion_softmax = torch.softmax(
                            pre_fusion_logits[b, p], dim=-1
                        )
                        pre_vals, pre_indices = torch.topk(
                            pre_fusion_softmax, k=min(10, pre_fusion_softmax.numel())
                        )
                        pre_tokens = [
                            tokenizer.convert_ids_to_tokens(int(token_id.item()))
                            for token_id in pre_indices
                        ]
                        pre_scores = [f"{float(v.item()):.3f}" for v in pre_vals]
                        pre_fusion_top10_rows.append(
                            {
                                "mask_pos": int(p),
                                "top10": list(zip(pre_tokens, pre_scores)),
                            }
                        )

                        post_fusion_softmax = torch.softmax(
                            post_fusion_logits[b, p], dim=-1
                        )
                        post_vals, post_indices = torch.topk(
                            post_fusion_softmax, k=min(10, post_fusion_softmax.numel())
                        )
                        post_tokens = [
                            tokenizer.convert_ids_to_tokens(int(token_id.item()))
                            for token_id in post_indices
                        ]
                        post_scores = [f"{float(v.item()):.3f}" for v in post_vals]
                        post_fusion_top10_rows.append(
                            {
                                "mask_pos": int(p),
                                "top10": list(zip(post_tokens, post_scores)),
                            }
                        )

                metric_rows.append(
                    {
                        "snr_db": snr_db_used,
                        "f1_score_text": f1_text,
                        "cosine_similarity": cosine_sim,
                        "bert_score": bert_f1_val,
                    }
                )

                if save_counts.get(snr_db_used, 0) < 2:
                    saved_entries.append(
                        {
                            "snr_db": snr_db_used,
                            "sample_idx": sample_idx + 1,
                            "clean_text": clean_text,
                            "errored_text": corrupted_text,
                            "masked_text": masked_text,
                            "restored_text": restored_text,
                            "f1_score_text": f"{f1_text:.4f}",
                            "cosine_similarity": f"{cosine_sim:.4f}",
                            "bert_score": f"{bert_f1_val:.4f}",
                            "q_top10": json.dumps(q_top10_rows, ensure_ascii=False),
                            "pre_fusion_top10": json.dumps(
                                pre_fusion_top10_rows, ensure_ascii=False
                            ),
                            "post_fusion_top10": json.dumps(
                                post_fusion_top10_rows, ensure_ascii=False
                            ),
                        }
                    )
                    save_counts[snr_db_used] = save_counts.get(snr_db_used, 0) + 1

                    print(f"\nSNR {snr_db_used:.2f} | Sample {sample_idx_global + 1}")
                    print(f"F1: {f1_text:.4f}")
                    print(f"Clean:   {clean_text}")
                    print(f"Errored: {corrupted_text}")
                    print(f"Masked:  {masked_text}")
                    print(f"Restored:{restored_text}")

                    # after every batch: overwrite CSV and save aggregated .npy
                    flush_csv_rows()
                    _save_snr_npy_atomic(snr_npy_path, metric_rows)

                    # periodic checkpoint retained for compatibility (no-op here because
                    # we already save every batch)
                    if (
                        checkpoint_every > 0
                        and (sample_idx + 1) % checkpoint_every == 0
                    ):
                        pass

                    sample_idx_global += 1

    # final flush and final snapshot
    flush_csv_rows()
    _save_snr_npy_atomic(snr_npy_path, metric_rows)

    # Plot aggregated per-SNR metrics and save plot image
    try:
        import matplotlib.pyplot as plt

        snr_metric_rows_grouped = []
        for snr in snr_choices:
            rows_for_snr = [r for r in metric_rows if r["snr_db"] == snr]
            if rows_for_snr:
                snr_metric_rows_grouped.append(
                    {
                        "snr_db": snr,
                        "mean_f1_score_text": float(
                            np.mean([r["f1_score_text"] for r in rows_for_snr])
                        ),
                        "mean_cosine_similarity": float(
                            np.mean([r["cosine_similarity"] for r in rows_for_snr])
                        ),
                        "mean_bert_score": float(
                            np.mean([r["bert_score"] for r in rows_for_snr])
                        ),
                    }
                )

        if snr_metric_rows_grouped:
            snr_metric_rows_grouped.sort(key=lambda x: x["snr_db"])
            snrs = [r["snr_db"] for r in snr_metric_rows_grouped]
            f1s = [r["mean_f1_score_text"] for r in snr_metric_rows_grouped]
            coses = [r["mean_cosine_similarity"] for r in snr_metric_rows_grouped]
            berts = [r["mean_bert_score"] for r in snr_metric_rows_grouped]

            plt.figure(figsize=(8, 6))
            plt.plot(snrs, f1s, marker="o", label="F1")
            plt.plot(snrs, coses, marker="x", label="Cosine")
            plt.plot(snrs, berts, marker="s", label="BERT")
            plt.xlabel("SNR (dB)")
            plt.ylabel("Metric")
            plt.title(f"{cfg.test_csv_name} - aggregated metrics by SNR")
            plt.legend()
            plt.grid(True)
            plot_path = os.path.join(csv_dir, f"{cfg.test_csv_name}_metrics.png")
            plt.savefig(plot_path)
            plt.close()
            print(f"Saved plot to {plot_path}")
    except Exception as e:
        print(f"Could not plot results: {e}")


def test_snr_nopropos():
    cfg = parse_test_args()
    import csv

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    test_modulation = (
        cfg.test_modulation if cfg.test_modulation is not None else cfg.modulation
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    snr_choices = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    texts = list(cfg.test_texts or [])
    if not texts:
        captions = list(
            iter_coco_captions(
                cfg.dataset_name, cfg.split, cfg.text_field, cfg.test_num_samples
            )
        )
        texts = captions[: cfg.test_num_samples]
    if not texts:
        raise ValueError("No test texts available for test_snr_nopropos()")

    mmdd = datetime.datetime.now().strftime("%m%d")
    csv_path = os.path.join(
        "C:/token_bit_transformer/results/",
        mmdd,
        f"{cfg.test_csv_name}_nopropos.csv",
    )
    csv_dir = os.path.dirname(csv_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)

    snr_npy_path = os.path.join(
        "C:/token_bit_transformer/results/",
        mmdd,
        f"{cfg.test_csv_name}_nopropos.npy",
    )

    csv_columns = [
        "snr_db",
        "sample_idx",
        "clean_text",
        "errored_text",
        "cosine_similarity",
    ]
    csv_rows: list[dict[str, str | int]] = []
    saved_entries: list[dict[str, str | int]] = []

    def flush_csv_rows() -> None:
        nonlocal csv_rows, saved_entries
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_columns)
            writer.writeheader()
            if saved_entries:
                writer.writerows(saved_entries)
        csv_rows = []

    vocab_size = tokenizer.vocab_size
    unk_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else 100
    save_counts = {snr: 0 for snr in snr_choices}
    metric_rows = []
    checkpoint_every = int(getattr(cfg, "checkpoint_every", 100))

    def _save_snr_npy_atomic(path: str, rows: list[dict]):
        snr_metric_rows_grouped = []
        for snr in snr_choices:
            rows_for_snr = [r for r in rows if r["snr_db"] == snr]
            if rows_for_snr:
                snr_metric_rows_grouped.append(
                    {
                        "snr_db": snr,
                        "mean_cosine_similarity": float(
                            np.mean([r["cosine_similarity"] for r in rows_for_snr])
                        ),
                    }
                )

        if not snr_metric_rows_grouped:
            return

        arr = np.array(
            [
                [r["snr_db"], r["mean_cosine_similarity"]]
                for r in snr_metric_rows_grouped
            ],
            dtype=np.float32,
        )
        tmp_path = path + ".tmp"
        np.save(tmp_path, arr)
        try:
            os.replace(tmp_path, path)
        except Exception:
            np.save(path, arr)

    print("\n[NoPropos] Running without MLM restoration")
    print(f"Running {cfg.test_iteration_num} iteration(s) through the data loader")

    sample_idx_global = 0
    total_runs_per_snr = len(texts) * cfg.test_iteration_num
    for snr_db_used in snr_choices:
        print(f"\n=== SNR {snr_db_used} dB ===")
        for iteration in range(cfg.test_iteration_num):
            print(
                f"\n--- Iteration {iteration + 1}/{cfg.test_iteration_num} (SNR {snr_db_used} dB) ---"
            )
            for sample_idx, clean_text_in in enumerate(texts):
                input_ids, attn_mask = collate_texts(
                    tokenizer, [clean_text_in], cfg.max_len
                )
                input_ids = input_ids.to(device)
                attn_mask = attn_mask.to(device)

                m_bits = int(math.ceil(math.log2(vocab_size)))
                xbar, _ = transmit_ids_awgn_llr(
                    input_ids, snr_db_used, test_modulation, m_bits, tokenizer
                )

                attn = attn_mask[0].bool()
                clean_ids = input_ids[0][attn]
                corrupted_ids = xbar[0][attn]

                clean_list = clean_ids.tolist()
                corrupted_list = sanitize_for_decode(corrupted_ids, vocab_size, unk_id)

                clean_text = tokenizer.decode(clean_list, skip_special_tokens=True)
                corrupted_text = tokenizer.decode(
                    corrupted_list, skip_special_tokens=True
                )

                try:
                    vec = TfidfVectorizer()
                    mat = vec.fit_transform([clean_text, corrupted_text])
                    cosine_sim = float(
                        sklearn_cosine_similarity(mat[0:1], mat[1:2])[0, 0]
                    )
                except Exception:
                    cosine_sim = 0.0

                metric_rows.append(
                    {
                        "snr_db": snr_db_used,
                        "cosine_similarity": cosine_sim,
                    }
                )

                if save_counts.get(snr_db_used, 0) < 2:
                    saved_entries.append(
                        {
                            "snr_db": snr_db_used,
                            "sample_idx": sample_idx + 1,
                            "clean_text": clean_text,
                            "errored_text": corrupted_text,
                            "cosine_similarity": f"{cosine_sim:.4f}",
                        }
                    )
                    save_counts[snr_db_used] = save_counts.get(snr_db_used, 0) + 1

                    print(f"\nSNR {snr_db_used:.2f} | Sample {sample_idx_global + 1}")
                    print(f"Cosine: {cosine_sim:.4f}")
                    print(f"Clean:   {clean_text}")
                    print(f"Errored: {corrupted_text}")

                    flush_csv_rows()
                    _save_snr_npy_atomic(snr_npy_path, metric_rows)

                    if (
                        checkpoint_every > 0
                        and (sample_idx + 1) % checkpoint_every == 0
                    ):
                        pass

                    sample_idx_global += 1

    flush_csv_rows()
    _save_snr_npy_atomic(snr_npy_path, metric_rows)



def test_input_fun():
    # test 할때 필요한 변수들은 대부분 config.py 에서 test_ 로 시작하는 것들
    cfg = parse_test_args()

    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # snr, modulation 지정해서 실험
    test_snr_db = cfg.test_snr_db if cfg.test_snr_db is not None else cfg.snr_db
    test_modulation = (
        cfg.test_modulation if cfg.test_modulation is not None else cfg.modulation
    )

    test_guide = cfg.test_guide if cfg.test_guide is not None else cfg.guide
    test_use_fusion = (
        cfg.test_use_fusion if cfg.test_use_fusion is not None else cfg.use_fusion
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForMaskedLM.from_pretrained(
        cfg.model_name, revision="refs/pr/1", use_safetensors=True
    ).to(device)

    # output fusion 하는 모델
    fusion = (
        AdaptiveFusion(model.config.hidden_size).to(device) if test_use_fusion else None
    )
    if fusion is not None:
        fusion.eval()

    # model_path = cfg.test_model_path
    # if model_path:
    #     ckpt = torch.load(model_path, map_location=device)
    #     model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    #     if fusion is not None and "fusion_state_dict" in ckpt:
    #         fusion.load_state_dict(ckpt["fusion_state_dict"])

    model.eval()

    # text 지정할 수 있고, 그냥 coco captions 에서 가져오고 싶으면 texts=[] 로 하면 됨
    texts = [
        "A cat is sitting on the mat.",
        # "A cat is sitting on the mat.",
        # "A cat is sitting on the mat.",
        # "A cat is sitting on the mat.",
        # "A cat is sitting on the mat.",
    ]
    if not texts:
        captions = list(
            iter_coco_captions(
                cfg.dataset_name, cfg.split, cfg.text_field, cfg.test_num_samples
            )
        )
        texts = captions[: cfg.test_num_samples]

    input_ids, attn_mask = collate_texts(tokenizer, texts, cfg.max_len)
    input_ids = input_ids.to(device)
    attn_mask = attn_mask.to(device)

    m_bits = int(
        math.ceil(math.log2(model.config.vocab_size))
    )  # vocab_size 를 표현할 수 있는 bit 수

    print(f"\nfusion: {test_use_fusion}, guide: {test_guide}")
    snr_db_used = test_snr_db if test_snr_db is not None else random.uniform(-5.0, 10.0)
    print(f"Using SNR (dB): {snr_db_used:.2f}")

    # x가 보내는 token id, noisy 채널 거쳐서 받은 errored tokein id가 xbar, llr_tok은 각 bit별 LLR 값.
    xbar, llr_tok = transmit_ids_awgn_llr(  # channel_corrupt.py 에 있음
        input_ids, snr_db_used, test_modulation, m_bits, tokenizer
    )
    xbar_texts = [
        "A cat is sitting on the mat.",
        # "A cat is swimming on the mat.",
        # "A cat is dinosaur on the mat.",
        # "A cat is [MASK] on the mat.",
        # "A [MASK] is sitting on the mat.",
    ]
    xbar, _ = collate_texts(tokenizer, xbar_texts, cfg.max_len)
    xbar = xbar.to(device)

    vocab_size = model.config.vocab_size
    unk_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else 100

    masked_positions = [(i, 4) for i in range(xbar.size(0))]

    xhat_ids = xbar.clone().to(device)
    for b, p in masked_positions:
        if p >= xhat_ids.size(1):
            raise ValueError(
                f"masked position {p} is out of range for sequence length {xhat_ids.size(1)}"
            )
        xhat_ids[b, p] = tokenizer.mask_token_id

    embeddings = model.bert.embeddings

    def token_embedding(text: str) -> torch.Tensor:
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(token_ids) == 0:
            raise ValueError(f"token '{text}' could not be tokenized")
        token_tensor = torch.tensor(token_ids, device=device, dtype=torch.long)
        return embeddings.word_embeddings.weight[token_tensor].mean(dim=0)

    mask_embedding = embeddings.word_embeddings.weight[tokenizer.mask_token_id]
    sitting_embedding = token_embedding("sitting")
    lying_embedding = token_embedding("lying")
    swimming_embedding = token_embedding("swimming")
    toilet_embedding = token_embedding("toilet")
    leaning_embedding = token_embedding("leaning")
    standing_embedding = token_embedding("standing")
    playing_embedding = token_embedding("playing")
    talking_embedding = token_embedding("talking")

    variant_specs = [
        ("1", mask_embedding),
        ("2", mask_embedding + 0.5 * lying_embedding),
        ("3", 0.5 * mask_embedding + 0.5 * lying_embedding),
        ("4", mask_embedding + 0.7 * lying_embedding),
        ("5", 0.3 * mask_embedding + 0.7 * lying_embedding),
        (
            "6",
            0.3 * lying_embedding
            + 0.2 * sitting_embedding
            + 0.2 * leaning_embedding
            + 0.2 * standing_embedding
            + 0.1 * playing_embedding,
        ),
        (
            "7",
            0.2 * mask_embedding
            + (
                0.3 * lying_embedding
                + 0.2 * sitting_embedding
                + 0.2 * leaning_embedding
                + 0.2 * standing_embedding
                + 0.1 * playing_embedding
            ),
        ),
        (
            "8",
            0.2 * mask_embedding
            + 0.8
            * (
                0.3 * lying_embedding
                + 0.2 * sitting_embedding
                + 0.2 * leaning_embedding
                + 0.2 * standing_embedding
                + 0.1 * playing_embedding
            ),
        ),
        (
            "9",
            mask_embedding
            + 4
            * (
                0.3 * lying_embedding
                + 0.2 * sitting_embedding
                + 0.2 * leaning_embedding
                + 0.2 * standing_embedding
                + 0.1 * playing_embedding
            ),
        ),
        (
            "9-2",
            4 * mask_embedding
            + (
                0.3 * lying_embedding
                + 0.2 * sitting_embedding
                + 0.2 * leaning_embedding
                + 0.2 * standing_embedding
                + 0.1 * playing_embedding
            ),
        ),
        (
            "10",
            0.2 * mask_embedding
            + (
                0.6 * lying_embedding
                + 0.1 * sitting_embedding
                + 0.1 * leaning_embedding
                + 0.1 * standing_embedding
                + 0.1 * playing_embedding
            ),
        ),
        (
            "11",
            0.2 * mask_embedding
            + 0.8
            * (
                0.6 * lying_embedding
                + 0.1 * sitting_embedding
                + 0.1 * leaning_embedding
                + 0.1 * standing_embedding
                + 0.1 * playing_embedding
            ),
        ),
        (
            "12",
            4 * mask_embedding
            + (
                0.6 * lying_embedding
                + 0.1 * sitting_embedding
                + 0.1 * leaning_embedding
                + 0.1 * standing_embedding
                + 0.1 * playing_embedding
            ),
        ),
        (
            "7",
            0.6 * lying_embedding
            + 0.1 * sitting_embedding
            + 0.1 * leaning_embedding
            + 0.1 * standing_embedding
            + 0.1 * playing_embedding,
        ),
        (
            "8",
            0.2
            * (
                lying_embedding
                + swimming_embedding
                + toilet_embedding
                + standing_embedding
                + talking_embedding
            ),
        ),
        (
            "9",
            0.1 * lying_embedding
            + 0.1 * swimming_embedding
            + 0.6 * toilet_embedding
            + 0.1 * standing_embedding
            + 0.1 * talking_embedding,
        ),
        ("8", 0.2 * sitting_embedding + 0.8 * lying_embedding),
        ("9", 0.0 * sitting_embedding + 1 * lying_embedding),
        ("10", 0.5 * (sitting_embedding + swimming_embedding)),
        ("11", 0.2 * sitting_embedding + 0.8 * swimming_embedding),
        ("12", 0.5 * (sitting_embedding + toilet_embedding)),
        ("13", 0.2 * sitting_embedding + 0.8 * toilet_embedding),
        ("14", 0.5 * sitting_embedding),
        ("15", 0.5 * (sitting_embedding + lying_embedding)),
        ("16", 0.5 * (sitting_embedding + swimming_embedding)),
        ("17", 0.5 * (sitting_embedding + toilet_embedding)),
    ]

    def build_inputs_embeds(mask_word_embedding: torch.Tensor) -> torch.Tensor:
        word_emb = embeddings.word_embeddings(xhat_ids).clone()
        for b, p in masked_positions:
            word_emb[b, p] = mask_word_embedding

        position_ids = (
            torch.arange(xhat_ids.size(1), device=device)
            .unsqueeze(0)
            .expand_as(xhat_ids)
        )
        token_type_ids = torch.zeros_like(xhat_ids)
        inputs_embeds = (
            word_emb
            + embeddings.position_embeddings(position_ids)
            + embeddings.token_type_embeddings(token_type_ids)
        )
        inputs_embeds = embeddings.LayerNorm(inputs_embeds)
        inputs_embeds = embeddings.dropout(inputs_embeds)
        return inputs_embeds

    restored_texts_by_variant: list[tuple[str, list[dict[str, str | float]]]] = []

    for variant_name, mask_word_embedding in variant_specs:
        inputs_embeds = build_inputs_embeds(mask_word_embedding)
        out = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        logits = out.logits
        probs = torch.softmax(logits, dim=-1)
        pred_ids = probs.argmax(dim=-1)

        restored = xhat_ids.clone()
        for b, p in masked_positions:
            restored[b, p] = pred_ids[b, p]

        restored_rows: list[dict[str, str | float]] = []
        for i in range(input_ids.size(0)):
            attn = attn_mask[i].bool()
            restored_ids = sanitize_for_decode(restored[i][attn], vocab_size, unk_id)
            restored_text = tokenizer.decode(restored_ids, skip_special_tokens=True)

            _, masked_pos = masked_positions[i]
            topk_vals, topk_ids = torch.topk(probs[i, masked_pos], k=5)
            pred_prob = float(topk_vals[0].item())
            top2_token = tokenizer.convert_ids_to_tokens(int(topk_ids[1].item()))
            top3_token = tokenizer.convert_ids_to_tokens(int(topk_ids[2].item()))
            top4_token = tokenizer.convert_ids_to_tokens(int(topk_ids[3].item()))
            top5_token = tokenizer.convert_ids_to_tokens(int(topk_ids[4].item()))

            restored_rows.append(
                {
                    "text": restored_text,
                    "pred_prob": pred_prob,
                    "top2": top2_token,
                    "top3": top3_token,
                    "top4": top4_token,
                    "top5": top5_token,
                }
            )

        restored_texts_by_variant.append((variant_name, restored_rows))

    for i in range(input_ids.size(0)):
        attn = attn_mask[i].bool()
        clean_ids = input_ids[i][attn]
        corrupted_ids = xbar[i][attn]
        masked_ids = xhat_ids[i][attn]

        clean_list = clean_ids.tolist()
        corrupted_list = sanitize_for_decode(corrupted_ids, vocab_size, unk_id)
        masked_list = sanitize_for_decode(masked_ids, vocab_size, unk_id)

        clean_text = tokenizer.decode(clean_list, skip_special_tokens=True)
        corrupted_text = tokenizer.decode(corrupted_list, skip_special_tokens=True)
        masked_text = tokenizer.decode(masked_list, skip_special_tokens=False)

        print(f"\nSample {i + 1}")
        print(f"Clean:   {clean_text}")
        print(f"Errored: {corrupted_text}")
        print(f"Masked:  {masked_text}")
        for variant_name, restored_rows in restored_texts_by_variant:
            row = restored_rows[i]
            print(
                f"Restored[{variant_name}]: {row['text']} | p={row['pred_prob']:.4f} | top2={row['top2']} | top3={row['top3']} | top4={row['top4']} | top5={row['top5']}"
            )

    print("end")


def test_input_fun2():
    # test 할때 필요한 변수들은 대부분 config.py 에서 test_ 로 시작하는 것들
    cfg = parse_test_args()

    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # snr, modulation 지정해서 실험
    test_snr_db = cfg.test_snr_db if cfg.test_snr_db is not None else cfg.snr_db
    test_modulation = (
        cfg.test_modulation if cfg.test_modulation is not None else cfg.modulation
    )

    # masking 방법, guidance 방법, fusion 사용 여부 등도 지정할 수 있음. None 으로 하면 train 때 설정 따라감
    test_mask_strategy = (
        cfg.test_mask_strategy
        if cfg.test_mask_strategy is not None
        else cfg.mask_strategy
    )
    test_mask_rate = (
        cfg.test_mask_rate if cfg.test_mask_rate is not None else cfg.mask_rate
    )
    test_rel_tau = cfg.test_rel_tau if cfg.test_rel_tau is not None else cfg.rel_tau
    test_guide = cfg.test_guide if cfg.test_guide is not None else cfg.guide
    test_use_fusion = (
        cfg.test_use_fusion if cfg.test_use_fusion is not None else cfg.use_fusion
    )
    test_attn_beta_init = (
        cfg.test_attn_beta_init
        if cfg.test_attn_beta_init is not None
        else cfg.attn_beta_init
    )
    test_rel_mode = (
        cfg.test_rel_mode if cfg.test_rel_mode is not None else cfg.real_mode
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForMaskedLM.from_pretrained(
        cfg.model_name, revision="refs/pr/1", use_safetensors=True
    ).to(device)

    # output fusion 하는 모델
    fusion = (
        AdaptiveFusion(model.config.hidden_size).to(device) if test_use_fusion else None
    )
    if fusion is not None:
        fusion.eval()

    model.eval()

    # text 지정할 수 있고, 그냥 coco captions 에서 가져오고 싶으면 texts=[] 로 하면 됨
    texts = [
        "A cat is sitting on the mat.",
        "A dog is sitting on the mat.",
        "A cat sleeping on the chair.",
        "A girl with a pink dress is dancing.",
    ]
    if not texts:
        captions = list(
            iter_coco_captions(
                cfg.dataset_name, cfg.split, cfg.text_field, cfg.test_num_samples
            )
        )
        texts = captions[: cfg.test_num_samples]

    input_ids, attn_mask = collate_texts(tokenizer, texts, cfg.max_len)
    input_ids = input_ids.to(device)
    attn_mask = attn_mask.to(device)

    m_bits = int(
        math.ceil(math.log2(model.config.vocab_size))
    )  # vocab_size 를 표현할 수 있는 bit 수
    lut = popcount_u16_lut(device)

    print(f"\nfusion: {test_use_fusion}, guide: {test_guide}")
    snr_db_used = test_snr_db if test_snr_db is not None else random.uniform(-5.0, 10.0)
    print(f"Using SNR (dB): {snr_db_used:.2f}")

    # x가 보내는 token id, noisy 채널 거쳐서 받은 errored tokein id가 xbar, llr_tok은 각 bit별 LLR 값.
    xbar, llr_tok = transmit_ids_awgn_llr(  # channel_corrupt.py 에 있음
        input_ids, snr_db_used, test_modulation, m_bits, tokenizer
    )

    # === Inline make_masked_input ===
    if cfg.test_mask:
        device = xbar.device
        B, L = xbar.shape
        vocab_size = getattr(tokenizer, "vocab_size", None)
        invalid_mask = (xbar < 0) | (xbar >= vocab_size)
        abs_llr = llr_tok.abs()
        if test_rel_mode == "minabs":
            rel = abs_llr.min(dim=-1).values
        elif test_rel_mode == "meanabs":
            rel = abs_llr.mean(dim=-1)

        rel = rel.masked_fill(invalid_mask, 0.0)
        if test_mask_strategy == "random":
            mask = torch.rand((B, L), device=device) < test_mask_rate
        elif test_mask_strategy == "reliability":
            mask = rel < test_rel_tau

        mask |= invalid_mask
        special = torch.zeros_like(mask)
        for sid in tokenizer.all_special_ids + [1012]:
            special |= xbar == sid
        mask = (mask & ~special) | invalid_mask
        xhat_ids = xbar.clone()
        xhat_ids[mask] = tokenizer.mask_token_id
    else:
        xhat_ids = xbar.clone()

    # === Inline forward_with_guidance ===
    mask_positions = (
        (xhat_ids == tokenizer.mask_token_id).nonzero(as_tuple=False).tolist()
    )

    if len(mask_positions) == 0:
        out = model(
            input_ids=xhat_ids,
            attention_mask=attn_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        logits = out.logits

    else:
        ber, _ = ber_from_snr(snr_db_used, test_modulation)

        # === Inline build_q_list_for_masks ===
        q_list = []
        for b, pos in mask_positions:
            # LLR
            llr_at_pos = llr_tok[b, pos]  # [m] LLR for m bits of this token
            q = q_full_vocab_from_llr(llr_at_pos, model.config.vocab_size, xbar.device)

            q_list.append(q)

        # === Input soft embeddings ===
        bert = model.bert
        embeddings = bert.embeddings
        word_emb = embeddings.word_embeddings(xhat_ids)
        E = embeddings.word_embeddings.weight

        for (b, pos), q in zip(mask_positions, q_list):
            e_soft = torch.matmul(q.to(xhat_ids.device), E)
            word_emb[b, pos, :] = e_soft

        position_ids = (
            torch.arange(xhat_ids.size(1), device=xhat_ids.device)
            .unsqueeze(0)
            .expand_as(xhat_ids)
        )
        token_type_ids = torch.zeros_like(xhat_ids)
        inputs_embeds = (
            word_emb
            + embeddings.position_embeddings(position_ids)
            + embeddings.token_type_embeddings(token_type_ids)
        )
        inputs_embeds = embeddings.LayerNorm(inputs_embeds)
        inputs_embeds = embeddings.dropout(inputs_embeds)
        out = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        logits = out.logits

    # store pre-fusion logits for comparison
    pre_fusion_logits = logits.detach().clone()

    if cfg.test_use_fusion:
        g = 0.85
        eps = 1e-12
        for idx, (b, p) in enumerate(mask_positions):
            q = q_list[idx].to(xhat_ids.device)
            log_q = torch.log(q + eps)
            fused = logits[b, p] + g * log_q
            logits[b, p] = fused

    # pre-fusion prediction
    pre_fusion_pred_ids = torch.softmax(pre_fusion_logits, dim=-1).argmax(dim=-1)
    pre_fusion_restored = xhat_ids.clone()
    mask = xhat_ids == tokenizer.mask_token_id
    pre_fusion_restored[mask] = pre_fusion_pred_ids[mask]

    # post-fusion prediction
    pred_ids = torch.softmax(logits, dim=-1).argmax(dim=-1)
    restored = xhat_ids.clone()
    restored[mask] = pred_ids[mask]

    loss_fn = nn.CrossEntropyLoss()

    vocab_size = model.config.vocab_size
    unk_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else 100

    for i in range(input_ids.size(0)):
        attn = attn_mask[i].bool()
        clean_ids = input_ids[i][attn]
        corrupted_ids = xbar[i][attn]
        masked_ids = xhat_ids[i][attn]
        pre_fusion_restored_ids = pre_fusion_restored[i][attn]
        restored_ids = restored[i][attn]

        clean_list = clean_ids.tolist()
        corrupted_list = sanitize_for_decode(corrupted_ids, vocab_size, unk_id)
        masked_list = sanitize_for_decode(masked_ids, vocab_size, unk_id)
        pre_fusion_restored_list = sanitize_for_decode(
            pre_fusion_restored_ids, vocab_size, unk_id
        )
        restored_list = sanitize_for_decode(restored_ids, vocab_size, unk_id)

        clean_text = tokenizer.decode(clean_list, skip_special_tokens=True)
        corrupted_text = tokenizer.decode(corrupted_list, skip_special_tokens=True)
        masked_text = tokenizer.decode(masked_list, skip_special_tokens=False)
        pre_fusion_restored_text = tokenizer.decode(
            pre_fusion_restored_list, skip_special_tokens=True
        )
        restored_text = tokenizer.decode(restored_list, skip_special_tokens=True)

        ce_loss = loss_fn(
            logits[i].view(-1, logits.size(-1)), input_ids[i].view(-1)
        ).item()

        pred_flat = pred_ids[i].view(-1).cpu().numpy()
        clean_flat = input_ids[i].view(-1).cpu().numpy()
        f1 = f1_score(clean_flat, pred_flat, average="macro", zero_division=0)

        print(f"\nSample {i + 1}")
        print(f"F1: {f1:.4f} | CE Loss: {ce_loss:.4f}")
        print(f"Clean:   {clean_text}")
        print(f"Errored: {corrupted_text}")
        print(f"Masked:  {masked_text}")

        # print q diagnostics for each mask position
        for idx, (b, p) in enumerate(mask_positions):
            if b == i:
                q = q_list[idx].cpu()
                clean_token_id = int(input_ids[i, p].item())
                clean_token = tokenizer.convert_ids_to_tokens(clean_token_id)
                clean_q = float(q[clean_token_id].item())
                clean_rank = int((q > q[clean_token_id]).sum().item()) + 1

                topk_vals, topk_indices = torch.topk(q, k=10)
                topk_tokens = [
                    tokenizer.convert_ids_to_tokens(int(token_id.item()))
                    for token_id in topk_indices
                ]
                topk_scores = [f"{float(v.item()):.3f}" for v in topk_vals]
                topk_pairs = list(zip(topk_tokens, topk_scores))

                # pre-fusion logits softmax for this mask position
                pre_fusion_logits_pos = pre_fusion_logits[b, p]
                pre_fusion_softmax = torch.softmax(pre_fusion_logits_pos, dim=-1)
                pre_fusion_topk_vals, pre_fusion_topk_indices = torch.topk(
                    pre_fusion_softmax, k=10
                )
                pre_fusion_topk_tokens = [
                    tokenizer.convert_ids_to_tokens(int(tid.item()))
                    for tid in pre_fusion_topk_indices
                ]
                pre_fusion_topk_scores = [
                    f"{float(v.item()):.3f}" for v in pre_fusion_topk_vals
                ]
                pre_fusion_topk_pairs = list(
                    zip(pre_fusion_topk_tokens, pre_fusion_topk_scores)
                )

                print(
                    f"  Mask @pos{p} - Clean token: {clean_token} | rank: {clean_rank} | q: {clean_q:.3f}"
                )
                print(f"  Mask @pos{p} - Top-10 q vocabs: {topk_pairs}")
                print(
                    f"  Mask @pos{p} - Top-10 Pre-Fusion softmax: {pre_fusion_topk_pairs}"
                )

                # post-fusion logits softmax for this mask position
                post_fusion_logits_pos = logits[b, p]
                post_fusion_softmax = torch.softmax(post_fusion_logits_pos, dim=-1)
                post_fusion_topk_vals, post_fusion_topk_indices = torch.topk(
                    post_fusion_softmax, k=10
                )
                post_fusion_topk_tokens = [
                    tokenizer.convert_ids_to_tokens(int(tid.item()))
                    for tid in post_fusion_topk_indices
                ]
                post_fusion_topk_scores = [
                    f"{float(v.item()):.3f}" for v in post_fusion_topk_vals
                ]
                post_fusion_topk_pairs = list(
                    zip(post_fusion_topk_tokens, post_fusion_topk_scores)
                )
                print(
                    f"  Mask @pos{p} - Top-10 Post-Fusion softmax: {post_fusion_topk_pairs}"
                )

        print(f"Pre-Fusion Restored:{pre_fusion_restored_text}")
        print(f"Restored:{restored_text}")


if __name__ == "__main__":
    # test_input_fun2()
    # test()
    test_snr()
    # test_snr_nopropos()
