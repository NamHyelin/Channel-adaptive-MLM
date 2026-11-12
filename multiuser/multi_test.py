import argparse
import math
import os
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from bert_score import score as bert_score

from transformers import AutoTokenizer, AutoModelForMaskedLM
from multiuser.multi_config import Config
from data_coco import iter_coco_captions
from multiuser.multi_channel_corrupt import (
    transmit_ids_awgn_llr_multiuser_allusers,
    build_multiuser_ids,
    make_masked_input,
    ber_from_snr,
)
from q_model import popcount_u16_lut, q_full_vocab_from_ber
from guidance import (
    make_soft_inputs_embeds,
    build_key_bias_from_q,
    BertModelWithKeyBias,
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
        if k in {"snr_db", "test_snr_db"}:
            p.add_argument(arg, type=str, default=None)
        elif k in {"test_texts"}:
            p.add_argument(arg, action="append", default=None)
        elif isinstance(v, bool):
            p.add_argument(arg, action="store_true" if not v else "store_false")
        else:
            p.add_argument(arg, type=type(v), default=v)
    ns = p.parse_args()
    data = vars(ns)
    for key in ("snr_db", "test_snr_db"):
        raw = data.get(key)
        if raw is None:
            data[key] = None
        else:
            s = str(raw).strip().lower()
            data[key] = None if s in {"none", "null", ""} else float(raw)
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


def parse_snr_values(spec: str) -> list[int]:
    s = spec.strip()
    if ":" in s:
        a, b, c = s.split(":")
        start, end, step = int(a), int(b), int(c)
        if step == 0:
            raise ValueError("test_snr_values step must not be 0")
        if (end - start) * step < 0:
            raise ValueError("test_snr_values has inconsistent range/step")
        vals = []
        cur = start
        if step > 0:
            while cur <= end:
                vals.append(cur)
                cur += step
        else:
            while cur >= end:
                vals.append(cur)
                cur += step
        return vals
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def build_power_profile(num_users: int, power_delta: float) -> list[float]:
    """
    Auto profile from single parameter (random per sample):
      P_k = Uniform(max(1 - k*delta, 0), 1.0), k=0..U-1
    """
    if num_users < 1:
        raise ValueError("num_users must be >= 1")
    if power_delta < 0.0:
        raise ValueError("power_delta must be >= 0")
    power_profile = []
    for k in range(num_users):
        min_p = max(1.0 - power_delta * k, 0.0)
        max_p = 1.0
        p_k = random.uniform(min_p, max_p)
        power_profile.append(p_k)
    return power_profile


def build_q_list_for_masks(
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
            q = q_full_vocab_from_ber(vocab_size, xbar, ber, xbar_ids.device, lut_u16)
        q_list.append(q)
    return q_list


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


def _empty_user_bucket() -> dict[str, list]:
    return {
        "sinr_db": [],
        "ce": [],
        "f1": [],
        "bert_score": [],
        "sentence_avg_q_values": [],
        "fusion_js_fused_softmax_vs_q": [],
        "fusion_js_fused_softmax_vs_logits_softmax": [],
        "fusion_cosine_fused_vs_logits": [],
        "fusion_pearson_fused_vs_logits": [],
    }


def test_snr():
    cfg = parse_test_args()
    set_seed(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForMaskedLM.from_pretrained(
        cfg.model_name, revision="refs/pr/1", use_safetensors=True
    ).to(device)

    bert_bias_wrapper = (
        BertModelWithKeyBias(model.bert).to(device)
        if cfg.test_guide == "attn_bias"
        else None
    )
    fusion = (
        AdaptiveFusion(model.config.hidden_size).to(device)
        if cfg.test_use_fusion
        else None
    )

    if cfg.test_model_path:
        ckpt = torch.load(cfg.test_model_path, map_location=device)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        if fusion is not None and "fusion_state_dict" in ckpt:
            fusion.load_state_dict(ckpt["fusion_state_dict"])

    model.eval()
    if fusion is not None:
        fusion.eval()

    captions = list(
        iter_coco_captions(cfg.dataset_name, cfg.split, cfg.text_field, cfg.max_samples)
    )
    texts = captions[: cfg.test_num_samples]
    val_dl = DataLoader(
        texts, batch_size=cfg.test_batch_size, shuffle=False, drop_last=False
    )

    m_bits = int(math.ceil(math.log2(model.config.vocab_size)))
    lut = popcount_u16_lut(device)
    loss_fn = nn.CrossEntropyLoss()
    vocab_size = model.config.vocab_size
    unk_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else 100

    snr_values = parse_snr_values(cfg.test_snr_values)
    results_list: list[dict] = []

    for snr_db_used in snr_values:
        print(
            f"Evaluating CQI-SNR={snr_db_used} dB, U={cfg.multi_num_users}, delta={cfg.multi_power_delta} (random per batch)"
        )

        per_user = {u: _empty_user_bucket() for u in range(cfg.multi_num_users)}
        global_sent_idx = {u: 0 for u in range(cfg.multi_num_users)}

        for batch_texts in val_dl:
            batch_texts = list(batch_texts)
            # Sample random power profile for this batch
            power_ratios = build_power_profile(
                cfg.multi_num_users, cfg.multi_power_delta
            )
            input_ids, attn_mask = collate_texts(tokenizer, batch_texts, cfg.max_len)
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)

            ids_users = build_multiuser_ids(
                input_ids,
                num_users=cfg.multi_num_users,
                mode=cfg.multi_interferer_mode,
            )
            xbar_all, llr_all, sinr_db_all = transmit_ids_awgn_llr_multiuser_allusers(
                ids_users,
                snr_db=float(snr_db_used),
                modulation=cfg.test_modulation,
                m_bits=m_bits,
                power_ratios=power_ratios,
                channel_gain_mode=cfg.multi_channel_gain_mode,
                pilot_tokens=cfg.multi_pilot_tokens,
            )

            for u in range(cfg.multi_num_users):
                clean_ids_u = ids_users[u]
                xbar = xbar_all[u]
                llr_tok = llr_all[u]
                sinr_db_u = float(sinr_db_all[u].item())

                bucket = per_user[u]
                bucket["sinr_db"].append(sinr_db_u)

                xhat_ids, _ = make_masked_input(
                    xbar,
                    llr_tok,
                    tokenizer,
                    strategy=cfg.test_mask_strategy,
                    mask_rate=cfg.test_mask_rate,
                    rel_tau=cfg.test_rel_tau,
                    rel_mode=cfg.test_rel_mode,
                )

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
                    restored = xhat_ids.clone()

                    ce_loss = loss_fn(
                        logits.view(-1, logits.size(-1)), clean_ids_u.view(-1)
                    ).item()
                    pred_flat = restored.view(-1).detach().cpu().numpy()
                    clean_flat = clean_ids_u.view(-1).detach().cpu().numpy()
                    f1 = f1_score(
                        clean_flat, pred_flat, average="macro", zero_division=0
                    )

                    clean_texts = []
                    restored_texts = []
                    for i in range(clean_ids_u.size(0)):
                        attn = attn_mask[i].bool()
                        clean_ids_list = clean_ids_u[i][attn].tolist()
                        restored_ids_list = [
                            t if 0 <= t < vocab_size else unk_id
                            for t in restored[i][attn].tolist()
                        ]
                        clean_texts.append(
                            tokenizer.decode(clean_ids_list, skip_special_tokens=True)
                        )
                        restored_texts.append(
                            tokenizer.decode(
                                restored_ids_list, skip_special_tokens=True
                            )
                        )

                    _p, _r, bert_f1_tensor = bert_score(
                        restored_texts, clean_texts, lang="en", verbose=False
                    )
                    bert_f1 = bert_f1_tensor.mean().item()

                    bucket["ce"].append(float(ce_loss))
                    bucket["f1"].append(float(f1))
                    bucket["bert_score"].append(float(bert_f1))

                    for i, orig_text in enumerate(batch_texts):
                        bucket["sentence_avg_q_values"].append(
                            {
                                "sentence_idx": global_sent_idx[u],
                                "text": f"[user {u}] {orig_text}",
                                "avg_q_value": None,
                            }
                        )
                        global_sent_idx[u] += 1
                    continue

                ber_u, _ = ber_from_snr(sinr_db_u, cfg.test_modulation)
                q_list = build_q_list_for_masks(
                    vocab_size=model.config.vocab_size,
                    xbar_ids=xbar,
                    mask_positions=[(b, p) for b, p in mask_positions],
                    ber=ber_u,
                    lut_u16=lut,
                )

                if cfg.test_guide == "soft_embed":
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
                elif cfg.test_guide == "attn_bias":
                    key_bias = build_key_bias_from_q(
                        input_ids=xhat_ids,
                        mask_positions=[(b, p) for b, p in mask_positions],
                        q_list=q_list,
                        ber=ber_u,
                        mode="rmax",
                        beta=cfg.test_attn_beta_init,
                    ).to(xhat_ids.device)
                    enc_out = bert_bias_wrapper(
                        input_ids=xhat_ids,
                        attention_mask=attn_mask,
                        key_bias=key_bias,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    hidden_last = enc_out["last_hidden_state"]
                    logits = model.cls(hidden_last)
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
                    pre_fusion_logits = logits.detach().clone()
                    for idx, (b, p) in enumerate(mask_positions):
                        fused, _gate = fusion(
                            logits[b, p],
                            hidden_last[b, p],
                            q_list[idx].to(xhat_ids.device),
                            ber=ber_u,
                        )
                        logits_vec = pre_fusion_logits[b, p]
                        fused_softmax = torch.softmax(fused, dim=-1)
                        logits_softmax = torch.softmax(logits_vec, dim=-1)
                        q_probs = q_list[idx].to(fused.device)

                        bucket["fusion_js_fused_softmax_vs_q"].append(
                            js_divergence_from_probs(fused_softmax, q_probs)
                        )
                        bucket["fusion_js_fused_softmax_vs_logits_softmax"].append(
                            js_divergence_from_probs(fused_softmax, logits_softmax)
                        )
                        bucket["fusion_cosine_fused_vs_logits"].append(
                            float(
                                F.cosine_similarity(
                                    fused.unsqueeze(0),
                                    logits_vec.unsqueeze(0),
                                    dim=-1,
                                ).item()
                            )
                        )
                        bucket["fusion_pearson_fused_vs_logits"].append(
                            pearson_corr(fused, logits_vec)
                        )

                        logits[b, p] = fused

                pred_ids = logits.argmax(dim=-1)
                restored = xhat_ids.clone()
                mask = xhat_ids == tokenizer.mask_token_id
                restored[mask] = pred_ids[mask]

                per_sentence_q_values: dict[int, list[float]] = {
                    i: [] for i in range(clean_ids_u.size(0))
                }
                for idx, (b, pos) in enumerate(mask_positions):
                    pred_id = int(pred_ids[b, pos].item())
                    q_val = float(q_list[idx][pred_id].item())
                    per_sentence_q_values[b].append(q_val)

                ce_loss = loss_fn(
                    logits.view(-1, logits.size(-1)), clean_ids_u.view(-1)
                ).item()
                pred_flat = restored.view(-1).detach().cpu().numpy()
                clean_flat = clean_ids_u.view(-1).detach().cpu().numpy()
                f1 = f1_score(clean_flat, pred_flat, average="macro", zero_division=0)

                clean_texts = []
                restored_texts = []
                for i in range(clean_ids_u.size(0)):
                    attn = attn_mask[i].bool()
                    clean_ids_list = clean_ids_u[i][attn].tolist()
                    restored_ids_list = [
                        t if 0 <= t < vocab_size else unk_id
                        for t in restored[i][attn].tolist()
                    ]
                    clean_texts.append(
                        tokenizer.decode(clean_ids_list, skip_special_tokens=True)
                    )
                    restored_texts.append(
                        tokenizer.decode(restored_ids_list, skip_special_tokens=True)
                    )

                _p, _r, F1 = bert_score(
                    restored_texts, clean_texts, lang="en", verbose=False
                )
                bert_f1 = F1.mean().item()

                bucket["ce"].append(float(ce_loss))
                bucket["f1"].append(float(f1))
                bucket["bert_score"].append(float(bert_f1))

                for i, orig_text in enumerate(batch_texts):
                    sentence_q_values = per_sentence_q_values[i]
                    avg_q_value = (
                        sum(sentence_q_values) / len(sentence_q_values)
                        if sentence_q_values
                        else None
                    )
                    bucket["sentence_avg_q_values"].append(
                        {
                            "sentence_idx": global_sent_idx[u],
                            "text": f"[user {u}] {orig_text}",
                            "avg_q_value": avg_q_value,
                        }
                    )
                    global_sent_idx[u] += 1

        users_result = []
        for u in range(cfg.multi_num_users):
            b = per_user[u]
            users_result.append(
                {
                    "user_idx": u,
                    "avg_sinr_db": mean_or_none(b["sinr_db"]),
                    "avg_ce_loss": mean_or_none(b["ce"]),
                    "avg_f1_score": mean_or_none(b["f1"]),
                    "avg_bert_score": mean_or_none(b["bert_score"]),
                    "sentence_avg_q_values": b["sentence_avg_q_values"],
                    **(
                        {
                            "avg_js_fused_softmax_vs_q": mean_or_none(
                                b["fusion_js_fused_softmax_vs_q"]
                            ),
                            "avg_js_fused_softmax_vs_logits_softmax": mean_or_none(
                                b["fusion_js_fused_softmax_vs_logits_softmax"]
                            ),
                            "avg_cosine_fused_vs_logits": mean_or_none(
                                b["fusion_cosine_fused_vs_logits"]
                            ),
                            "avg_pearson_fused_vs_logits": mean_or_none(
                                b["fusion_pearson_fused_vs_logits"]
                            ),
                        }
                        if cfg.test_use_fusion
                        else {}
                    ),
                }
            )

        results_list.append(
            {
                "snr_db_cqi": snr_db_used,
                "num_users": cfg.multi_num_users,
                "pilot_tokens_ignored": cfg.multi_pilot_tokens,
                "power_delta_from_max": cfg.multi_power_delta,
                "power_ratios": power_ratios,
                "channel_gain_mode": cfg.multi_channel_gain_mode,
                "users": users_result,
            }
        )

    os.makedirs("results", exist_ok=True)
    out_filename = (
        f"multiuser__fusion_{cfg.test_use_fusion}"
        f"__guide_{cfg.test_guide}"
        f"__numusers_{cfg.multi_num_users}"
        f"__powerdelta_{cfg.multi_power_delta}"
        f"__{cfg.test_modulation}.json"
    )
    out_path = os.path.join("results", out_filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results_list, f, ensure_ascii=False, indent=2)

    print(f"Saved multi-user sequential-user SINR results to {out_path}")


if __name__ == "__main__":
    test_snr()
