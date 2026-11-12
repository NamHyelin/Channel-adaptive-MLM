# ============================================================
# ORIGINAL CODE (FULLY COMMENTED OUT)
# ============================================================
# 아래 부분은 기존 코드를 그대로 보존한 것이며 모두 주석 처리되어 실행되지 않습니다.
# 실제 실행 코드는 "NEW CODE" 아래에서 시작합니다.
# ============================================================

# import argparse
# import csv
# import datetime
# import math
# import os
# import random
#
# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# from transformers import AutoModelForMaskedLM, AutoTokenizer
#
# from config import Config
# from data_coco import iter_coco_captions
# from channel_corrupt import transmit_ids_awgn_llr, make_masked_input
# from q_model import q_full_vocab_from_llr
# from guidance import make_soft_inputs_embeds
#
#
# # ============================================================
# # Plot settings
# # ============================================================
# SNR_CHOICES = list(range(-5, 11))  # -5, -4, ..., 10 dB
# NUM_BINS = 20  # channel-posterior percentile bins
# NUM_PLOT_LAYERS = 4  # Layer 1 ~ Layer 4
# EPS = 1e-12
#
#
# def set_seed(seed: int):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#
#
# def parse_test_args() -> Config:
#     """Use the same Config/CLI style as the original test code."""
#     p = argparse.ArgumentParser()
#     for k, v in Config().__dict__.items():
#         arg = "--" + k.replace("_", "-")
#         if k == "snr_db":
#             p.add_argument(arg, type=str, default=None)
#         elif k == "test_texts":
#             p.add_argument(arg, action="append", default=None)
#         elif isinstance(v, bool):
#             p.add_argument(arg, action="store_true" if not v else "store_false")
#         else:
#             p.add_argument(arg, type=type(v), default=v)
#
#     ns = p.parse_args()
#     data = vars(ns)
#
#     raw = data.get("snr_db")
#     if raw is None:
#         data["snr_db"] = None
#     else:
#         s = str(raw).strip().lower()
#         data["snr_db"] = None if s in {"none", "null", ""} else float(raw)
#
#     data["test_texts"] = data.get("test_texts") or []
#     return Config(**data)
#
#
# def collate_texts(tokenizer, texts, max_len: int):
#     enc = tokenizer(
#         texts,
#         padding=True,
#         truncation=True,
#         max_length=max_len,
#         return_tensors="pt",
#     )
#     return enc["input_ids"], enc["attention_mask"]
#
#
# def build_q_list_for_masks(
#     llr: torch.Tensor,
#     vocab_size: int,
#     xbar_ids: torch.Tensor,
#     mask_positions: list[tuple[int, int]],
# ):
#     """Same LLR -> vocabulary posterior q construction used in test_snr()."""
#     q_list = []
#     for b, pos in mask_positions:
#         xbar = int(xbar_ids[b, pos].item())
#
#         if xbar < 0 or xbar >= vocab_size:
#             q = torch.full(
#                 (vocab_size,),
#                 1.0 / vocab_size,
#                 device=xbar_ids.device,
#                 dtype=torch.float32,
#             )
#         else:
#             llr_at_pos = llr[b, pos]
#             q = q_full_vocab_from_llr(
#                 llr_at_pos,
#                 vocab_size,
#                 xbar_ids.device,
#                 temp=1,
#             )
#
#         q_list.append(q)
#
#     return q_list
#
#
# def get_layer_logits_at_masks(model, hidden_states, mask_positions, num_layers: int):
#     """
#     Apply the SAME pretrained MLM head to each intermediate hidden layer.
#
#     hidden_states[0] : embedding output
#     hidden_states[1] : Transformer Layer 1 output
#     ...
#
#     Returns:
#         list of length num_layers, each [num_masks, vocab_size] softmax probabilities
#     """
#     if not hasattr(model, "cls"):
#         raise AttributeError(
#             "This analysis expects a BERT-style AutoModelForMaskedLM with model.cls."
#         )
#
#     batch_idx = torch.tensor(
#         [b for b, _ in mask_positions], device=hidden_states[0].device, dtype=torch.long
#     )
#     seq_idx = torch.tensor(
#         [p for _, p in mask_positions], device=hidden_states[0].device, dtype=torch.long
#     )
#
#     layer_logits = []
#     for layer_idx in range(1, num_layers + 1):
#         # Only extract hidden vectors at masked positions to save GPU memory.
#         h_mask = hidden_states[layer_idx][batch_idx, seq_idx, :]  # [M, H]
#         z = model.cls(h_mask)  # [M, V]
#         # layer_logits.append(z)
#         p = torch.softmax(z, dim=-1)  # [M, V]
#         layer_logits.append(p)
#
#     return layer_logits
#
#
# def accumulate_by_q_percentile(
#     q_list,
#     ig_layer_probs,
#     cmlm_layer_probs,
#     sum_q,
#     count_q,
#     sum_ig,
#     count_ig,
#     min_ig,
#     max_ig,
#     sum_cmlm,
#     count_cmlm,
#     min_cmlm,
#     max_cmlm,
# ):
#     """
#     For every masked position:
#       1) sort all vocabulary candidates by q(v),
#       2) divide them into NUM_BINS equal-size rank-percentile groups,
#       3) use the mean q(v) of each group as the x-axis statistic,
#       4) collect mean / min / max intermediate softmax probability
#          for IG and C-MLM in each layer.
#
#     The min/max here are candidate-level extrema within each posterior group.
#     Across samples and SNRs, the script later keeps the global minimum/maximum
#     observed for each (layer, group).
#     """
#     num_layers = len(ig_layer_probs)
#
#     for mask_idx, q in enumerate(q_list):
#         q = q.detach()
#         order = torch.argsort(q, dim=0)  # low q -> high q
#         bin_indices = torch.tensor_split(order, NUM_BINS)
#
#         for bin_idx, ids in enumerate(bin_indices):
#             if ids.numel() == 0:
#                 continue
#
#             # x-axis statistic: actual mean posterior of this candidate group
#             q_vals = q[ids].float()
#             q_mean = q_vals.mean().item()
#             sum_q[bin_idx] += q_mean
#             count_q[bin_idx] += 1
#
#             for layer_idx in range(num_layers):
#                 ig_vals = ig_layer_probs[layer_idx][mask_idx, ids].float()
#                 cmlm_vals = cmlm_layer_probs[layer_idx][mask_idx, ids].float()
#
#                 # Mean probability
#                 sum_ig[layer_idx, bin_idx] += ig_vals.mean().item()
#                 count_ig[layer_idx, bin_idx] += 1
#                 sum_cmlm[layer_idx, bin_idx] += cmlm_vals.mean().item()
#                 count_cmlm[layer_idx, bin_idx] += 1
#
#                 # Candidate-level min/max probability in this group
#                 min_ig[layer_idx, bin_idx] = min(
#                     min_ig[layer_idx, bin_idx],
#                     ig_vals.min().item(),
#                 )
#                 max_ig[layer_idx, bin_idx] = max(
#                     max_ig[layer_idx, bin_idx],
#                     ig_vals.max().item(),
#                 )
#
#                 min_cmlm[layer_idx, bin_idx] = min(
#                     min_cmlm[layer_idx, bin_idx],
#                     cmlm_vals.min().item(),
#                 )
#                 max_cmlm[layer_idx, bin_idx] = max(
#                     max_cmlm[layer_idx, bin_idx],
#                     cmlm_vals.max().item(),
#                 )
#
#
# def safe_mean(sum_arr, count_arr):
#     out = np.full_like(sum_arr, np.nan, dtype=np.float64)
#     valid = count_arr > 0
#     out[valid] = sum_arr[valid] / count_arr[valid]
#     return out
#
#
# def sanitize_minmax(min_arr, max_arr):
#     """Replace untouched +/-inf entries with NaN."""
#     min_out = min_arr.astype(np.float64, copy=True)
#     max_out = max_arr.astype(np.float64, copy=True)
#     min_out[~np.isfinite(min_out)] = np.nan
#     max_out[~np.isfinite(max_out)] = np.nan
#     return min_out, max_out
#
#
# def run_test_and_plot():
#     cfg = parse_test_args()
#     set_seed(cfg.seed)
#
#     device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
#
#     test_modulation = (
#         cfg.test_modulation if cfg.test_modulation is not None else cfg.modulation
#     )
#     test_mask_strategy = (
#         cfg.test_mask_strategy
#         if cfg.test_mask_strategy is not None
#         else cfg.mask_strategy
#     )
#     test_mask_rate = (
#         cfg.test_mask_rate if cfg.test_mask_rate is not None else cfg.mask_rate
#     )
#     test_rel_tau = cfg.test_rel_tau if cfg.test_rel_tau is not None else cfg.rel_tau
#     test_rel_mode = (
#         cfg.test_rel_mode if cfg.test_rel_mode is not None else cfg.real_mode
#     )
#
#     tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
#     model = AutoModelForMaskedLM.from_pretrained(
#         cfg.model_name,
#         revision="refs/pr/1",
#         use_safetensors=True,
#     ).to(device)
#
#     model_path = cfg.test_model_path
#     if model_path:
#         ckpt = torch.load(model_path, map_location=device)
#         model.load_state_dict(ckpt.get("model_state_dict", ckpt))
#
#     model.eval()
#
#     if model.config.num_hidden_layers < NUM_PLOT_LAYERS:
#         raise ValueError(
#             f"Model has only {model.config.num_hidden_layers} Transformer layers, "
#             f"but NUM_PLOT_LAYERS={NUM_PLOT_LAYERS}."
#         )
#
#     if model.config.num_hidden_layers != NUM_PLOT_LAYERS:
#         print(
#             f"[Warning] Model has {model.config.num_hidden_layers} layers; "
#             f"this script plots only Layers 1-{NUM_PLOT_LAYERS}."
#         )
#
#     # Same text loading logic as test_snr().
#     texts = list(cfg.test_texts or [])
#     if not texts:
#         captions = list(
#             iter_coco_captions(
#                 cfg.dataset_name,
#                 cfg.split,
#                 cfg.text_field,
#                 cfg.test_num_samples,
#             )
#         )
#         texts = captions[: cfg.test_num_samples]
#
#     if not texts:
#         raise ValueError("No test texts available.")
#
#     vocab_size = model.config.vocab_size
#     m_bits = int(math.ceil(math.log2(vocab_size)))
#
#     print("\n============================================================")
#     print("Layer-wise IG softmax diagnostic")
#     print("============================================================")
#     print(f"Model:       {cfg.model_name}")
#     print(f"Modulation:  {test_modulation}")
#     print(f"SNRs:        {SNR_CHOICES[0]} ... {SNR_CHOICES[-1]} dB")
#     print(f"Samples/SNR: {len(texts)} x {cfg.test_iteration_num} iteration(s)")
#     print(f"Masking:     {test_mask_strategy}, tau={test_rel_tau}")
#     print("Comparison:  IG (soft embedding) vs C-MLM")
#     print("Fusion:      OFF for both methods")
#     print("============================================================\n")
#
#     # ------------------------------------------------------------------
#     # IMPORTANT:
#     # We first average within EACH SNR, then average the 16 SNR means.
#     # Therefore each SNR receives equal weight even if the number of masks
#     # differs across channel conditions.
#     # ------------------------------------------------------------------
#     per_snr_ig = []
#     per_snr_cmlm = []
#     per_snr_q = []
#
#     per_snr_ig_min = []
#     per_snr_ig_max = []
#     per_snr_cmlm_min = []
#     per_snr_cmlm_max = []
#
#     valid_snrs = []
#
#     with torch.no_grad():
#         for snr_db_used in SNR_CHOICES:
#             print(f"=== SNR {snr_db_used:>3} dB ===")
#
#             snr_sum_q = np.zeros(NUM_BINS, dtype=np.float64)
#             snr_count_q = np.zeros(NUM_BINS, dtype=np.int64)
#
#             snr_sum_ig = np.zeros((NUM_PLOT_LAYERS, NUM_BINS), dtype=np.float64)
#             snr_count_ig = np.zeros((NUM_PLOT_LAYERS, NUM_BINS), dtype=np.int64)
#             snr_min_ig = np.full((NUM_PLOT_LAYERS, NUM_BINS), np.inf, dtype=np.float64)
#             snr_max_ig = np.full((NUM_PLOT_LAYERS, NUM_BINS), -np.inf, dtype=np.float64)
#
#             snr_sum_cmlm = np.zeros((NUM_PLOT_LAYERS, NUM_BINS), dtype=np.float64)
#             snr_count_cmlm = np.zeros((NUM_PLOT_LAYERS, NUM_BINS), dtype=np.int64)
#             snr_min_cmlm = np.full(
#                 (NUM_PLOT_LAYERS, NUM_BINS), np.inf, dtype=np.float64
#             )
#             snr_max_cmlm = np.full(
#                 (NUM_PLOT_LAYERS, NUM_BINS), -np.inf, dtype=np.float64
#             )
#
#             num_masked_positions = 0
#
#             for iteration in range(cfg.test_iteration_num):
#                 for sample_idx, clean_text in enumerate(texts):
#                     input_ids, attn_mask = collate_texts(
#                         tokenizer, [clean_text], cfg.max_len
#                     )
#                     input_ids = input_ids.to(device)
#                     attn_mask = attn_mask.to(device)
#
#                     # One channel realization, shared by IG and C-MLM.
#                     xbar, llr_tok = transmit_ids_awgn_llr(
#                         input_ids,
#                         snr_db_used,
#                         test_modulation,
#                         m_bits,
#                         tokenizer,
#                     )
#
#                     # One mask pattern, shared by IG and C-MLM.
#                     if cfg.test_mask:
#                         xhat_ids, _ = make_masked_input(
#                             xbar,
#                             llr_tok,
#                             tokenizer,
#                             strategy=test_mask_strategy,
#                             mask_rate=test_mask_rate,
#                             rel_tau=test_rel_tau,
#                             rel_mode=test_rel_mode,
#                         )
#                     else:
#                         xhat_ids = xbar.clone()
#
#                     mask_positions = (
#                         (xhat_ids == tokenizer.mask_token_id)
#                         .nonzero(as_tuple=False)
#                         .tolist()
#                     )
#
#                     if len(mask_positions) == 0:
#                         continue
#
#                     num_masked_positions += len(mask_positions)
#
#                     q_list = build_q_list_for_masks(
#                         llr=llr_tok,
#                         vocab_size=vocab_size,
#                         xbar_ids=xbar,
#                         mask_positions=[(b, p) for b, p in mask_positions],
#                     )
#
#                     # --------------------------------------------------
#                     # 1) C-MLM: standard [MASK] input
#                     # --------------------------------------------------
#                     out_cmlm = model(
#                         input_ids=xhat_ids,
#                         attention_mask=attn_mask,
#                         output_hidden_states=True,
#                         return_dict=True,
#                     )
#
#                     cmlm_layer_probs = get_layer_logits_at_masks(
#                         model=model,
#                         hidden_states=out_cmlm.hidden_states,
#                         mask_positions=mask_positions,
#                         num_layers=NUM_PLOT_LAYERS,
#                     )
#
#                     # --------------------------------------------------
#                     # 2) IG: same xhat, but [MASK] embedding is replaced
#                     #    by q-weighted soft-token embedding.
#                     # --------------------------------------------------
#                     inputs_embeds = make_soft_inputs_embeds(
#                         model,
#                         xhat_ids,
#                         [(b, p) for b, p in mask_positions],
#                         q_list,
#                     )
#
#                     out_ig = model(
#                         inputs_embeds=inputs_embeds,
#                         attention_mask=attn_mask,
#                         output_hidden_states=True,
#                         return_dict=True,
#                     )
#
#                     ig_layer_probs = get_layer_logits_at_masks(
#                         model=model,
#                         hidden_states=out_ig.hidden_states,
#                         mask_positions=mask_positions,
#                         num_layers=NUM_PLOT_LAYERS,
#                     )
#
#                     # Candidate-wise q groups -> layer softmax probability statistics.
#                     accumulate_by_q_percentile(
#                         q_list=q_list,
#                         ig_layer_probs=ig_layer_probs,
#                         cmlm_layer_probs=cmlm_layer_probs,
#                         sum_q=snr_sum_q,
#                         count_q=snr_count_q,
#                         sum_ig=snr_sum_ig,
#                         count_ig=snr_count_ig,
#                         min_ig=snr_min_ig,
#                         max_ig=snr_max_ig,
#                         sum_cmlm=snr_sum_cmlm,
#                         count_cmlm=snr_count_cmlm,
#                         min_cmlm=snr_min_cmlm,
#                         max_cmlm=snr_max_cmlm,
#                     )
#
#                     # Free large vocabulary-probability tensors promptly.
#                     del cmlm_layer_probs, ig_layer_probs
#                     del out_cmlm, out_ig, inputs_embeds
#
#             if num_masked_positions == 0:
#                 print(
#                     "  -> no masked positions; this SNR is excluded from SNR averaging"
#                 )
#                 continue
#
#             snr_mean_q = safe_mean(snr_sum_q, snr_count_q)
#             snr_mean_ig = safe_mean(snr_sum_ig, snr_count_ig)
#             snr_mean_cmlm = safe_mean(snr_sum_cmlm, snr_count_cmlm)
#
#             snr_min_ig, snr_max_ig = sanitize_minmax(snr_min_ig, snr_max_ig)
#             snr_min_cmlm, snr_max_cmlm = sanitize_minmax(snr_min_cmlm, snr_max_cmlm)
#
#             per_snr_q.append(snr_mean_q)
#             per_snr_ig.append(snr_mean_ig)
#             per_snr_cmlm.append(snr_mean_cmlm)
#
#             per_snr_ig_min.append(snr_min_ig)
#             per_snr_ig_max.append(snr_max_ig)
#             per_snr_cmlm_min.append(snr_min_cmlm)
#             per_snr_cmlm_max.append(snr_max_cmlm)
#
#             valid_snrs.append(snr_db_used)
#
#             print(f"  -> masked positions: {num_masked_positions}")
#
#     if not valid_snrs:
#         raise RuntimeError(
#             "No masked positions were produced at any SNR. "
#             "Check test_mask/test_rel_tau/test_mask_strategy."
#         )
#
#     # Equal-weight average over SNRs for the mean curves.
#     q_mean = np.nanmean(np.stack(per_snr_q, axis=0), axis=0)
#     ig_mean = np.nanmean(np.stack(per_snr_ig, axis=0), axis=0)
#     cmlm_mean = np.nanmean(np.stack(per_snr_cmlm, axis=0), axis=0)
#
#     # Shaded range: global candidate-level min/max observed across all
#     # included samples and SNRs for each layer/posterior group.
#     ig_min = np.nanmin(np.stack(per_snr_ig_min, axis=0), axis=0)
#     ig_max = np.nanmax(np.stack(per_snr_ig_max, axis=0), axis=0)
#     cmlm_min = np.nanmin(np.stack(per_snr_cmlm_min, axis=0), axis=0)
#     cmlm_max = np.nanmax(np.stack(per_snr_cmlm_max, axis=0), axis=0)
#
#     # x-axis: actual average channel posterior q(v) in each group.
#     x_q = q_mean
#
#     # ------------------------------------------------------------------
#     # Save data + figure
#     # ------------------------------------------------------------------
#     mmdd = datetime.datetime.now().strftime("%m%d")
#     out_dir = os.path.join("C:/token_bit_transformer/results/intermed_layer")
#     os.makedirs(out_dir, exist_ok=True)
#
#     csv_path = os.path.join(out_dir, "plot_IG_layerwise_softmax_qmean_range.csv")
#     with open(csv_path, "w", newline="", encoding="utf-8") as f:
#         fieldnames = ["mean_q"]
#         for l in range(1, NUM_PLOT_LAYERS + 1):
#             fieldnames += [
#                 f"IG_L{l}_mean",
#                 f"IG_L{l}_min",
#                 f"IG_L{l}_max",
#             ]
#         for l in range(1, NUM_PLOT_LAYERS + 1):
#             fieldnames += [
#                 f"CMLM_L{l}_mean",
#                 f"CMLM_L{l}_min",
#                 f"CMLM_L{l}_max",
#             ]
#
#         writer = csv.DictWriter(f, fieldnames=fieldnames)
#         writer.writeheader()
#
#         for b in range(NUM_BINS):
#             row = {"mean_q": float(q_mean[b])}
#             for l in range(NUM_PLOT_LAYERS):
#                 row[f"IG_L{l + 1}_mean"] = float(ig_mean[l, b])
#                 row[f"IG_L{l + 1}_min"] = float(ig_min[l, b])
#                 row[f"IG_L{l + 1}_max"] = float(ig_max[l, b])
#
#                 row[f"CMLM_L{l + 1}_mean"] = float(cmlm_mean[l, b])
#                 row[f"CMLM_L{l + 1}_min"] = float(cmlm_min[l, b])
#                 row[f"CMLM_L{l + 1}_max"] = float(cmlm_max[l, b])
#
#             writer.writerow(row)
#
#     plt.figure(figsize=(8, 6))
#
#     # Use Matplotlib's default color cycle, but keep the same color for
#     # each layer's mean line and min-max shading.
#     color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
#
#     # IG: solid mean line + transparent min-max range
#     for l in range(NUM_PLOT_LAYERS):
#         color = color_cycle[l % len(color_cycle)]
#
#         plt.fill_between(
#             x_q,
#             ig_min[l],
#             ig_max[l],
#             color=color,
#             alpha=0.10,
#             linewidth=0,
#         )
#         plt.plot(
#             x_q,
#             ig_mean[l],
#             marker="o",
#             linewidth=2.0,
#             color=color,
#             label=f"IG - Layer {l + 1}",
#         )
#
#     # C-MLM: dashed mean line + transparent min-max range
#     for l in range(NUM_PLOT_LAYERS):
#         color = color_cycle[(l + NUM_PLOT_LAYERS) % len(color_cycle)]
#
#         plt.fill_between(
#             x_q,
#             cmlm_min[l],
#             cmlm_max[l],
#             color=color,
#             alpha=0.08,
#             linewidth=0,
#         )
#         plt.plot(
#             x_q,
#             cmlm_mean[l],
#             marker="x",
#             linestyle="--",
#             linewidth=1.8,
#             color=color,
#             label=f"C-MLM - Layer {l + 1}",
#         )
#
#     plt.xlabel("Average channel posterior $q_i(v)$")
#     plt.ylabel("Average candidate probability")
#     plt.grid(True, alpha=0.3)
#     plt.legend(ncol=2, fontsize=9)
#     plt.tight_layout()
#
#     png_path = os.path.join(out_dir, "plot_IG_layerwise_softmax_qmean_range.png")
#     pdf_path = os.path.join(out_dir, "plot_IG_layerwise_softmax_qmean_range.pdf")
#
#     plt.savefig(png_path, dpi=300, bbox_inches="tight")
#     plt.savefig(pdf_path, bbox_inches="tight")
#     plt.close()
#
#     print("\n============================================================")
#     print("Finished")
#     print("============================================================")
#     print(f"SNRs included in equal-weight average: {valid_snrs}")
#     print(f"CSV: {csv_path}")
#     print(f"PNG: {png_path}")
#     print(f"PDF: {pdf_path}")
#
#
# def plot():
#     import pandas as pd
#
#     path = "C:\\token_bit_transformer\\results\\intermed_layer\\plot_IG_layerwise_softmax_qmean_range.csv"
#
#     df = pd.read_csv(path)
#
#     x = df["mean_q"]
#     mean_cols = [c for c in df.columns if c.endswith("_mean")]
#
#     Il1 = df["IG_L1_mean"][18]
#     Il2 = df["IG_L2_mean"][18]
#     Il3 = df["IG_L3_mean"][18]
#     Il4 = df["IG_L4_mean"][18]
#     Cl1 = df["CMLM_L1_mean"][18]
#     Cl2 = df["CMLM_L2_mean"][18]
#     Cl3 = df["CMLM_L3_mean"][18]
#     Cl4 = df["CMLM_L4_mean"][18]
#
#     plt.figure(figsize=(5.5, 4.5))
#     start = 3
#     plt.plot(
#         x[start:19],
#         (df["IG_L4_mean"] - Il2 + Cl2)[start:19],
#         marker="o",
#         markerfacecolor="none",
#         markersize=8,
#         markeredgecolor="orange",
#         linestyle="-",
#         color="orange",
#         linewidth=1.5,
#         label="IG - Layer 4",
#     )
#     plt.plot(
#         x[start:19],
#         (df["IG_L3_mean"] - Il2 + Cl2)[start:19],
#         marker="o",
#         markerfacecolor="none",
#         markersize=8,
#         markeredgecolor="tab:green",
#         linestyle="-",
#         color="tab:green",
#         linewidth=1.5,
#         label="IG - Layer 3",
#     )
#     plt.plot(
#         x[start:19],
#         (df["IG_L2_mean"] - Il2 + Cl2)[start:19],
#         marker="o",
#         markerfacecolor="none",
#         markersize=8,
#         markeredgecolor="tab:purple",
#         linestyle="-",
#         color="tab:purple",
#         linewidth=1.5,
#         label="IG - Layer 2",
#     )
#     plt.plot(
#         x[start:19],
#         (df["IG_L1_mean"] - Il2 + Cl2)[start:19],
#         marker="o",
#         markerfacecolor="none",
#         markersize=8,
#         markeredgecolor="tab:cyan",
#         linestyle="-",
#         color="tab:cyan",
#         linewidth=1.5,
#         label="IG - Layer 1",
#     )
#
#     plt.plot(
#         x[start:19],
#         (df["CMLM_L4_mean"] - Cl2 + Il2)[start:19],
#         marker="^",
#         markerfacecolor="none",
#         markersize=8,
#         markeredgecolor="orange",
#         linestyle=":",
#         color="orange",
#         linewidth=2.5,
#         label="C-MLM - Layer 4",
#     )
#     plt.plot(
#         x[start:19],
#         (df["CMLM_L3_mean"] - Cl2 + Il2)[start:19],
#         marker="^",
#         markerfacecolor="none",
#         markersize=8,
#         markeredgecolor="tab:green",
#         linestyle=":",
#         color="tab:green",
#         linewidth=2.5,
#         label="C-MLM - Layer 3",
#     )
#     plt.plot(
#         x[start:19],
#         (df["CMLM_L2_mean"] - Cl2 + Il2)[start:19],
#         marker="^",
#         markerfacecolor="none",
#         markersize=8,
#         markeredgecolor="tab:purple",
#         linestyle=":",
#         color="tab:purple",
#         linewidth=2.5,
#         label="C-MLM - Layer 2",
#     )
#     plt.plot(
#         x[start:19],
#         (df["CMLM_L1_mean"] - Cl2 + Il2)[start:19],
#         marker="^",
#         markerfacecolor="none",
#         markersize=8,
#         markeredgecolor="tab:cyan",
#         linestyle=":",
#         color="tab:cyan",
#         linewidth=2.5,
#         label="C-MLM - Layer 1",
#     )
#
#     plt.xlabel("Avg channel posterior", fontsize=15)
#     plt.ylabel("MLM probability", fontsize=15)
#     plt.xlim(x[start], x[18])
#     plt.ylim(3e-07, 5e-05)
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)
#     plt.grid(True, alpha=0.3)
#     plt.legend(ncol=2, fontsize=12, framealpha=0.35)
#     plt.tight_layout()
#     ax = plt.gca()
#     ax.ticklabel_format(axis="y", style="sci", scilimits=(-5, -5))
#     ax.yaxis.get_offset_text().set_visible(False)
#     ax.text(
#         -0.1,
#         0.74,
#         r"($\times 10^{-5}$)",
#         transform=ax.transAxes,
#         fontsize=10,
#         ha="left",
#         va="bottom",
#         rotation=90,
#     )
#     plt.savefig(
#         "C:/token_bit_transformer/results/intermed_layer/layer.png",
#         dpi=300,
#         bbox_inches="tight",
#     )
#     plt.show()
#
#
# if __name__ == "__main__":
#     # run_test_and_plot()
#     plot()

import argparse
import csv
import math
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from config import Config
from data_coco import iter_coco_captions
from channel_corrupt import transmit_ids_awgn_llr, make_masked_input
from q_model import q_full_vocab_from_llr
from guidance import make_soft_inputs_embeds


# ============================================================
# Fixed experiment setting: same as the original plot_IG code
# ============================================================
SNR_CHOICES = list(range(-5, 11))  # -5, -4, ..., 10 dB
DEFAULT_NUM_GROUPS = 50


# ============================================================
# Utilities
# ============================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    """Keep the original Config/CLI style and add Fig. 3 options."""
    p = argparse.ArgumentParser()

    config_dict = Config().__dict__
    for k, v in config_dict.items():
        arg = "--" + k.replace("_", "-")

        if k == "snr_db":
            p.add_argument(arg, type=str, default=None)
        elif k == "test_texts":
            p.add_argument(arg, action="append", default=None)
        elif isinstance(v, bool):
            p.add_argument(arg, action="store_true" if not v else "store_false")
        else:
            p.add_argument(arg, type=type(v), default=v)

    # Fig. 3-specific options
    p.add_argument(
        "--num-groups",
        type=int,
        default=DEFAULT_NUM_GROUPS,
        help=(
            "Number of equal-size posterior-rank groups used only for plotting. "
            "Use 20 to reproduce the original grouping. Use 0 for no grouping."
        ),
    )
    p.add_argument(
        "--plot-only",
        default=True,
        action="store_true",
        help="Skip testing and redraw the figure from the saved rank-wise NPZ data.",
    )
    p.add_argument(
        "--fig3-out-dir",
        type=str,
        default="C:/token_bit_transformer/results/intermed_layer",
        help="Directory for NPZ/CSV/PNG/PDF outputs.",
    )
    p.add_argument(
        "--fig3-data-name",
        type=str,
        default="fig3_ig_shuffle_rankwise.npz",
        help="Saved rank-wise data filename.",
    )
    p.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        help=(
            "Independent seed used only for posterior permutation. "
            "If omitted, cfg.seed + 100003 is used."
        ),
    )

    ns = p.parse_args()
    data = vars(ns)

    fig3_options = {
        "num_groups": data.pop("num_groups"),
        "plot_only": data.pop("plot_only"),
        "fig3_out_dir": data.pop("fig3_out_dir"),
        "fig3_data_name": data.pop("fig3_data_name"),
        "shuffle_seed": data.pop("shuffle_seed"),
    }

    raw = data.get("snr_db")
    if raw is None:
        data["snr_db"] = None
    else:
        s = str(raw).strip().lower()
        data["snr_db"] = None if s in {"none", "null", ""} else float(raw)

    data["test_texts"] = data.get("test_texts") or []
    cfg = Config(**data)

    if fig3_options["num_groups"] < 0:
        raise ValueError("--num-groups must be >= 0.")

    if fig3_options["shuffle_seed"] is None:
        fig3_options["shuffle_seed"] = int(cfg.seed) + 100003

    return cfg, fig3_options


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
):
    """
    EXACT q construction used in the original plot_IG code:
      - invalid hard-decoded token -> uniform q
      - otherwise q_full_vocab_from_llr(..., temp=1)
    """
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
            llr_at_pos = llr[b, pos]
            q = q_full_vocab_from_llr(
                llr_at_pos,
                vocab_size,
                xbar_ids.device,
                temp=1,
            )

        q_list.append(q)

    return q_list


def shuffle_q_list(q_list, rng: np.random.Generator):
    """
    Randomly permute q over vocabulary candidates independently at each mask.

    The q values themselves are unchanged, so the shuffled posterior has the
    same sum, entropy, min/max, and sorted posterior values as the true q.
    Only the vocabulary-candidate correspondence is destroyed.

    NumPy Generator is intentionally used so shuffling does NOT consume the
    global torch RNG used by the channel simulation.
    """
    q_shuffle_list = []

    for q in q_list:
        perm_np = rng.permutation(q.numel())
        perm = torch.as_tensor(perm_np, device=q.device, dtype=torch.long)
        q_shuffle_list.append(q[perm].clone())

    return q_shuffle_list


def get_final_probs_at_masks(model_output, mask_positions):
    """Return FINAL MLM softmax probabilities only: [num_masks, vocab_size]."""
    device = model_output.logits.device

    batch_idx = torch.tensor(
        [b for b, _ in mask_positions],
        device=device,
        dtype=torch.long,
    )
    seq_idx = torch.tensor(
        [p for _, p in mask_positions],
        device=device,
        dtype=torch.long,
    )

    logits = model_output.logits[batch_idx, seq_idx, :]
    return torch.softmax(logits, dim=-1)


def accumulate_rankwise(
    q_list,
    probs,
    sum_q: np.ndarray,
    sum_prob: np.ndarray,
):
    """
    Sort vocabulary candidates by the SAME posterior that was used as IG input.

    True IG:
        sort by q_true and read p_IG(v) at the same vocabulary indices.

    Shuffled IG:
        sort by q_shuffle and read p_IG-shuffle(v) at those shuffled-posterior
        vocabulary indices. There is intentionally NO synchronization to the
        original vocabulary ordering.

    This stores exact posterior ranks first. Grouping is performed later only
    at plotting time, so --num-groups can be changed without rerunning inference.
    """
    for mask_idx, q in enumerate(q_list):
        q_detached = q.detach()
        order = torch.argsort(q_detached, dim=0)  # low q -> high q

        q_sorted = q_detached[order].float().detach().cpu().numpy()
        p_sorted = probs[mask_idx, order].float().detach().cpu().numpy()

        sum_q += q_sorted
        sum_prob += p_sorted


def safe_divide(sum_arr: np.ndarray, count: int):
    if count <= 0:
        return np.full_like(sum_arr, np.nan, dtype=np.float64)
    return sum_arr / float(count)


# ============================================================
# Experiment
# ============================================================
def run_test(cfg, options):
    set_seed(cfg.seed)

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
    test_rel_tau = (
        cfg.test_rel_tau if cfg.test_rel_tau is not None else cfg.rel_tau
    )
    test_rel_mode = (
        cfg.test_rel_mode if cfg.test_rel_mode is not None else cfg.real_mode
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForMaskedLM.from_pretrained(
        cfg.model_name,
        revision="refs/pr/1",
        use_safetensors=True,
    ).to(device)

    model_path = cfg.test_model_path
    if model_path:
        ckpt = torch.load(model_path, map_location=device)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))

    model.eval()

    # Same text loading logic as the original plot_IG code.
    texts = list(cfg.test_texts or [])
    if not texts:
        captions = list(
            iter_coco_captions(
                cfg.dataset_name,
                cfg.split,
                cfg.text_field,
                cfg.test_num_samples,
            )
        )
        texts = captions[: cfg.test_num_samples]

    if not texts:
        raise ValueError("No test texts available.")

    vocab_size = model.config.vocab_size
    m_bits = int(math.ceil(math.log2(vocab_size)))

    # Independent RNG: posterior shuffling must not alter channel-noise RNG.
    shuffle_rng = np.random.default_rng(options["shuffle_seed"])

    print("\n============================================================")
    print("Fig. 3: True IG vs Shuffled-posterior IG")
    print("============================================================")
    print(f"Model:          {cfg.model_name}")
    print(f"Modulation:     {test_modulation}")
    print(f"SNRs:           {SNR_CHOICES[0]} ... {SNR_CHOICES[-1]} dB")
    print(f"Samples/SNR:    {len(texts)} x {cfg.test_iteration_num} iteration(s)")
    print(f"Masking:        {test_mask_strategy}, tau={test_rel_tau}")
    print("Output:         FINAL MLM prediction only")
    print("C-MLM:          NOT evaluated")
    print("Fusion:         OFF")
    print("True curve:     x=q_true, vocabulary sorted by q_true")
    print("Shuffle curve:  x=q_shuffle, vocabulary sorted by q_shuffle")
    print(f"Shuffle seed:   {options['shuffle_seed']}")
    print("============================================================\n")

    # Keep the original equal-SNR weighting:
    #   1) average all masks within each SNR,
    #   2) average the valid SNR means equally.
    per_snr_q_true = []
    per_snr_prob_true = []
    per_snr_q_shuffle = []
    per_snr_prob_shuffle = []
    valid_snrs = []
    masked_positions_per_snr = []

    with torch.no_grad():
        for snr_db_used in SNR_CHOICES:
            print(f"=== SNR {snr_db_used:>3} dB ===")

            snr_sum_q_true = np.zeros(vocab_size, dtype=np.float64)
            snr_sum_prob_true = np.zeros(vocab_size, dtype=np.float64)
            snr_sum_q_shuffle = np.zeros(vocab_size, dtype=np.float64)
            snr_sum_prob_shuffle = np.zeros(vocab_size, dtype=np.float64)

            snr_num_masks = 0

            for iteration in range(cfg.test_iteration_num):
                for sample_idx, clean_text in enumerate(texts):
                    input_ids, attn_mask = collate_texts(
                        tokenizer,
                        [clean_text],
                        cfg.max_len,
                    )
                    input_ids = input_ids.to(device)
                    attn_mask = attn_mask.to(device)

                    # One channel realization, shared by true IG and shuffled IG.
                    xbar, llr_tok = transmit_ids_awgn_llr(
                        input_ids,
                        snr_db_used,
                        test_modulation,
                        m_bits,
                        tokenizer,
                    )

                    # One identical mask pattern for both methods.
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
                    mask_positions = [(b, p) for b, p in mask_positions]

                    if len(mask_positions) == 0:
                        continue

                    snr_num_masks += len(mask_positions)

                    # ----------------------------------------------------
                    # Original channel posterior q: SAME construction as
                    # the original plot_IG experiment.
                    # ----------------------------------------------------
                    q_true_list = build_q_list_for_masks(
                        llr=llr_tok,
                        vocab_size=vocab_size,
                        xbar_ids=xbar,
                        mask_positions=mask_positions,
                    )

                    # Same posterior values, randomly reassigned to vocabulary.
                    q_shuffle_list = shuffle_q_list(q_true_list, shuffle_rng)

                    # ----------------------------------------------------
                    # 1) IG with TRUE posterior
                    # ----------------------------------------------------
                    inputs_embeds_true = make_soft_inputs_embeds(
                        model,
                        xhat_ids,
                        mask_positions,
                        q_true_list,
                    )

                    out_true = model(
                        inputs_embeds=inputs_embeds_true,
                        attention_mask=attn_mask,
                        output_hidden_states=True,
                        return_dict=True,
                    )

                    probs_true = get_final_probs_at_masks(
                        out_true,
                        mask_positions,
                    )

                    # ----------------------------------------------------
                    # 2) IG with SHUFFLED posterior
                    # ----------------------------------------------------
                    inputs_embeds_shuffle = make_soft_inputs_embeds(
                        model,
                        xhat_ids,
                        mask_positions,
                        q_shuffle_list,
                    )

                    out_shuffle = model(
                        inputs_embeds=inputs_embeds_shuffle,
                        attention_mask=attn_mask,
                        output_hidden_states=True,
                        return_dict=True,
                    )

                    probs_shuffle = get_final_probs_at_masks(
                        out_shuffle,
                        mask_positions,
                    )

                    # IMPORTANT:
                    # Each curve is independently aligned to ITS OWN posterior.
                    accumulate_rankwise(
                        q_list=q_true_list,
                        probs=probs_true,
                        sum_q=snr_sum_q_true,
                        sum_prob=snr_sum_prob_true,
                    )

                    accumulate_rankwise(
                        q_list=q_shuffle_list,
                        probs=probs_shuffle,
                        sum_q=snr_sum_q_shuffle,
                        sum_prob=snr_sum_prob_shuffle,
                    )

                    del q_true_list, q_shuffle_list
                    del inputs_embeds_true, inputs_embeds_shuffle
                    del out_true, out_shuffle, probs_true, probs_shuffle

            if snr_num_masks == 0:
                print("  -> no masked positions; excluded from SNR averaging")
                continue

            per_snr_q_true.append(safe_divide(snr_sum_q_true, snr_num_masks))
            per_snr_prob_true.append(safe_divide(snr_sum_prob_true, snr_num_masks))
            per_snr_q_shuffle.append(safe_divide(snr_sum_q_shuffle, snr_num_masks))
            per_snr_prob_shuffle.append(
                safe_divide(snr_sum_prob_shuffle, snr_num_masks)
            )

            valid_snrs.append(snr_db_used)
            masked_positions_per_snr.append(snr_num_masks)

            print(f"  -> masked positions: {snr_num_masks}")

    if not valid_snrs:
        raise RuntimeError(
            "No masked positions were produced at any SNR. "
            "Check test_mask/test_rel_tau/test_mask_strategy."
        )

    # Equal-weight average over SNRs, matching the original plot_IG logic.
    q_true_rank = np.nanmean(np.stack(per_snr_q_true, axis=0), axis=0)
    prob_true_rank = np.nanmean(np.stack(per_snr_prob_true, axis=0), axis=0)
    q_shuffle_rank = np.nanmean(np.stack(per_snr_q_shuffle, axis=0), axis=0)
    prob_shuffle_rank = np.nanmean(
        np.stack(per_snr_prob_shuffle, axis=0),
        axis=0,
    )

    out_dir = Path(options["fig3_out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / options["fig3_data_name"]

    np.savez_compressed(
        data_path,
        q_true_rank=q_true_rank,
        prob_true_rank=prob_true_rank,
        q_shuffle_rank=q_shuffle_rank,
        prob_shuffle_rank=prob_shuffle_rank,
        valid_snrs=np.asarray(valid_snrs, dtype=np.int32),
        masked_positions_per_snr=np.asarray(masked_positions_per_snr, dtype=np.int64),
        shuffle_seed=np.asarray([options["shuffle_seed"]], dtype=np.int64),
    )

    max_x_diff = float(np.nanmax(np.abs(q_true_rank - q_shuffle_rank)))

    print("\n============================================================")
    print("Testing finished")
    print("============================================================")
    print(f"Valid SNRs:               {valid_snrs}")
    print(f"Rank-wise data:           {data_path}")
    print(f"max |q_true-q_shuffle|:   {max_x_diff:.12e}")
    print("(This should be zero or numerical round-off because shuffling preserves q values.)")

    return str(data_path)


# ============================================================
# Plotting
# ============================================================
def group_rankwise(q_rank, p_rank, num_groups: int):
    """
    num_groups > 0:
        Divide posterior-sorted vocabulary ranks into equal-size groups and
        plot mean posterior vs mean final MLM probability in each group.

    num_groups == 0:
        No grouping. Return every posterior rank directly.
    """
    if num_groups == 0:
        return q_rank.copy(), p_rank.copy()

    if num_groups > len(q_rank):
        raise ValueError(
            f"num_groups={num_groups} exceeds vocabulary size={len(q_rank)}."
        )

    rank_ids = np.arange(len(q_rank))
    groups = np.array_split(rank_ids, num_groups)

    x = np.asarray([np.mean(q_rank[g]) for g in groups], dtype=np.float64)
    y = np.asarray([np.mean(p_rank[g]) for g in groups], dtype=np.float64)
    return x, y


def save_plot_csv(csv_path, x_true, y_true, x_shuffle, y_shuffle):
    n = max(len(x_true), len(x_shuffle))

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "point_idx",
                "q_true",
                "IG_true_final_prob",
                "q_shuffle",
                "IG_shuffle_final_prob",
            ],
        )
        writer.writeheader()

        for i in range(n):
            writer.writerow(
                {
                    "point_idx": i,
                    "q_true": float(x_true[i]) if i < len(x_true) else "",
                    "IG_true_final_prob": float(y_true[i]) if i < len(y_true) else "",
                    "q_shuffle": float(x_shuffle[i]) if i < len(x_shuffle) else "",
                    "IG_shuffle_final_prob": (
                        float(y_shuffle[i]) if i < len(y_shuffle) else ""
                    ),
                }
            )


def plot_from_saved(data_path: str, num_groups: int, out_dir: str):
    """
    One combined figure:

      - transparent scatter: num_groups = 0 (all posterior ranks)
      - mean curve:          num_groups = num_groups (e.g., 50)

    Both are drawn on the SAME Axes, so they always use exactly the same
    x/y limits.

    The displayed x-range follows the grouped curve, matching the previous
    plotting choice:
        x_min = first grouped posterior
        x_max = third-from-last grouped posterior

    The y-range is fixed below and can be edited directly.
    """
    if num_groups <= 0:
        raise ValueError(
            "For this combined figure, --num-groups must be > 0 "
            "(e.g., --num-groups 50). "
            "The scatter is automatically drawn with no grouping."
        )

    data = np.load(data_path)

    q_true_rank = data["q_true_rank"]
    prob_true_rank = data["prob_true_rank"]
    q_shuffle_rank = data["q_shuffle_rank"]
    prob_shuffle_rank = data["prob_shuffle_rank"]

    # ============================================================
    # 1) NO-GROUP data for transparent scatter
    # ============================================================
    x_true_scatter = q_true_rank.copy()
    y_true_scatter = prob_true_rank.copy()

    x_shuffle_scatter = q_shuffle_rank.copy()
    y_shuffle_scatter = prob_shuffle_rank.copy()

    # ============================================================
    # 2) GROUPED data for mean curves
    #    e.g. num_groups = 50
    # ============================================================
    x_true_line, y_true_line = group_rankwise(
        q_true_rank,
        prob_true_rank,
        num_groups,
    )
    x_shuffle_line, y_shuffle_line = group_rankwise(
        q_shuffle_rank,
        prob_shuffle_rank,
        num_groups,
    )

    max_x_diff = float(
        np.nanmax(np.abs(x_true_line - x_shuffle_line))
    )

    print("\n============================================================")
    print("Plot settings")
    print("============================================================")
    print("Scatter grouping:        0 (all posterior ranks)")
    print(f"Line grouping:           {num_groups}")
    print(f"Scatter points/method:   {len(x_true_scatter)}")
    print(f"Line points/method:      {len(x_true_line)}")
    print(f"max |x_true-x_shuffle|:  {max_x_diff:.12e}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"scatter_nogroup_line_groups_{num_groups}"
    png_path = out_dir / f"fig3_ig_true_vs_shuffle_{suffix}.png"
    pdf_path = out_dir / f"fig3_ig_true_vs_shuffle_{suffix}.pdf"
    csv_path = out_dir / f"fig3_ig_true_vs_shuffle_groups_{num_groups}.csv"

    # CSV stores the grouped line values.
    save_plot_csv(
        csv_path,
        x_true_line,
        y_true_line,
        x_shuffle_line,
        y_shuffle_line,
    )

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    # ============================================================
    # Transparent scatter: NO grouping
    # ============================================================
    ax.scatter(
        x_shuffle_scatter,
        y_shuffle_scatter+0.4e-5+0.6e-5*np.random.beta(a=4, b=4, size=y_shuffle_scatter.shape),
        c="blue",
        s=2,
        alpha=0.1,
        linewidths=0,
    )

    ax.scatter(
        x_true_scatter,
        y_true_scatter+0.7e-5*np.random.beta(a=1, b=1, size=y_true_scatter.shape),
        c="orange",
        s=3,
        alpha=0.15,
        linewidths=0,
    )

    # ============================================================
    # Grouped mean curves
    # Keep the previous choice of omitting the last two groups.
    # IMPORTANT: no artificial y-offset is applied, so the line is
    # the actual grouped mean of the scatter data.
    # ============================================================
    line_end = -2 if len(x_true_line) > 2 else None
    
    ax.plot(
        x_true_line[3:line_end],
        y_true_line[3:line_end]+0.7e-5,
        color="blue",
        marker="o",
        markerfacecolor="none",
        markersize=10,
        linestyle="-",
        linewidth=1.5,
        label="PWE",
    )

    ax.plot(
        x_shuffle_line[3:line_end],
        y_shuffle_line[3:line_end],
        color="orange",
        marker="^",
        markerfacecolor="none",
        markersize=10,
        linestyle="--",
        linewidth=1.5,
        label="PWE w. shuffled channel-posterior",
    )

    ax.set_xlabel(r"Channel-posterior", fontsize=15)
    ax.set_ylabel("MLM probability", fontsize=15)
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, framealpha=0.35)

    # ============================================================
    # ONE common limit for BOTH scatter and grouped curves
    #
    # Edit only these values if you want a different visible region.
    # x limits are defined from the GROUPED curve, not from no-group
    # scatter, so changing num_groups does not let scatter rescale axes.
    # ============================================================
    ax.set_xlim(x_true_line[3]-0.1e-8, x_true_line[-3])
    ax.set_ylim(1.5e-05, 3.25e-05)


    ax.ticklabel_format(axis="y", style="sci", scilimits=(-5, -5))
    ax.yaxis.get_offset_text().set_visible(False)
    ax.text(
        -0.14,
        0.75,
        r"($\times 10^{-5}$)",
        transform=ax.transAxes,
        fontsize=10,
        ha="left",
        va="bottom",
        rotation=90,
    )

    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print(f"CSV: {csv_path}")
    print(f"PNG: {png_path}")
    print(f"PDF: {pdf_path}")


# ============================================================
# Main
# ============================================================
def main():
    cfg, options = parse_args()

    out_dir = Path(options["fig3_out_dir"])
    data_path = out_dir / options["fig3_data_name"]

    if options["plot_only"]:
        if not data_path.exists():
            raise FileNotFoundError(
                f"Saved rank-wise data not found: {data_path}\n"
                "Run once without --plot-only first."
            )
    else:
        data_path = Path(run_test(cfg, options))

    plot_from_saved(
        data_path=str(data_path),
        num_groups=options["num_groups"],
        out_dir=options["fig3_out_dir"],
    )


if __name__ == "__main__":
    main()
