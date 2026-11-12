from dataclasses import dataclass, field
import argparse


@dataclass
class Config: # | None = None
    # model/data
    model_name: str = "google/bert_uncased_L-4_H-512_A-8"  # bert-base-uncased
    dataset_name: str = "HuggingFaceM4/COCO"  # alt: "whyen-wang/coco_captions"  HuggingFaceM4/COCO #nlphuji/flickr30k
    split: str = "train"
    text_field: str = "caption"  # HuggingFaceM4/COCO: list[str]
    max_samples: int = 5000  # keep small for quick experiments
    max_len: int = 32

    # corruption / channel
    snr_db: float | None = None  # SNR in dB (None -> uniform random in [-5,10])
    modulation: str = "qpsk"  # bpsk | qpsk | 16qam | 64qam
    mask_strategy: str = "reliability"  # "random" or "reliability"
    mask_rate: float = 0.15  # used when mask_strategy=="random"
    rel_tau: float = 1.0  # used when mask_strategy=="reliability"
    real_mode: str = "minabs"  # "minabs" | "meanabs"

    # guidance
    guide: str = "soft_embed"  # "none" | "attn_bias" | "soft_embed"
    use_fusion: bool = True  # "True" (1-A) adaptive full-vocab fusion
    attn_beta_init: float = 1.0  # scale for 2-A bias

    # training
    lr: float = 5e-5
    epochs: int =500
    log_every: int = 50
    seed: int = 42
    device: str = "cuda"  # "cuda" or "cpu"

    # test
    test_batch_size: int = 1 #6
    test_model_path: str = None #"C:/token_bit_transformer/results/models/fusion_True__guide_soft_embed__snr_random__mod_bpsk_20260210_095631.pth"
    test_texts: list[str] = field(default_factory=list)
    test_num_samples: int = 50
    test_snr_db: float = 2.0
    test_modulation: str = "16qam"
    test_mask: bool = True
    test_mask_strategy: str = "reliability"
    test_mask_rate: float =0.15
    test_rel_tau: float = 4
    test_rel_mode: str = "minabs"
    test_guide: str = "None" #"soft_embed" # "None"
    test_use_fusion: bool = True #True
    test_attn_beta_init: float = 1.0
    test_iteration_num: int = 30  # number of times to run through the data loader
    test_csv_name: str = f"q_llr__no_outputfusion__16QAM_snr_-5to10__temp_no" #2.5 다시해야돼

def parse_args() -> Config:
    p = argparse.ArgumentParser()
    for k, v in Config().__dict__.items():
        arg = "--" + k.replace("_", "-")
        if k in {"snr_db", "test_snr_db"}:
            p.add_argument(arg, type=str, default=None)
        elif k == "test_texts":
            p.add_argument(arg, action="append", default=None)
        elif isinstance(v, bool):
            p.add_argument(arg, action="store_true" if not v else "store_false")
        else:
            p.add_argument(arg, type=type(v), default=v)
    ns = p.parse_args()
    data = vars(ns)
    if "snr_db" in data:
        raw = data["snr_db"]
        if raw is None:
            data["snr_db"] = None
        else:
            s = str(raw).strip().lower()
            data["snr_db"] = None if s in {"none", "null", ""} else float(raw)
    if "test_snr_db" in data:
        raw = data["test_snr_db"]
        if raw is None:
            data["test_snr_db"] = None
        else:
            s = str(raw).strip().lower()
            data["test_snr_db"] = None if s in {"none", "null", ""} else float(raw)
    if "test_texts" in data:
        data["test_texts"] = data["test_texts"] or []
    return Config(**data)
