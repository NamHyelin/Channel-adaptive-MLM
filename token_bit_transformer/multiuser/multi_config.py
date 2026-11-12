from dataclasses import dataclass, field
import argparse


@dataclass
class Config:
    # model/data
    model_name: str = "google/bert_uncased_L-4_H-512_A-8"
    dataset_name: str = "HuggingFaceM4/COCO"
    split: str = "train"
    text_field: str = "caption"
    max_samples: int = 5000
    max_len: int = 32

    # channel / CQI boundary
    snr_db: float | None = None
    test_snr_values: str = "-5:10:1"  # start:end:step inclusive

    # multi-user
    multi_num_users: int = 2
    multi_interferer_mode: str = "roll"  # roll | random_permute
    multi_channel_gain_mode: str = "rayleigh"  # rayleigh | ones
    multi_pilot_tokens: int = 0  # tail pilot tokens ignored
    multi_power_delta: float = 0.1  # power gap from max power(=1.0). user-k power: P_k = random_uniform(1.0 - k*delta, 1.0)

    # test
    seed: int = 35  # 42
    device: str = "cuda"
    test_batch_size: int = 16
    test_model_path: str = "C:/token_bit_transformer/results/models/fusion_True__guide_soft_embed__snr_random__mod_bpsk_20260210_095631.pth"
    test_texts: list[str] = field(default_factory=list)
    test_num_samples: int = 25
    test_snr_db: float | None = 0.0
    test_modulation: str = "bpsk"
    test_mask_strategy: str = "reliability"
    test_mask_rate: float = 0.15
    test_rel_tau: float = 1.0
    test_rel_mode: str = "minabs"
    test_guide: str = "soft_embed"
    test_use_fusion: bool = True
    test_attn_beta_init: float = 1.0


def parse_args() -> Config:
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

    if int(data["multi_num_users"]) < 1:
        raise ValueError("multi_num_users must be >= 1")
    if int(data["multi_pilot_tokens"]) < 0:
        raise ValueError("multi_pilot_tokens must be >= 0")
    if float(data["multi_power_delta"]) < 0.0:
        raise ValueError("multi_power_delta must be >= 0")

    return Config(**data)
