# config.py
from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 30522
    d_model: int = 256
    n_heads: int = 4
    dropout: float = 0.1

    n_layers_enc: int = 3
    n_layers_dec: int = 4

    # Variable-length bits per token
    l_min: int = 1
    l_max: int = 31  # length choices: [1..31]

    # Conditioning memory
    n_state_tokens: int = 4
    n_imp_tokens: int = 2


@dataclass
class TrainConfig:
    batch_size: int = 1
    seq_len: int = 64
    steps: int = 2000
    lr: float = 2e-4
    steps_per_epoch: int = 1000

    # rate/importance
    lambda_rate: float = 0.02
    alpha_imp: float = 1.5  # importance normalization sharpening


@dataclass
class DataConfig:
    dataset_name: str = "wikitext"
    dataset_config_name: str = "wikitext-2-raw-v1"
    tokenizer_name: str = "bert-base-uncased"
    cache_dir: str = "/app/huggingface_cache"


@dataclass
class ChannelConfig:
    snr_db: float = 10.0
    modulation: str = "BPSK"


@dataclass
class LogConfig:
    csv_name: str = "batch_logs.csv"
    bert_score_model: str = "roberta-large"
    bert_score_max_length: int = 512


@dataclass
class WandbConfig:
    enabled: bool = True
    project: str = "token-bit-transformer"
    run_name: str = "base"
