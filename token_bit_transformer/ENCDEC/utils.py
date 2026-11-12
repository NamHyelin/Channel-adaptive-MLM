import math
import torch


def _q_function(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.erfc(x / math.sqrt(2.0))


def ber_from_snr(snr_db: torch.Tensor, modulation: str) -> torch.Tensor:
    if not torch.is_tensor(snr_db):
        snr_db = torch.tensor(snr_db, dtype=torch.float32)

    snr_linear = torch.pow(10.0, snr_db / 10.0)
    mod = modulation.upper()

    if mod in {"BPSK", "QPSK"}:
        ber = _q_function(torch.sqrt(2.0 * snr_linear))
    elif mod in {"16QAM", "16-QAM", "QAM16"}:
        m = 16.0
        k = math.log2(m)
        ber = (4.0 / k) * (1.0 - 1.0 / math.sqrt(m)) * _q_function(
            torch.sqrt(3.0 * k / (m - 1.0) * snr_linear)
        )
    elif mod in {"64QAM", "64-QAM", "QAM64"}:
        m = 64.0
        k = math.log2(m)
        ber = (4.0 / k) * (1.0 - 1.0 / math.sqrt(m)) * _q_function(
            torch.sqrt(3.0 * k / (m - 1.0) * snr_linear)
        )
    else:
        raise ValueError(f"Unsupported modulation scheme: {modulation}")

    return torch.clamp(ber, min=1e-8, max=0.5)


def bsc_flip(payload_bits: torch.Tensor, snr_db: torch.Tensor, modulation: str) -> torch.Tensor:
    """
    payload_bits: (B,N,Lmax) float {0,1}
    snr_db: (B,) or (B,1)
    flips each bit independently using BER derived from SNR and modulation.
    """
    ber = ber_from_snr(snr_db, modulation)
    if ber.ndim == 1:
        ber = ber[:, None, None]
    elif ber.ndim == 2:
        ber = ber[:, None]
    flip = (torch.rand_like(payload_bits) < ber).to(payload_bits.dtype)
    return (payload_bits + flip) % 2