from __future__ import annotations

# corrupt.py (Option B: AWGN + soft demod LLR + LLR-based masking)
import math
import random
import torch


# -------------------------
# Utility: Q-function, BER (theory reference)
# -------------------------
def _qfunc(x: float) -> float:
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def ber_from_snr(
    snr_db: float | None,
    modulation: str,
    *,
    snr_db_min: float = -5.0,
    snr_db_max: float = 10.0,
    rng: random.Random | None = None,
) -> tuple[float, float]:
    """
    Theoretical uncoded BER over AWGN (approx).
    Interprets snr_db as Eb/N0 in dB (common in comm papers).
    Returns (ber, snr_db_used).
    """
    if snr_db is None:
        rng = rng or random
        snr_db = rng.uniform(snr_db_min, snr_db_max)

    mod = modulation.strip().lower()
    snr_lin = 10 ** (snr_db / 10.0)  # Eb/N0

    if mod in {"bpsk", "qpsk"}:
        # BER = Q(sqrt(2 Eb/N0)) = 0.5 erfc(sqrt(Eb/N0))
        ber = 0.5 * math.erfc(math.sqrt(snr_lin))
    elif mod in {"16qam"}:
        m = 16
        k = math.log2(m)
        ber = (
            (4.0 / k)
            * (1.0 - 1.0 / math.sqrt(m))
            * _qfunc(math.sqrt(3.0 * k / (m - 1.0) * snr_lin))
        )
    elif mod in {"64qam"}:
        m = 64
        k = math.log2(m)
        ber = (
            (4.0 / k)
            * (1.0 - 1.0 / math.sqrt(m))
            * _qfunc(math.sqrt(3.0 * k / (m - 1.0) * snr_lin))
        )
    else:
        raise ValueError("modulation must be one of bpsk | qpsk | 16qam | 64qam")

    ber = float(min(max(ber, 0.0), 0.5))
    return ber, float(snr_db)


# -------------------------
# Bit packing / unpacking for token ids
# (LSB-first convention to match (id >> bitpos) & 1)
# -------------------------
@torch.no_grad()
def ids_to_bits(ids: torch.Tensor, m_bits: int) -> torch.Tensor:
    """
    ids: [B,L] int64
    returns bits: [B,L,m_bits] float32 in {0,1}, LSB first
    """
    device = ids.device
    bit_positions = torch.arange(m_bits, device=device, dtype=torch.int64)  # 0..m-1
    bits = ((ids.unsqueeze(-1) >> bit_positions) & 1).to(torch.float32)
    return bits


@torch.no_grad()
def bits_to_ids(bits: torch.Tensor) -> torch.Tensor:
    """
    bits: [B,L,m_bits] float/bool in {0,1}, LSB first
    returns ids: [B,L] int64
    """
    device = bits.device
    m_bits = bits.shape[-1]
    bitvals = (1 << torch.arange(m_bits, device=device, dtype=torch.int64)).view(
        1, 1, -1
    )
    ids = torch.sum(bits.to(torch.int64) * bitvals, dim=-1)
    return ids


# -------------------------
# Constellations (Gray-coded) and max-log LLR demod
# -------------------------
def _gray(n: int) -> int:
    return n ^ (n >> 1)


def _pam_levels_gray(m_side: int) -> tuple[list[float], list[list[int]]]:
    """
    1D PAM levels with Gray labeling.
    m_side: 2,4,8 (for QPSK->2, 16QAM->4, 64QAM->8)
    returns:
      levels: length m_side (unnormalized)
      bits:   list of bit lists per level (MSB..LSB for that axis)
    """
    # levels: e.g. m_side=4 -> [-3,-1,+1,+3]
    levels = [-(m_side - 1) + 2 * i for i in range(m_side)]
    nbits = int(math.log2(m_side))
    labels = []
    for i in range(m_side):
        g = _gray(i)
        b = [(g >> (nbits - 1 - j)) & 1 for j in range(nbits)]
        labels.append(b)
    return [float(x) for x in levels], labels


def constellation_gray(
    modulation: str, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      const: [M] complex64 constellation points (unit average Es=1)
      bits:  [M,k] int64 bit labels per symbol (bit order: [I_bits..., Q_bits...], MSB..LSB per axis)
    """
    mod = modulation.strip().lower()
    if mod == "bpsk":
        # Map bit 0->-1, bit 1->+1 (real axis)
        const = torch.tensor([-1.0, +1.0], device=device, dtype=torch.float32)
        bits = torch.tensor([[0], [1]], device=device, dtype=torch.int64)
        const_c = const.to(torch.complex64)
        # Normalize Es=1 already
        return const_c, bits

    if mod == "qpsk":
        # 2x2 (4-QAM), Gray on each axis: I bit, Q bit
        m_side = 2
        levels, lab = _pam_levels_gray(m_side)  # levels [-1,+1], labels [[0],[1]]
        pts = []
        bls = []
        for i_idx, i_val in enumerate(levels):
            for q_idx, q_val in enumerate(levels):
                pts.append(complex(i_val, q_val))
                bls.append(lab[i_idx] + lab[q_idx])  # [Ibit] + [Qbit]
        const = torch.tensor(pts, device=device, dtype=torch.complex64)
        bits = torch.tensor(bls, device=device, dtype=torch.int64)
        # Normalize to unit average energy
        const = const / torch.sqrt(torch.mean(const.real**2 + const.imag**2))
        return const, bits

    if mod in {"16qam", "64qam"}:
        M = 16 if mod == "16qam" else 64
        m_side = int(math.sqrt(M))  # 4 or 8
        levels, lab = _pam_levels_gray(m_side)
        pts = []
        bls = []
        for i_idx, i_val in enumerate(levels):
            for q_idx, q_val in enumerate(levels):
                pts.append(complex(i_val, q_val))
                bls.append(lab[i_idx] + lab[q_idx])  # [I bits..., Q bits...]
        const = torch.tensor(pts, device=device, dtype=torch.complex64)
        bits = torch.tensor(bls, device=device, dtype=torch.int64)
        const = const / torch.sqrt(torch.mean(const.real**2 + const.imag**2))  # Es=1
        return const, bits

    raise ValueError("modulation must be one of bpsk | qpsk | 16qam | 64qam")


def bits_per_symbol(modulation: str) -> int:
    mod = modulation.strip().lower()
    if mod == "bpsk":
        return 1
    if mod == "qpsk":
        return 2
    if mod == "16qam":
        return 4
    if mod == "64qam":
        return 6
    raise ValueError("modulation must be one of bpsk | qpsk | 16qam | 64qam")


@torch.no_grad()
def modulate(bits: torch.Tensor, modulation: str) -> torch.Tensor:
    """
    bits: [B, Nsym, k] int/float in {0,1}
    returns x: [B, Nsym] complex64
    """
    device = bits.device
    const, labels = constellation_gray(modulation, device=device)  # [M], [M,k]
    # Find symbol index by matching label (small M so brute is fine)
    # bits are 0/1, labels are 0/1
    b = bits.to(torch.int64)  # [B,N,k]
    # Compute Hamming distance to each label and argmin
    # dist: [B,N,M]
    dist = (b.unsqueeze(-2) != labels.view(1, 1, labels.shape[0], labels.shape[1])).sum(
        dim=-1
    )
    idx = dist.argmin(dim=-1)  # [B,N]
    x = const[idx]  # [B,N]
    return x


@torch.no_grad()
def awgn(x: torch.Tensor, snr_db: float, k_bits: int) -> tuple[torch.Tensor, float]:
    """
    x: [B,N] complex symbols with Es=1
    snr_db is interpreted as Eb/N0 (dB).
    k_bits = bits per symbol.
    returns y, sigma2 (noise variance per real dimension)
    """
    snr_lin = 10 ** (snr_db / 10.0)  # Eb/N0
    # Es = 1. Eb = Es/k = 1/k
    # N0 = Eb / (Eb/N0) = (1/k) / snr_lin
    # sigma^2 per real dim = N0/2 = 1 / (2*k*snr_lin)
    sigma2 = 1.0 / (2.0 * k_bits * snr_lin)

    n_re = torch.randn_like(x.real) * math.sqrt(sigma2)
    if torch.is_complex(x):
        n_im = torch.randn_like(x.imag) * math.sqrt(sigma2)
        y = (x.real + n_re).to(torch.float32) + 1j * (x.imag + n_im).to(torch.float32)
        y = y.to(torch.complex64)
    else:
        y = (x + n_re).to(torch.float32)
    return y, float(sigma2)


@torch.no_grad()
def demod_llr_maxlog(y: torch.Tensor, modulation: str, sigma2: float) -> torch.Tensor:
    """
    Max-log LLR for Gray-coded constellations.
    y: [B,N] complex64
    returns llr: [B,N,k] float32  (LLR = log P(b=1|y)/P(b=0|y))
    """
    device = y.device
    const, labels = constellation_gray(modulation, device=device)  # const = complex [M]
    M, k = labels.shape  # M= constellation points 수, k=bits per symbol

    # distance^2: [B,N,M]
    # |y - s|^2 = (re)^2 + (im)^2
    yr = y.real.unsqueeze(-1)  # [B,N,1]
    yi = y.imag.unsqueeze(-1)
    sr = const.real.view(1, 1, M)
    si = const.imag.view(1, 1, M)
    d2 = (yr - sr) ** 2 + (yi - si) ** 2

    llrs = []
    for j in range(k):
        mask1 = (labels[:, j] == 1).view(1, 1, M)
        mask0 = ~mask1
        # min distance among symbols with bit=0/1
        d1 = d2.masked_fill(~mask1, float("inf")).min(dim=-1).values  # [B,N]
        d0 = d2.masked_fill(~mask0, float("inf")).min(dim=-1).values  # [B,N]
        # Max-log: LLR ≈ (min_{b=0} d^2 - min_{b=1} d^2) / sigma2
        llr_j = (d0 - d1) / float(sigma2)
        llrs.append(llr_j)

    llr = torch.stack(llrs, dim=-1).to(torch.float32)  # [B,N,k]
    return llr


# -------------------------
# End-to-end: ids -> channel -> (xbar_ids, llr_bits_per_token)
# -------------------------
@torch.no_grad()
def transmit_ids_awgn_llr(
    ids: torch.Tensor,
    snr_db: float,
    modulation: str,
    m_bits: int,
    tokenizer=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    ids: [B,L] int64 (token ids). L은 최대 시퀀스 길이로, 짧은건 0 패딩
    m_bits 개수만큼 2진법으로 표현해서 modulation 심볼로 매핑 -> AWGN 채널 통과 -> 각 토큰별 bit LLR 계산 -> hard-decision으로 xbar_ids 복원 (LLR sign) 및 llr_tok 반환

    Returns:
      xbar_ids: [B,L] int64 (hard-decoded from LLR sign)
      llr_tok:  [B,L,m_bits] float32 (bit LLRs aligned to token bits)
    """
    device = ids.device
    B, L = ids.shape
    k = bits_per_symbol(modulation)
    special_ids = list(
        dict.fromkeys(
            [
                *(
                    getattr(tokenizer, "all_special_ids", [])
                    if tokenizer is not None
                    else []
                ),
                1012,
            ]
        )
    )

    special_mask = torch.zeros((B, L), device=device, dtype=torch.bool)
    for sid in special_ids:
        special_mask |= ids == sid

    # Token bits: [B,L,m_bits]
    b_tok = ids_to_bits(ids, m_bits=m_bits)  # float {0,1}
    # Flatten to stream: [B, L*m_bits]
    b_stream = b_tok.reshape(B, L * m_bits)

    # Pad to multiple of k
    n_bits = b_stream.shape[1]
    pad = (k - (n_bits % k)) % k  # k의 배수가 되도록 패딩 (0으로 채움)
    if pad > 0:
        b_stream = torch.cat([b_stream, torch.zeros((B, pad), device=device)], dim=1)

    # Group into symbols: b_sym= [B, Nsym, k]
    Nsym = b_stream.shape[1] // k
    b_sym = b_stream.reshape(B, Nsym, k).to(torch.int64)

    # Modulate + AWGN
    x = modulate(b_sym, modulation=modulation)  # [B,Nsym] complex
    y, sigma2 = awgn(x, snr_db=snr_db, k_bits=k)  # [B,Nsym] complex, scalar sigma2

    # Soft demod LLR: [B,Nsym,k]
    llr_sym = demod_llr_maxlog(y, modulation=modulation, sigma2=sigma2)

    # Unpack back to stream: [B, Nsym*k] then remove pad
    llr_stream = llr_sym.reshape(B, Nsym * k)
    if pad > 0:
        llr_stream = llr_stream[:, :n_bits]

    # Align to tokens: [B,L,m_bits]
    llr_tok = llr_stream.reshape(B, L, m_bits)

    if special_mask.any():
        llr_tok = llr_tok.clone()
        llr_tok[special_mask] = (b_tok[special_mask] * 2.0 - 1.0) * 1e9

    # Hard bits from LLR sign
    b_hat_tok = (llr_tok > 0).to(torch.int64).to(torch.float32)  # float {0,1}
    xbar_ids = bits_to_ids(b_hat_tok).to(torch.int64)
    if special_mask.any():
        xbar_ids = xbar_ids.clone()
        xbar_ids[special_mask] = ids[special_mask]
    return xbar_ids, llr_tok


# -------------------------
# Masking based on LLR-derived per-token reliability
# -------------------------
@torch.no_grad()
def make_masked_input(
    xbar_ids: torch.Tensor,
    llr_tok: torch.Tensor,
    tokenizer,
    strategy: str,
    mask_rate: float,
    rel_tau: float,
    *,
    rel_mode: str,  # "minabs" | "meanabs"
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build xhat_ids by masking uncertain tokens based on LLR.

    Inputs:
      xbar_ids: [B,L] received (hard) token ids
      llr_tok:  [B,L,m_bits] bit-level LLRs aligned to each token
      rel_mode:
        - "minabs": rel_i = min_k |LLR_{i,k}|  (very strict)
        - "meanabs": rel_i = mean_k |LLR_{i,k}| (smoother)

    Returns:
      xhat_ids: [B,L] with [MASK] where uncertain
      rel:      [B,L] token reliability score (>=0)
    """
    device = xbar_ids.device
    B, L = xbar_ids.shape

    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is None:
        raise ValueError("tokenizer must have vocab_size")

    if llr_tok.shape[:2] != (B, L):
        raise ValueError(
            f"llr_tok shape must start with [B,L]={B, L}, got {tuple(llr_tok.shape)}"
        )

    invalid_mask = (xbar_ids < 0) | (xbar_ids >= vocab_size)

    abs_llr = llr_tok.abs()
    if rel_mode == "minabs":
        rel = abs_llr.min(dim=-1).values
    elif rel_mode == "meanabs":
        rel = abs_llr.mean(dim=-1)
    else:
        raise ValueError("rel_mode must be 'minabs' or 'meanabs'")

    # invalid ids forced unreliable
    rel = rel.masked_fill(invalid_mask, 0.0)

    # decide mask
    if strategy == "random":
        mask = torch.rand((B, L), device=device) < mask_rate
    elif strategy == "reliability":
        mask = rel < rel_tau
    else:
        raise ValueError("mask_strategy must be 'random' or 'reliability'")

    # always mask invalid
    mask |= invalid_mask

    # don't mask special tokens unless invalid
    special = torch.zeros_like(mask)
    for sid in tokenizer.all_special_ids + [
        1012
    ]:  # [100, 102, 0, 101, 103],  [CLS], [SEP], [PAD], [MASK], 그리고 1012는 점.
        special |= xbar_ids == sid
    mask = (mask & ~special) | invalid_mask

    xhat_ids = xbar_ids.clone()
    xhat_ids[mask] = tokenizer.mask_token_id
    return xhat_ids, rel
