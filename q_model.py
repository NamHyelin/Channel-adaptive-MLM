# q_model.py
import math
import torch


@torch.no_grad()
def popcount_u16_lut(device: torch.device) -> torch.Tensor:
    lut = torch.arange(0, 65536, dtype=torch.int32, device=device)  # lookup table
    x = lut.clone()
    cnt = torch.zeros_like(x)
    while True:
        nz = x != 0
        if not torch.any(nz):
            break
        xn = x[nz]
        x[nz] = xn & (xn - 1)
        cnt[nz] += 1
    return cnt  # 1의 개수


@torch.no_grad()
def q_full_vocab_from_ber(
    vocab_size: int,
    xhat_id: int,
    ber: float,
    device: torch.device,
    lut_u16: torch.Tensor,
) -> torch.Tensor:
    """
    q[v] ∝ ber^d * (1-ber)^(m-d), d=popcount(v xor xhat)
    using m = ceil(log2(vocab_size)).
    Returns q: [V] normalized.
    """
    if not (0.0 < ber < 1.0):
        raise ValueError("ber must be in (0,1)")
    m = int(math.ceil(math.log2(vocab_size)))  # mbits

    ids = torch.arange(vocab_size, dtype=torch.int32, device=device)
    xor = torch.bitwise_xor(ids, int(xhat_id)).to(torch.int32)

    # BERT vocab fits in 16 bits
    if xor.max().item() < 65536:
        d = lut_u16[xor].float()  # 해당 토큰과 다른 모든 토큰index 사이에 다른 1의 개수
    else:
        # fallback popcount for larger ids (unlikely)
        tmp = xor.clone()
        d = torch.zeros_like(tmp, dtype=torch.float32)
        for _ in range(32):
            d += (tmp & 1).float()
            tmp = tmp >> 1

    log_q = d * math.log(ber) + (m - d) * math.log(
        1.0 - ber
    )  # 각 bit이 독립적인 ber 을 가진다고 가정하는거.
    # Normalize
    log_q = log_q - torch.max(log_q)
    q = torch.exp(log_q)
    q = q / torch.sum(q)
    return q


@torch.no_grad()
def q_full_vocab_from_llr(
    llr: torch.Tensor,
    vocab_size: int,
    device: torch.device,
    temp: float = 1.0,
) -> torch.Tensor:
    """
    Compute q[v] for all vocab tokens from LLR values of individual bits.

    Args:
        llr: [m] tensor of LLR values for m bits forming one token
        vocab_size: total vocabulary size (must be <= 2^m)
        device: torch device

    Returns:
        q: [vocab_size] normalized probability distribution

    Bit order follows the same LSB-first convention as ids_to_bits:
        bit 0 = least significant bit, bit m-1 = most significant bit.

    For each possible token ID v in [0, vocab_size):
        - Convert v to an m-bit binary vector
        - For each bit i: P(bit_i=1) = sigmoid(llr[i])
        - q[v] = ∏_i P(bit_i = bit_value_in_v[i])
    Then normalize q to sum to 1.
    """
    llr = llr.to(device=device, dtype=torch.float32)
    llr = llr/temp
    m = llr.shape[0]

    # P(bit_i = 1) and P(bit_i = 0)
    p_1 = torch.sigmoid(llr)  # [m]
    p_0 = 1.0 - p_1  # [m]

    # Build all token bit patterns with the same LSB-first convention.
    ids = torch.arange(vocab_size, dtype=torch.int64, device=device)
    bit_positions = torch.arange(m, dtype=torch.int64, device=device)
    bits = ((ids.unsqueeze(1) >> bit_positions) & 1).to(torch.float32)  # [V, m]

    # q[v] = ∏_i P(bit_i = bit_value_in_v[i])
    log_q = torch.sum(
        bits * torch.log(p_1 + 1e-12) + (1.0 - bits) * torch.log(p_0 + 1e-12), dim=-1
    )

    # Normalize in log domain for stability
    log_q = log_q - torch.max(log_q)
    q = torch.exp(log_q)
    q = q / torch.sum(q)

    return q
