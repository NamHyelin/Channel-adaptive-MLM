from __future__ import annotations

import math
import torch

from channel_corrupt import (
    ber_from_snr,
    bits_per_symbol,
    bits_to_ids,
    demod_llr_maxlog,
    ids_to_bits,
    make_masked_input,
    modulate,
    transmit_ids_awgn_llr,
)


def _parse_power_ratios(
    power_ratios: list[float] | torch.Tensor,
    num_users: int,
    device: torch.device,
) -> torch.Tensor:
    p = torch.as_tensor(power_ratios, dtype=torch.float32, device=device).flatten()
    if p.numel() == 0:
        raise ValueError("power_ratios must not be empty")
    if p.numel() < num_users:
        p = torch.cat([p, p.new_ones(num_users - p.numel())], dim=0)
    elif p.numel() > num_users:
        p = p[:num_users]
    if torch.any(p < 0):
        raise ValueError("power_ratios must be >= 0")
    if float(p.sum().item()) <= 0.0:
        raise ValueError("sum(power_ratios) must be > 0")
    return p


def _sample_user_channels(
    num_users: int,
    device: torch.device,
    mode: str,
) -> torch.Tensor:
    if mode == "ones":
        return torch.ones((num_users,), device=device, dtype=torch.complex64)
    if mode == "rayleigh":
        h_re = torch.randn(
            (num_users,), device=device, dtype=torch.float32
        ) / math.sqrt(2.0)
        h_im = torch.randn(
            (num_users,), device=device, dtype=torch.float32
        ) / math.sqrt(2.0)
        return (h_re + 1j * h_im).to(torch.complex64)
    raise ValueError("multi_channel_gain_mode must be 'rayleigh' or 'ones'")


def _sigma2_from_snr_db(snr_db: float, k_bits: int) -> float:
    snr_lin = 10 ** (snr_db / 10.0)
    return float(1.0 / (2.0 * k_bits * snr_lin))


@torch.no_grad()
def build_multiuser_ids(
    desired_ids: torch.Tensor,
    num_users: int,
    mode: str = "roll",
) -> torch.Tensor:
    """
    desired_ids: [B,L] of the target user.
    returns ids_users: [U,B,L] where user 0 is target user.
    """
    if num_users < 1:
        raise ValueError("num_users must be >= 1")

    B, L = desired_ids.shape
    ids_users = [desired_ids]

    if num_users == 1:
        return torch.stack(ids_users, dim=0)

    if mode == "roll":
        for u in range(1, num_users):
            shift = u % max(B, 1)
            ids_users.append(torch.roll(desired_ids, shifts=shift, dims=0))
    elif mode == "random_permute":
        gen = torch.Generator(device=desired_ids.device)
        gen.manual_seed(1234)
        for _ in range(1, num_users):
            perm = torch.randperm(B, generator=gen, device=desired_ids.device)
            ids_users.append(desired_ids[perm])
    else:
        raise ValueError("mode must be one of: roll | random_permute")

    return torch.stack(ids_users, dim=0)  # [U,B,L]


@torch.no_grad()
def transmit_ids_awgn_llr_multiuser(
    ids_users: torch.Tensor,
    snr_db: float,
    modulation: str,
    m_bits: int,
    *,
    target_user: int = 0,
    pilot_tokens: int = 0,
    power_ratios: list[float] | torch.Tensor | None = None,
    channel_gain_mode: str = "rayleigh",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward-compatible single-target wrapper."""
    U = ids_users.shape[0]
    if not (0 <= target_user < U):
        raise ValueError("target_user out of range")
    if power_ratios is None:
        power_ratios = [1.0] * U

    xbar_all, llr_all, _sinr_db_all = transmit_ids_awgn_llr_multiuser_allusers(
        ids_users,
        snr_db,
        modulation,
        m_bits,
        power_ratios=power_ratios,
        channel_gain_mode=channel_gain_mode,
        pilot_tokens=pilot_tokens,
    )
    return xbar_all[target_user], llr_all[target_user]


@torch.no_grad()
def transmit_ids_awgn_llr_multiuser_allusers(
    ids_users: torch.Tensor,
    snr_db: float,
    modulation: str,
    m_bits: int,
    *,
    power_ratios: list[float] | torch.Tensor,
    channel_gain_mode: str = "rayleigh",
    pilot_tokens: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    ids_users: [U,B,L]
    Returns:
      xbar_all: [U,B,L]
      llr_all:  [U,B,L,m_bits]
      sinr_db_all: [U]
    """
    if ids_users.dim() != 3:
        raise ValueError("ids_users must have shape [U,B,L]")

    U, B, L = ids_users.shape
    if pilot_tokens < 0 or pilot_tokens > L:
        raise ValueError("pilot_tokens must be in [0, L]")

    device = ids_users.device
    k = bits_per_symbol(modulation)

    p = _parse_power_ratios(power_ratios, U, device)
    h = _sample_user_channels(U, device, channel_gain_mode)
    a = torch.sqrt(p).to(torch.complex64) * h  # [U]

    b_tok = ids_to_bits(ids_users.reshape(U * B, L), m_bits=m_bits).reshape(
        U, B, L, m_bits
    )
    b_stream = b_tok.reshape(U, B, L * m_bits)

    n_bits = b_stream.shape[-1]
    pad = (k - (n_bits % k)) % k
    if pad > 0:
        zeros = torch.zeros((U, B, pad), device=device, dtype=b_stream.dtype)
        b_stream = torch.cat([b_stream, zeros], dim=-1)

    Nsym = b_stream.shape[-1] // k
    b_sym = b_stream.reshape(U * B, Nsym, k).to(torch.int64)
    x_users = modulate(b_sym, modulation=modulation).reshape(U, B, Nsym)

    x_sum = torch.sum(a.view(U, 1, 1) * x_users, dim=0)

    sigma2_noise = _sigma2_from_snr_db(snr_db, k)
    n_re = torch.randn_like(x_sum.real) * math.sqrt(sigma2_noise)
    n_im = torch.randn_like(x_sum.imag) * math.sqrt(sigma2_noise)
    y = (x_sum.real + n_re).to(torch.float32) + 1j * (x_sum.imag + n_im).to(
        torch.float32
    )
    y = y.to(torch.complex64)

    abs_a2 = (a.real * a.real + a.imag * a.imag).to(torch.float32)
    total_signal_power = abs_a2.sum()
    noise_power = torch.tensor(2.0 * sigma2_noise, device=device, dtype=torch.float32)

    xbar_all = torch.empty((U, B, L), device=device, dtype=torch.int64)
    llr_all = torch.empty((U, B, L, m_bits), device=device, dtype=torch.float32)
    sinr_db_all = torch.empty((U,), device=device, dtype=torch.float32)

    for u in range(U):
        desired_power = abs_a2[u]
        interference_power = total_signal_power - desired_power

        sinr_lin_u = desired_power / (noise_power + interference_power + 1e-12)
        sinr_db_all[u] = 10.0 * torch.log10(sinr_lin_u + 1e-12)

        y_eq = y / (a[u] + 1e-12)
        sigma2_eff_real = (sigma2_noise + float(interference_power.item()) / 2.0) / (
            float(desired_power.item()) + 1e-12
        )

        llr_sym_u = demod_llr_maxlog(
            y_eq, modulation=modulation, sigma2=sigma2_eff_real
        )
        llr_stream_u = llr_sym_u.reshape(B, Nsym * k)
        if pad > 0:
            llr_stream_u = llr_stream_u[:, :n_bits]

        llr_tok_u = llr_stream_u.reshape(B, L, m_bits)
        b_hat_tok_u = (llr_tok_u > 0).to(torch.float32)
        xbar_u = bits_to_ids(b_hat_tok_u).to(torch.int64)

        if pilot_tokens > 0:
            data_len = L - pilot_tokens
            xbar_u[:, data_len:] = ids_users[u, :, data_len:]
            llr_tok_u[:, data_len:, :] = 1e6

        xbar_all[u] = xbar_u
        llr_all[u] = llr_tok_u

    return xbar_all, llr_all, sinr_db_all


__all__ = [
    "ber_from_snr",
    "make_masked_input",
    "transmit_ids_awgn_llr",
    "build_multiuser_ids",
    "transmit_ids_awgn_llr_multiuser",
    "transmit_ids_awgn_llr_multiuser_allusers",
]
