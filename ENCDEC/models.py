import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import TransformerEncoder, CrossAttnBlock

class SemanticBitEncoder(nn.Module):
    """
    Input: token ids (B,N), channel state (SNR, BER etc.)
    Output:
      - lengths (B,N) in [l_min, l_max]
      - payload_bits (B,N,l_max) in {0,1} (ST)
      - importance weights w (B,N) detached (computed from attention elsewhere)
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(4096, cfg.d_model)

        self.ctx = TransformerEncoder(cfg.d_model, cfg.n_heads, cfg.n_layers_enc, cfg.dropout)

        # (snr, ber) 2개 실수값을 d_model 차원벡터로 바꿈
        self.state_mlp = nn.Sequential(
            nn.Linear(2, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.state_tokens = nn.Parameter(torch.randn(cfg.n_state_tokens, cfg.d_model))
        self.imp_tokens = nn.Parameter(torch.randn(cfg.n_imp_tokens, cfg.d_model))

        self.cross = CrossAttnBlock(cfg.d_model, cfg.n_heads, cfg.dropout)

        # Heads
        self.len_head = nn.Linear(cfg.d_model, cfg.l_max - cfg.l_min + 1)  # categorical over lengths
        self.bit_head = nn.Linear(cfg.d_model, cfg.l_max)                  # payload logits for max bits

    def forward(self, tokens: torch.Tensor, snr: torch.Tensor, ber: torch.Tensor):
        B, N = tokens.shape
        pos = torch.arange(N, device=tokens.device)[None, :].expand(B, N)
        h = self.tok(tokens) + self.pos(pos)
        h, attn_weights = self.ctx(h, return_attn=True)  # h: (B,N,D), attn: (L,B,H,N,N)

        # build conditioning memory M = [state_tokens conditioned] + [importance summary tokens placeholder]
        # 채널관련은 state_vec과 state_tokens를 더한거. state_tokens는 학습가능한 파라미터. 학습되게 하려고.
        state_vec = self.state_mlp(torch.stack([snr, ber], dim=-1))  # (B,D)
        m_state = self.state_tokens[None, :, :].expand(B, -1, -1) + state_vec[:, None, :]  # (B,S,D)
        # importance tokens are trainable slots; you will fill them via cross-attn later if desired
        m_imp = self.imp_tokens[None, :, :].expand(B, -1, -1)  # (B,I,D)
        memory = torch.cat([m_state, m_imp], dim=1)  # (B,S+I,D)

        h2 = self.cross(h, memory)

        # length distribution
        len_logits = self.len_head(h2)  # (B,N,Lchoices)
        len_probs = F.softmax(len_logits, dim=-1)
        # expected length (for differentiable rate)
        lengths_expect = (len_probs * torch.arange(self.cfg.l_min, self.cfg.l_max + 1, device=tokens.device)).sum(dim=-1)

        # sample hard length with straight-through (Gumbel softmax)
        g = F.gumbel_softmax(len_logits, tau=1.0, hard=True, dim=-1)  # (B,N,Lchoices)
        lengths_hard = (g * torch.arange(self.cfg.l_min, self.cfg.l_max + 1, device=tokens.device)).sum(dim=-1).long()

        # payload bits
        bit_logits = self.bit_head(h2)           # (B,N,l_max)
        bit_probs = torch.sigmoid(bit_logits)
        bits_hard = (bit_probs > 0.5).float()
        bits_st = bits_hard + (bit_probs - bit_probs.detach())  # ST

        return {
            "lengths_hard": lengths_hard,           # (B,N)
            "lengths_expect": lengths_expect,       # (B,N) float
            "payload_bits": bits_st,                # (B,N,l_max) float {0,1} with ST grad
            "len_probs": len_probs,                 # for analysis
            "bit_probs": bit_probs,                 # for analysis
            "ctx_states": h,
            "attn_weights": attn_weights,
        }



class BitToTokenDecoder(nn.Module):
    """
    Uses header-provided per-token segments.
    Input (batched, padded): per-token payload (B,N,Lmax) + lengths + (snr,ber)
    Output: token logits (B,N,V)
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.bit_emb = nn.Embedding(3, cfg.d_model)  # 0/1 plus optional mask=2 if you ever use it
        self.pos = nn.Embedding(4096, cfg.d_model)

        self.ctx = TransformerEncoder(cfg.d_model, cfg.n_heads, cfg.n_layers_dec, cfg.dropout)

        self.state_mlp = nn.Sequential(
            nn.Linear(2, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.state_tokens = nn.Parameter(torch.randn(cfg.n_state_tokens, cfg.d_model))
        self.cross = CrossAttnBlock(cfg.d_model, cfg.n_heads, cfg.dropout)

        self.out = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(self, payload_bits: torch.Tensor, lengths: torch.Tensor, snr: torch.Tensor, ber: torch.Tensor):
        """
        payload_bits: (B,N,Lmax) float {0,1}
        lengths:      (B,N) int
        """
        B, N, Lmax = payload_bits.shape

        # build a per-token fixed-length representation by embedding bits and pooling with a length mask
        bits_int = payload_bits.round().clamp(0, 1).long()  # (B,N,Lmax)
        # add a "token-local position" encoding
        pos = torch.arange(Lmax, device=payload_bits.device)[None, None, :].expand(B, N, Lmax)
        h = self.bit_emb(bits_int) + self.pos(pos)  # (B,N,Lmax,D)

        # length mask for pooling
        mask = (pos < lengths[:, :, None]).float()  # (B,N,Lmax)
        h = h * mask[..., None]
        # simple mean pool (can replace with small local transformer if needed)
        pooled = h.sum(dim=2) / (mask.sum(dim=2, keepdim=True) + 1e-8)  # (B,N,D)

        # sequence modeling over tokens
        z = self.ctx(pooled)

        # channel conditioning via cross-attn
        state_vec = self.state_mlp(torch.stack([snr, ber], dim=-1))  # (B,D)
        m_state = self.state_tokens[None, :, :].expand(B, -1, -1) + state_vec[:, None, :]
        z2 = self.cross(z, m_state)

        logits = self.out(z2)  # (B,N,V)
        return logits