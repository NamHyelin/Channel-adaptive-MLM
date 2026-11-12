import torch
import torch.nn as nn


class TransformerEncoderLayerWithAttn(nn.TransformerEncoderLayer):
    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal=False, need_weights=False):
        attn_output, attn_weights = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            average_attn_weights=False,
            is_causal=is_causal,
        )
        return self.dropout1(attn_output), attn_weights

    def forward(
        self,
        src,
        src_mask=None,
        src_key_padding_mask=None,
        is_causal=False,
        need_weights=False,
    ):
        x = src
        if self.norm_first:
            x2, attn = self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal, need_weights)
            x = x + x2
            x = x + self._ff_block(self.norm2(x))
        else:
            x2, attn = self._sa_block(x, src_mask, src_key_padding_mask, is_causal, need_weights)
            x = self.norm1(x + x2)
            x = self.norm2(x + self._ff_block(x))
        if need_weights:
            return x, attn
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayerWithAttn(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=4*d_model,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            for _ in range(n_layers)
        ])

    def forward(self, x, src_key_padding_mask=None, return_attn=False):
        attn_list = []
        out = x
        for layer in self.layers:
            if return_attn:
                out, attn = layer(out, src_key_padding_mask=src_key_padding_mask, need_weights=True) #attn: (B,H,64,64)
                attn_list.append(attn)
            else:
                out = layer(out, src_key_padding_mask=src_key_padding_mask, need_weights=False)
        if return_attn:
            # (L,B,H,N,N)
            return out, torch.stack(attn_list, dim=0)
        return out


class CrossAttnBlock(nn.Module):
    """Cross-attention: Q from token states, K/V from memory tokens."""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4*d_model, d_model),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, q_tokens, memory, memory_key_padding_mask=None):
        q = self.ln1(q_tokens)
        out, _ = self.attn(q, memory, memory, key_padding_mask=memory_key_padding_mask, need_weights=False)
        x = q_tokens + out
        x = x + self.ff(self.ln2(x))
        return x


@torch.no_grad()
def attention_in_degree(attn_weights: torch.Tensor) -> torch.Tensor:
    """
    attn_weights: (B, H, N, N) where A[b,h,i,j] = attention from i -> j
    Return in-degree per token: sum_i A[i->j], averaged over heads.
    """
    # sum over source positions i
    indeg = attn_weights.sum(dim=2)  # (B,H,N) = sum_i A[i->j]
    indeg = indeg.mean(dim=1)        # (B,N)
    return indeg


def normalize_importance(indeg: torch.Tensor, alpha: float = 1.5, eps: float = 1e-8) -> torch.Tensor:
    """
    Make importance a stable distribution per sequence, then sharpen.
    indeg: (B,N), nonnegative
    """
    w = indeg.clamp_min(0.0)
    w = w / (w.sum(dim=-1, keepdim=True) + eps)   # sum to 1
    # sharpen to emphasize peaks, but keep normalized
    w = (w + eps) ** alpha
    w = w / (w.sum(dim=-1, keepdim=True) + eps)
    return w
