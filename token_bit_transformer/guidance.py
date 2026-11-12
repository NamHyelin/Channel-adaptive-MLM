# guidance.py
import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import (
    BertModel,
    BertEmbeddings,
    BertEncoder,
)


@torch.no_grad()
def q_stats(q: torch.Tensor):
    eps = 1e-12
    rmax = torch.max(q)
    ent = -(q * (q + eps).log()).sum()
    return rmax, ent


@torch.no_grad()
def build_key_bias_from_q(
    input_ids: torch.Tensor,  # [B,L]
    mask_positions: list[tuple[int, int]],
    q_list: list[torch.Tensor],
    ber: float,
    mode: str = "rmax",  # "rmax" or "entropy"
    beta: float = 1.0,
):
    """
    key_bias: [B,1,1,L], broadcasted over heads and query positions.
    Only mask positions get bias from q; others 0.
    """
    B, L = input_ids.shape
    device = input_ids.device
    kb = torch.zeros((B, 1, 1, L), device=device, dtype=torch.float32)
    for (b, pos), q in zip(mask_positions, q_list):
        rmax, ent = q_stats(q)
        val = torch.log(rmax + 1e-12) if mode == "rmax" else (-ent)
        val = val * (1.0 - ber)
        kb[b, 0, 0, pos] = beta * val
    return kb


@torch.no_grad()
def make_soft_inputs_embeds(
    bert_model,  # BertForMaskedLM or has .bert
    input_ids: torch.Tensor,  # [B,L]
    mask_positions: list[tuple[int, int]],
    q_list: list[torch.Tensor],  # each [V]
):
    """
    2-B: replace word embedding at mask positions with sum_v q(v)E[v]
    returns inputs_embeds: [B,L,D] (full embedding incl pos + token_type + LN + dropout)
    """
    bert = bert_model.bert
    device = input_ids.device
    embeddings: BertEmbeddings = bert.embeddings

    word_emb = embeddings.word_embeddings(input_ids)  # [B,L,D]
    E = embeddings.word_embeddings.weight  # [V,D] 보통 모델이고 위에서는 모델에 넣은거고, 이건 parameter. pretrained 된거고 training 하면 이거도 tuning 됨.

    for (b, pos), q in zip(mask_positions, q_list):
        e_soft = torch.matmul(q.to(device), E)  # [D] q 확률이랑 그 모든 vocab 의 embedding 곱한거
        word_emb[b, pos, :] = e_soft

    position_ids = (
        torch.arange(input_ids.size(1), device=device).unsqueeze(0).expand_as(input_ids)
    )
    token_type_ids = torch.zeros_like(input_ids)

    x = (
        word_emb
        + embeddings.position_embeddings(position_ids)
        + embeddings.token_type_embeddings(token_type_ids)
    ) # 토큰 순서 임베딩, 문장 구분 임베딩 원래 더함
    x = embeddings.LayerNorm(x)
    x = embeddings.dropout(x)
    return x


class BertModelWithKeyBias(nn.Module):
    """
    2-A without forking transformers:
    - wraps an existing BertModel (from BertForMaskedLM.bert)
    - manually runs encoder layers so we can pass key_bias into self-attention
    """

    def __init__(self, bert: BertModel):
        super().__init__()
        self.embeddings = bert.embeddings
        self.encoder: BertEncoder = bert.encoder
        self.pooler = bert.pooler
        self.config = bert.config

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        key_bias=None,  # [B,1,1,L]
        output_hidden_states=False,
        return_dict=True,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Use either input_ids or inputs_embeds, not both.")

        if attention_mask is None:
            if input_ids is not None:
                attention_mask = torch.ones_like(input_ids)
            else:
                attention_mask = torch.ones(
                    inputs_embeds.size()[:-1], device=inputs_embeds.device
                )

        # HF BERT uses additive mask: 0 for keep, -inf for masked.
        extended_attention_mask = attention_mask[:, None, None, :].to(
            dtype=torch.float32
        )
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        if key_bias is not None:
            extended_attention_mask = extended_attention_mask + key_bias

        if inputs_embeds is None:
            embedding_output = self.embeddings(input_ids=input_ids)
        else:
            embedding_output = inputs_embeds

        hidden_states = embedding_output
        all_hs = [] if output_hidden_states else None

        # run each layer (bias already folded into extended_attention_mask)
        for layer_module in self.encoder.layer:
            if output_hidden_states:
                all_hs.append(hidden_states)

            hidden_states = layer_module(hidden_states, extended_attention_mask)[0]

        if output_hidden_states:
            all_hs.append(hidden_states)

        if not return_dict:
            out = (hidden_states,)
            if output_hidden_states:
                out = out + (tuple(all_hs),)
            return out

        return {
            "last_hidden_state": hidden_states,
            "hidden_states": tuple(all_hs) if output_hidden_states else None,
        }
