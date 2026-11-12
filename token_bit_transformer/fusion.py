# fusion.py
import torch
import torch.nn as nn


class AdaptiveFusion(nn.Module):
    """
    (1-A) full-vocab fusion:
      fused_logits = logits + gate_i * phi(log_q, ber, rmax, ent)
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_size + 3, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.temp = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        logits_i: torch.Tensor,
        hidden_i: torch.Tensor,
        q_i: torch.Tensor,
        ber: float,
    ):
        eps = 1e-12
        log_q = (q_i + eps).log()
        rmax = torch.max(q_i)
        ent = -(q_i * log_q).sum() #entropy. 낮으면 q 분포가 퍼진거고 채널이 불확실하다는것
        ber_t = torch.tensor([ber], device=logits_i.device, dtype=logits_i.dtype)

        gate_in = torch.cat([hidden_i, rmax.view(1), ent.view(1), ber_t], dim=0)
        g = self.gate(gate_in)  # [1] 채널이 얼마나 불확실한지 반영하는 gate. 1에 가까울수록 채널 q를 더 믿음

        phi = self.scale * (log_q / (self.temp.abs() + 1e-6)) # q를 더하는데 log_q. 이건 model output logit 이 softmax 들어가기 전의 값이라서, 결국 LLM이 출력하는 분포와 채널 posterior q 를 곱해서 새로운 분포를 만들기 위해서는 log 취해서 더해야됨.
        fused = logits_i + g * phi
        return fused, g, self.scale, self.temp
