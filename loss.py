from typing import List

import torch
from torch import Tensor
from torch.nn import MSELoss


def calculate_content_loss(enc_out: Tensor, norm_out: Tensor, loss_obj):
    return loss_obj(enc_out, norm_out)


def calculate_style_loss(enc_out: List[Tensor], dec_out: List[Tensor], loss_obj) -> Tensor:
    assert len(enc_out) == len(dec_out)

    loss = torch.tensor([0.])
    for i in range(len(enc_out)):
        N, C, H, W = enc_out[i].size()
        g_t = dec_out[i].view(N, C, -1)
        s = enc_out[i].view(N, C, -1)

        # calc mean difference
        loss += loss_obj(g_t.mean(2).view(N, C, 1, 1), s.mean(2).view(N, C, 1, 1))

        # calc variance difference
        loss += loss_obj(g_t.var(2).view(N, C, 1, 1), s.var(2).view(N, C, 1, 1))

    return loss
