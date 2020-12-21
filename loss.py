from typing import List

import torch
from torch import Tensor

from typing import Callable, Tuple


def calculate_content_loss(enc_out: Tensor, norm_out: Tensor, loss_obj: Callable):
    return loss_obj(enc_out, norm_out)


def get_instance_statistics(x: Tensor, eps: float = 1e-5) -> Tuple[Tensor, Tensor]:
    N, C, H, W = x.size()
    instance = x.view(N, C, -1)  # flatten HW
    mean = instance.mean(2).view(N, C, 1, 1)
    std = torch.sqrt(instance.var(2) + eps).view(N, C, 1, 1)
    return mean, std


def calculate_style_loss(enc_out: List[Tensor], dec_out: List[Tensor], loss_fn: Callable) -> Tensor:
    # print(len(enc_out), len(dec_out))
    assert len(enc_out) == len(dec_out)
    # assert all([x == y for x, y in zip(enc_out, dec_out)])

    loss = torch.tensor([0.], device=enc_out[0].device)
    for i in range(len(enc_out)):
        N, C, H, W = enc_out[i].size()
        g_t = dec_out[i]
        s = enc_out[i]

        g_mean, g_std = get_instance_statistics(g_t)
        s_mean, s_std = get_instance_statistics(s)

        loss += loss_fn(g_mean, s_mean) + loss_fn(g_std, s_std)

    return loss
