import torch
from torch import nn


class AdaIN(nn.Module):

    def __init__(self, eps=1e-5):
        super(AdaIN, self).__init__()
        self.eps = eps

    def forward(self, content, style):
        assert len(content.size()) == len(style.size()) == 4    # make sure its NCHW format
        assert content.size() == style.size()                   # make sure the shapes match

        content_mean, content_std = self.get_mean_and_std(content)
        style_mean, style_std = self.get_mean_and_std(style)

        # Formula (8) - Adaptive Instance Normalization
        out = (content - content_mean) / content_std
        out = out * style_std + style_mean
        return out

    def get_mean_and_std(self, x):
        N, C, H, W = x.size()
        instance = x.view(N, C, -1)
        mean = instance.mean(2).view(N, C, 1, 1)
        std = torch.sqrt(instance.var(2) + self.eps).view(N, C, 1, 1)
        return mean, std


class EncoderVGG(nn.Module):

    def __init__(self):
        super(EncoderVGG, self).__init__()

    def forward(self, x):
        pass


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

    def forward(self, x):
        pass


if __name__ == '__main__':
    ada = AdaIN(3)
    c = torch.randn((2, 3, 4, 4))
    s = torch.randn((2, 3, 4, 4))
    ada(c, s)
