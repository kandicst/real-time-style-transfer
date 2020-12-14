import torch
from torch import nn
from torch import Tensor
from torchvision import models

from typing import Tuple, List, Optional, Union


class AdaIN(nn.Module):

    def __init__(self, eps: float = 1e-5):
        super(AdaIN, self).__init__()
        self.eps = eps

    def forward(self, content: Tensor, style: Tensor) -> Tensor:
        assert len(content.size()) == len(style.size()) == 4  # make sure its NCHW format
        assert content.size() == style.size()  # make sure the shapes match

        content_mean, content_std = self.get_instance_statistics(content)
        style_mean, style_std = self.get_instance_statistics(style)

        # Equation (8)
        out = (content - content_mean) / content_std
        out = out * style_std + style_mean
        return out

    def get_instance_statistics(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        N, C, H, W = x.size()
        instance = x.view(N, C, -1)  # flatten HW
        mean = instance.mean(2).view(N, C, 1, 1)
        std = torch.sqrt(instance.var(2) + self.eps).view(N, C, 1, 1)
        return mean, std


class EncoderVGG(nn.Module):

    def __init__(self, loss_layers: List[int] = None):
        super(EncoderVGG, self).__init__()

        if loss_layers is None:
            loss_layers = [1, 6, 11, 20]  # relu1_1, relu2_1, relu3_1, relu4_1

        # load pretrained vgg and remove unwanted layers
        vgg19 = models.vgg19(pretrained=True)
        modules = list(vgg19.children())[0][:loss_layers[-1] + 1]

        # create sub nets for each output we need to calculate the loss
        self.subnets: List[Optional[nn.Sequential]] = [None] * len(loss_layers)
        for i in range(len(loss_layers)):
            start = 0 if i == 0 else loss_layers[i - 1] + 1
            end = loss_layers[i] + 1
            self.subnets[i] = nn.Sequential(*modules[start: end])

        for net in self.subnets:
            # set padding mode of conv layers to reflect
            for feature in net:
                if isinstance(feature, nn.Conv2d):
                    feature.padding_mode = 'reflect'

            # freeze weights in all layers
            for param in net.parameters():
                param.requires_grad = False

    def forward(self, x: Tensor) -> List[Optional[Tensor]]:
        feature_maps: List[Optional[Tensor]] = [None] * len(self.subnets)

        inp = x
        for i, net in enumerate(self.subnets):
            inp = net(inp)
            feature_maps[i] = inp

        return feature_maps


class Decoder(nn.Module):

    def __init__(self, encoder: EncoderVGG):
        super(Decoder, self).__init__()

        # iterate backwards through encoder modules
        modules = []
        for i in range(len(encoder.subnets) - 1, -1, -1):
            for j in range(len(encoder.subnets[i]) - 1, -1, -1):
                layer: nn.Module = encoder.subnets[i][j]
                if isinstance(layer, nn.Conv2d):
                    in_channels, out_channels = layer.out_channels, layer.in_channels
                    modules.append(nn.Conv2d(in_channels, out_channels, 3, padding=1, padding_mode='reflect'))
                    modules.append(nn.ReLU(inplace=True))
                elif isinstance(layer, nn.MaxPool2d):
                    modules.append(nn.Upsample(scale_factor=2, mode='nearest'))

        self.features = nn.Sequential(*modules)

    def forward(self, x: Tensor) -> Tensor:
        return self.features(x)


if __name__ == '__main__':
    ada = AdaIN(3)
    # c = torch.randn((2, 3, 4, 4))
    # s = torch.randn((2, 3, 4, 4))
    # ada(c, s)

    encoder = EncoderVGG()
    xx = torch.randn((2, 3, 64, 64))
    oot = encoder(xx)

    model = Decoder(encoder)
    print(sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values()))

    test_c = torch.randn((1, 3, 128, 128))
    test_s = torch.randn((1, 3, 128, 128))

    enc_c = encoder(test_c)[-1]
    enc_s = encoder(test_s)[-1]

    test_ada = ada(enc_c, enc_s)
    print(test_ada.shape)
    dec_out = model(test_ada)
    print(dec_out.shape)
