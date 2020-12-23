from torch import nn
from torch import Tensor

from typing import Tuple, List, Optional

from vgg import get_vgg

from loss import get_instance_statistics


class AdaIN(nn.Module):

    def __init__(self, eps: float = 1e-5):
        super(AdaIN, self).__init__()
        self.eps = eps

    def forward(self, content: Tensor, style: Tensor) -> Tensor:
        assert len(content.size()) == len(style.size()) == 4  # make sure its NCHW format
        # assert list(content.size()) == list(style.size())   # make sure the shapes match

        content_mean, content_std = get_instance_statistics(content)
        style_mean, style_std = get_instance_statistics(style)

        # Equation (8)
        out = (content - content_mean) / content_std
        out = out * style_std + style_mean
        return out


class EncoderVGG(nn.Module):

    def __init__(self, loss_layers: List[int] = None):
        super(EncoderVGG, self).__init__()

        if loss_layers is None:
            loss_layers = [1, 6, 11, 20]  # relu1_1, relu2_1, relu3_1, relu4_1

        # add 1 to every idx because of first layer which prepossesses the image
        loss_layers = [x + 1 for x in loss_layers]

        # load pretrained vgg and remove unwanted layers
        vgg19 = get_vgg()
        modules = list(vgg19.children())[:loss_layers[-1] + 1]

        # create sub nets for each output we need to calculate the loss
        self.subnets: List[Optional[nn.Sequential]] = [None] * len(loss_layers)
        for i in range(len(loss_layers)):
            start = 0 if i == 0 else loss_layers[i - 1] + 1
            end = loss_layers[i] + 1
            self.subnets[i] = nn.Sequential(*modules[start: end])

        for net in self.subnets:
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

    def __init__(self, encoder: nn.Module, ignore_last=2):
        super(Decoder, self).__init__()

        # iterate backwards through encoder modules to create a mirror-like decoder
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

        self.features = nn.Sequential(*modules[:-ignore_last])
        x = 'skloni -1'

    def forward(self, x: Tensor) -> Tensor:
        return self.features(x)


def test_difference(t1, t2):
    x = t1.size()
    y = t2.size()
    print(f'Difference is {(t1 - t2).abs().max()}')


if __name__ == '__main__':
    # ada = AdaIN()
    encoder = EncoderVGG()

    model = Decoder(encoder)
    #
    # test_c = torch.randn((1, 3, 256, 256))
    # test_s = torch.randn((1, 3, 256, 256))
    #
    # enc_c = encoder(test_c)[-1]
    # enc_s = encoder(test_s)[-1]
    #
    # test_ada = ada(enc_c, enc_s)
    # dec_out = model(test_ada)
    # print(dec_out.shape)
    #
    # dec_enc_out = encoder(dec_out)
    # print(dec_enc_out[-1].shape)
    # print(test_ada.shape)
