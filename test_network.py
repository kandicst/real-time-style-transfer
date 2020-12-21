from torch import nn
import torch


class TestNet(nn.Module):
    def __init__(self, encoder, decoder, net_type="vgg"):
        super(TestNet, self).__init__()
        encoder_layers = list(encoder.children())

        self.encoder1 = nn.Sequential(*encoder_layers[:4])
        self.encoder2 = nn.Sequential(*encoder_layers[4:11])
        self.encoder3 = nn.Sequential(*encoder_layers[11:18])
        self.encoder4 = nn.Sequential(*encoder_layers[18:31])
        for nnet in [self.encoder1, self.encoder2, self.encoder3, self.encoder4]:
            for layer in nnet.modules():
                if type(layer) != nn.Sequential:
                    layer.requires_grad = False

        self.decoder = decoder
        self.loss = nn.MSELoss()

    def encode_with_layers(self, input):
        enc_list = [input]
        for i, nnet in enumerate([self.encoder1, self.encoder2,
                                  self.encoder3, self.encoder4]):
            enc_list.append(nnet(enc_list[i]))
        return enc_list

    def encode(self, input):
        enc_input = input
        for nnet in [self.encoder1, self.encoder2, self.encoder3, self.encoder4]:
            enc_input = nnet(enc_input)
        return enc_input

    def content_loss(self, input, target):
        return self.loss(input, target)

    def style_loss(self, input, style_target):
        input_std, input_mean = std_and_mean(input)
        style_std, style_mean = std_and_mean(style_target)
        return self.loss(input_mean, style_mean) + self.loss(input_std, style_std)

    def forward(self, content, style, alpha=1.0, full=True):
        enc_x = self.encode(content)
        enc_y = self.encode_with_layers(style)

        target = adaIN(enc_x, enc_y[-1])
        target = (1 - alpha) * enc_x + alpha * target
        dec = self.decoder(target)
        dec_layers = self.encode_with_layers(dec)

        content_loss = self.content_loss(dec_layers[-1], target)
        style_loss = self.style_loss(dec_layers[1], enc_y[1])
        for i in range(2, 5):
            style_loss += self.style_loss(dec_layers[i], enc_y[i])
        return dec, content_loss, style_loss


class Encoder(nn.Module):

    def __init__(self, encoder):
        super(Encoder, self).__init__()
        encoder_layers = list(encoder.children())

        self.encoder1 = nn.Sequential(*encoder_layers[:4])
        self.encoder2 = nn.Sequential(*encoder_layers[4:11])
        self.encoder3 = nn.Sequential(*encoder_layers[11:18])
        self.encoder4 = nn.Sequential(*encoder_layers[18:31])
        for nnet in [self.encoder1, self.encoder2, self.encoder3, self.encoder4]:
            for layer in nnet.modules():
                if type(layer) != nn.Sequential:
                    layer.requires_grad = False

    def forward(self, x):
        enc_list = [x]
        for i, nnet in enumerate([self.encoder1, self.encoder2,
                                  self.encoder3, self.encoder4]):
            enc_list.append(nnet(enc_list[i]))
        return enc_list[1:]


def std_and_mean(features, eps=1e-5):
    size = features.size()
    N, C = size[:2]
    features_var = features.view(N, C, -1).var(dim=2) + eps
    features_std = features_var.sqrt().view(N, C, 1, 1)
    features_mean = features.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return features_std, features_mean


def adaIN(x, y):
    size = x.size()
    std_y, mean_y = std_and_mean(y)
    std_x, mean_x = std_and_mean(x)
    normalized_feat = (x - mean_x.expand(size)) / std_x.expand(size)
    return normalized_feat * std_y.expand(size) + mean_y.expand(size)


def get_vgg():
    vgg = nn.Sequential(
        nn.Conv2d(3, 3, (1, 1)),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        # nn.ReLU(inplace=True),  # relu1-1
        nn.Conv2d(3, 64, (3, 3)),
        nn.ReLU(inplace=True),  # relu1-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(inplace=True),  # relu1-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 128, (3, 3)),
        nn.ReLU(inplace=True),  # relu2-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(inplace=True),  # relu2-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 256, (3, 3)),
        nn.ReLU(inplace=True),  # relu3-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(inplace=True),  # relu3-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(inplace=True),  # relu3-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(inplace=True),  # relu3-4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 512, (3, 3)),
        nn.ReLU(inplace=True),  # relu4-1
        # -----------------------------------------------------
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu4-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu4-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu4-4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu5-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu5-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu5-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU()  # relu5-4
    )
    vgg.load_state_dict(torch.load("vgg_normalized.pth"))
    print('LOADED')
    return vgg


def get_vgg_mine():
    vgg = nn.Sequential(
        nn.Conv2d(3, 3, (1, 1)),
        nn.Conv2d(3, 64, (3, 3), padding=1, padding_mode='reflect'),
        nn.ReLU(inplace=True),  # relu1-1
        nn.Conv2d(64, 64, (3, 3), padding=1, padding_mode='reflect'),
        nn.ReLU(inplace=True),  # relu1-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.Conv2d(64, 128, (3, 3), padding=1, padding_mode='reflect'),
        nn.ReLU(inplace=True),  # relu2-1
        nn.Conv2d(128, 128, (3, 3), padding=1, padding_mode='reflect'),
        nn.ReLU(inplace=True),  # relu2-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.Conv2d(128, 256, (3, 3), padding=1, padding_mode='reflect'),
        nn.ReLU(inplace=True),  # relu3-1
        nn.Conv2d(256, 256, (3, 3), padding=1, padding_mode='reflect'),
        nn.ReLU(inplace=True),  # relu3-2
        nn.Conv2d(256, 256, (3, 3), padding=1, padding_mode='reflect'),
        nn.ReLU(inplace=True),  # relu3-3
        nn.Conv2d(256, 256, (3, 3), padding=1, padding_mode='reflect'),
        nn.ReLU(inplace=True),  # relu3-4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.Conv2d(256, 512, (3, 3), padding=1, padding_mode='reflect'),
        nn.ReLU(inplace=True),  # relu4-1
        # -----------------------------------------------------
        nn.Conv2d(512, 512, (3, 3), padding=1, padding_mode='reflect'),
        nn.ReLU(),  # relu4-2
        nn.Conv2d(512, 512, (3, 3), padding=1, padding_mode='reflect'),
        nn.ReLU(),  # relu4-3
        nn.Conv2d(512, 512, (3, 3), padding=1, padding_mode='reflect'),
        nn.ReLU(),  # relu4-4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.Conv2d(512, 512, (3, 3), padding=1, padding_mode='reflect'),
        nn.ReLU(),  # relu5-1
        nn.Conv2d(512, 512, (3, 3), padding=1, padding_mode='reflect'),
        nn.ReLU(),  # relu5-2
        nn.Conv2d(512, 512, (3, 3), padding=1, padding_mode='reflect'),
        nn.ReLU(),  # relu5-3
        nn.Conv2d(512, 512, (3, 3), padding=1, padding_mode='reflect'),
        nn.ReLU()  # relu5-4
    )
    vgg.load_state_dict(torch.load("vgg19_normalized_mine.pth"))
    return vgg


def get_decoder():
    decoder = nn.Sequential(
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 256, (3, 3)),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 128, (3, 3)),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 64, (3, 3)),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 3, (3, 3))
    )
    return decoder


def get_net():
    return TestNet(get_vgg(), get_decoder())


def get_encoder():
    return Encoder(get_vgg())


if __name__ == '__main__':
    vgg19 = get_vgg_mine()

    print(vgg19)
