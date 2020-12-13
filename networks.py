import torch


class AdaIN(torch.nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(AdaIN, self).__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x):
        content, style = x

        content_mean, content_std = self.get_mean_and_std(content)
        style_mean, style_std = self.get_mean_and_std(style)

        out = (content - content_mean) / content_std
        out = out * style_std + style_mean
        return out

    def get_mean_and_std(self, x):
        N, C, H, W = x.size()
        x_view = x.view(N, C, -1)
        mean = x_view.mean(2).view(N, C, 1, 1)
        std = torch.sqrt(x_view.var(2) + self.eps).view(N, C, 1, 1)
        return mean, std


if __name__ == '__main__':
    ada = AdaIN(3)
    c = torch.randn((2, 3, 4, 4))
    s = torch.randn((2, 3, 4, 4))
    ada((c, s))
