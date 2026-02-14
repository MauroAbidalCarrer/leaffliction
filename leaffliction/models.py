from itertools import pairwise, repeat
from torch import nn, Tensor


class CNN(nn.Module):
    def __init__(
        self,
        kernels_per_layer: list[int],
        mlp_width: int,
        mlp_depth: int,
        n_classes: int,
    ):
        super().__init__()
        conv_layers = []
        channels = [3] + kernels_per_layer
        for in_channels, out_channels in pairwise(channels):
            conv_layer = nn.Conv2d(
                in_channels,
                out_channels,
                5,
                padding=2,
            )
            conv_layers.append(conv_layer)
        self.conv_layers = nn.ModuleList(conv_layers)
        self.batch_norms = nn.ModuleList([
            nn.LazyBatchNorm2d()
            for _ in range(len(conv_layers))
        ])

        self.linear_layers = []
        for width in repeat(mlp_width, mlp_depth - 1):
            self.linear_layers.append(nn.LazyLinear(width))
        self.linear_layers.append(nn.LazyLinear(n_classes))
        self.linear_layers = nn.ModuleList(self.linear_layers)

    def forward(self, x: Tensor) -> Tensor:
        layer_it = enumerate(zip(self.batch_norms, self.conv_layers))
        for layer_idx, (b_norm, conv) in layer_it:
            x = b_norm(x)
            x = conv(x)
            x = nn.functional.relu(x)
            x = nn.functional.max_pool2d(x, 2)

        x = x.flatten(1)
        for layer_idx, linear_layer in enumerate(self.linear_layers):
            x = linear_layer(x)
            if layer_idx != len(self.linear_layers) - 1:
                x = nn.functional.relu(x)
        return x
