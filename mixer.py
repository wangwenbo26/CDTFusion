import torch
import torch.nn as nn
from torch.nn import Conv2d
from einops.layers.torch import Rearrange
import numpy as np


class mixer(nn.Module):
    def __init__(self):
        super(mixer, self).__init__()

        self.conv1 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)

        self.c = 512
        self.concat = lambda x: torch.cat(x, dim=1)

    def forward(self, opt, sar):
        x1 = self.concat([opt, sar])
        x1 = self.relu1(self.conv1(x1))
        x1 = self.relu2(self.conv2(x1))
        x1 = self.relu3(self.conv3(x1))

        return x1


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self,x):
        x = self.net(x)
        return x


class MixerBlock(nn.Module):
    def __init__(self, dim, num_patch, token_dim, channel_dim):
        super().__init__()
        self.token_mixer = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            FeedForward(num_patch, token_dim),
            Rearrange('b d n -> b n d')

        )
        self.channel_mixer = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim)
        )

    def forward(self, x):
        x = x + self.token_mixer(x)
        x = x + self.channel_mixer(x)
        return x


class MLPMixer(nn.Module):
    def __init__(self, in_channels, dim, patch_size, image_size, depth, token_dim, channel_dim,
                 dropout=0.):
        super().__init__()
        assert image_size % patch_size == 0
        self.num_patches = (image_size // patch_size) ** 2

        self.to_embedding = nn.Sequential(
            Conv2d(in_channels=in_channels, out_channels=dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b c h w -> b (h w) c')
            )

        self.mixer_blocks = nn.ModuleList([])
        for _ in range(depth):
            self.mixer_blocks.append(MixerBlock(dim, self.num_patches, token_dim, channel_dim))

        #
        self.layer_normal = nn.LayerNorm(dim)

        self.conv1 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)

        self.c = 512
        self.concat = lambda x: torch.cat(x, dim=1)

    def forward(self, opt, sar):
        x = self.concat([opt, sar])
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.to_embedding(x)
        # print(x.shape)
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        x = self.layer_normal(x)
        b, h_w, c = x.shape
        h = w = int(np.sqrt(h_w))
        x = x.reshape(b, c, h, w)

        return x




import torch
import torch.nn as nn
from torch.nn import Conv2d
from einops.layers.torch import Rearrange
import numpy as np


class Mixer(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim_channel, hidden_dim_token):
        super(Mixer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim_channel = hidden_dim_channel
        self.hidden_dim_token = hidden_dim_token
        self.conv1 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.c = 512
        self.concat = lambda x: torch.cat(x, dim=1)
        self.mix1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim_channel, kernel_size=1),
            nn.LayerNorm([hidden_dim_channel, 64, 64]), # Layer Normalization
            nn.ReLU(),
            nn.Conv2d(hidden_dim_channel, out_channels, kernel_size=1)
        )

        self.mix2 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim_token, kernel_size=1),
            nn.LayerNorm([hidden_dim_token, 64, 64]), # Layer Normalization
            nn.ReLU(),
            nn.Conv2d(hidden_dim_token, out_channels, kernel_size=1)
        )
        self.gelu = nn.GELU()

    def forward(self, opt, sar):
        x1 = self.concat([opt, sar])
        x1 = self.relu1(self.conv1(x1))
        x1 = self.relu2(self.conv2(x1))
        x1 = self.relu3(self.conv3(x1))
        channel_mixed = self.mix1(x1)
        x2 = opt + sar
        token_mixed = self.mix2(x2)
        mixed_features = self.gelu(x1 + x2 + channel_mixed + token_mixed)
        output = mixed_features + opt + sar
        return output


class Mixer_resize(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim_channel, hidden_dim_token, x, y):
        super(Mixer_resize, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim_channel = hidden_dim_channel
        self.hidden_dim_token = hidden_dim_token
        self.conv1 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.c = 512
        self.concat = lambda x: torch.cat(x, dim=1)
        self.mix1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim_channel, kernel_size=1),
            nn.LayerNorm([hidden_dim_channel, x//8, y//8]), # Layer Normalization
            nn.ReLU(),
            nn.Conv2d(hidden_dim_channel, out_channels, kernel_size=1)
        )

        self.mix2 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim_token, kernel_size=1),
            nn.LayerNorm([hidden_dim_token, x//8, y//8]), # Layer Normalization
            nn.ReLU(),
            nn.Conv2d(hidden_dim_token, out_channels, kernel_size=1)
        )
        self.gelu = nn.GELU()

    def forward(self, opt, sar):
        x1 = self.concat([opt, sar])
        x1 = self.relu1(self.conv1(x1))
        x1 = self.relu2(self.conv2(x1))
        x1 = self.relu3(self.conv3(x1))
        channel_mixed = self.mix1(x1)
        x2 = opt + sar
        token_mixed = self.mix2(x2)
        mixed_features = self.gelu(x1 + x2 + channel_mixed + token_mixed)
        output = mixed_features + opt + sar
        return output


class Mixer_resize1(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim_channel, hidden_dim_token):
        super(Mixer_resize1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim_channel = hidden_dim_channel
        self.hidden_dim_token = hidden_dim_token

        self.conv1 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.c = 512
        self.concat = lambda x: torch.cat(x, dim=1)

        # 使用InstanceNorm或GroupNorm替代LayerNorm
        self.mix1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim_channel, kernel_size=1),
            nn.GroupNorm(1, hidden_dim_channel),  # 使用GroupNorm替代LayerNorm
            nn.ReLU(),
            nn.Conv2d(hidden_dim_channel, out_channels, kernel_size=1)
        )

        self.mix2 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim_token, kernel_size=1),
            nn.GroupNorm(1, hidden_dim_token),  # 使用GroupNorm替代LayerNorm
            nn.ReLU(),
            nn.Conv2d(hidden_dim_token, out_channels, kernel_size=1)
        )
        self.gelu = nn.GELU()

    def forward(self, opt, sar):
        x1 = self.concat([opt, sar])
        x1 = self.relu1(self.conv1(x1))
        x1 = self.relu2(self.conv2(x1))
        x1 = self.relu3(self.conv3(x1))
        channel_mixed = self.mix1(x1)
        x2 = opt + sar
        token_mixed = self.mix2(x2)
        mixed_features = self.gelu(x1 + x2 + channel_mixed + token_mixed)
        output = mixed_features + opt + sar
        return output