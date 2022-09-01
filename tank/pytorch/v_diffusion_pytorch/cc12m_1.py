from functools import partial
import math

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return self.main(input) + self.skip(input)


class ResLinearBlock(ResidualBlock):
    def __init__(self, f_in, f_mid, f_out, is_last=False):
        skip = None if f_in == f_out else nn.Linear(f_in, f_out, bias=False)
        super().__init__(
            [
                nn.Linear(f_in, f_mid),
                nn.ReLU(inplace=True),
                nn.Linear(f_mid, f_out),
                nn.ReLU(inplace=True) if not is_last else nn.Identity(),
            ],
            skip,
        )


class Modulation2d(nn.Module):
    def __init__(self, state, feats_in, c_out):
        super().__init__()
        self.state = state
        self.layer = nn.Linear(feats_in, c_out * 2, bias=False)

    def forward(self, input):
        scales, shifts = self.layer(self.state["cond"]).chunk(2, dim=-1)
        return torch.addcmul(
            shifts[..., None, None], input, scales[..., None, None] + 1
        )


class ResModConvBlock(ResidualBlock):
    def __init__(self, state, feats_in, c_in, c_mid, c_out, is_last=False):
        skip = None if c_in == c_out else nn.Conv2d(c_in, c_out, 1, bias=False)
        super().__init__(
            [
                nn.Conv2d(c_in, c_mid, 3, padding=1),
                nn.GroupNorm(1, c_mid, affine=False),
                Modulation2d(state, feats_in, c_mid),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_mid, c_out, 3, padding=1),
                nn.GroupNorm(1, c_out, affine=False)
                if not is_last
                else nn.Identity(),
                Modulation2d(state, feats_in, c_out)
                if not is_last
                else nn.Identity(),
                nn.ReLU(inplace=True) if not is_last else nn.Identity(),
            ],
            skip,
        )


class SkipBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return torch.cat([self.main(input), self.skip(input)], dim=1)


class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.0):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = nn.Parameter(
            torch.randn([out_features // 2, in_features]) * std
        )
        self.weight.requires_grad_(False)
        # self.register_buffer('weight', torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


class SelfAttention2d(nn.Module):
    def __init__(self, c_in, n_head=1, dropout_rate=0.1):
        super().__init__()
        assert c_in % n_head == 0
        self.norm = nn.GroupNorm(1, c_in)
        self.n_head = n_head
        self.qkv_proj = nn.Conv2d(c_in, c_in * 3, 1)
        self.out_proj = nn.Conv2d(c_in, c_in, 1)
        self.dropout = (
            nn.Identity()
        )  # nn.Dropout2d(dropout_rate, inplace=True)

    def forward(self, input):
        n, c, h, w = input.shape
        qkv = self.qkv_proj(self.norm(input))
        qkv = qkv.view(
            [n, self.n_head * 3, c // self.n_head, h * w]
        ).transpose(2, 3)
        q, k, v = qkv.chunk(3, dim=1)
        scale = k.shape[3] ** -0.25
        att = ((q * scale) @ (k.transpose(2, 3) * scale)).softmax(3)
        y = (att @ v).transpose(2, 3).contiguous().view([n, c, h, w])
        return input + self.dropout(self.out_proj(y))


def expand_to_planes(input, shape):
    return input[..., None, None].repeat([1, 1, shape[2], shape[3]])


class CC12M1Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.shape = (3, 256, 256)
        self.clip_model = "ViT-B/16"
        self.min_t = 0.0
        self.max_t = 1.0

        c = 128  # The base channel count
        cs = [c, c * 2, c * 2, c * 4, c * 4, c * 8, c * 8]

        self.mapping_timestep_embed = FourierFeatures(1, 128)
        self.mapping = nn.Sequential(
            ResLinearBlock(512 + 128, 1024, 1024),
            ResLinearBlock(1024, 1024, 1024, is_last=True),
        )

        with torch.no_grad():
            for param in self.mapping.parameters():
                param *= 0.5**0.5

        self.state = {}
        conv_block = partial(ResModConvBlock, self.state, 1024)

        self.timestep_embed = FourierFeatures(1, 16)
        self.down = nn.AvgPool2d(2)
        self.up = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )

        self.net = nn.Sequential(  # 256x256
            conv_block(3 + 16, cs[0], cs[0]),
            conv_block(cs[0], cs[0], cs[0]),
            conv_block(cs[0], cs[0], cs[0]),
            conv_block(cs[0], cs[0], cs[0]),
            SkipBlock(
                [
                    self.down,  # 128x128
                    conv_block(cs[0], cs[1], cs[1]),
                    conv_block(cs[1], cs[1], cs[1]),
                    conv_block(cs[1], cs[1], cs[1]),
                    conv_block(cs[1], cs[1], cs[1]),
                    SkipBlock(
                        [
                            self.down,  # 64x64
                            conv_block(cs[1], cs[2], cs[2]),
                            conv_block(cs[2], cs[2], cs[2]),
                            conv_block(cs[2], cs[2], cs[2]),
                            conv_block(cs[2], cs[2], cs[2]),
                            SkipBlock(
                                [
                                    self.down,  # 32x32
                                    conv_block(cs[2], cs[3], cs[3]),
                                    conv_block(cs[3], cs[3], cs[3]),
                                    conv_block(cs[3], cs[3], cs[3]),
                                    conv_block(cs[3], cs[3], cs[3]),
                                    SkipBlock(
                                        [
                                            self.down,  # 16x16
                                            conv_block(cs[3], cs[4], cs[4]),
                                            SelfAttention2d(
                                                cs[4], cs[4] // 64
                                            ),
                                            conv_block(cs[4], cs[4], cs[4]),
                                            SelfAttention2d(
                                                cs[4], cs[4] // 64
                                            ),
                                            conv_block(cs[4], cs[4], cs[4]),
                                            SelfAttention2d(
                                                cs[4], cs[4] // 64
                                            ),
                                            conv_block(cs[4], cs[4], cs[4]),
                                            SelfAttention2d(
                                                cs[4], cs[4] // 64
                                            ),
                                            SkipBlock(
                                                [
                                                    self.down,  # 8x8
                                                    conv_block(
                                                        cs[4], cs[5], cs[5]
                                                    ),
                                                    SelfAttention2d(
                                                        cs[5], cs[5] // 64
                                                    ),
                                                    conv_block(
                                                        cs[5], cs[5], cs[5]
                                                    ),
                                                    SelfAttention2d(
                                                        cs[5], cs[5] // 64
                                                    ),
                                                    conv_block(
                                                        cs[5], cs[5], cs[5]
                                                    ),
                                                    SelfAttention2d(
                                                        cs[5], cs[5] // 64
                                                    ),
                                                    conv_block(
                                                        cs[5], cs[5], cs[5]
                                                    ),
                                                    SelfAttention2d(
                                                        cs[5], cs[5] // 64
                                                    ),
                                                    SkipBlock(
                                                        [
                                                            self.down,  # 4x4
                                                            conv_block(
                                                                cs[5],
                                                                cs[6],
                                                                cs[6],
                                                            ),
                                                            SelfAttention2d(
                                                                cs[6],
                                                                cs[6] // 64,
                                                            ),
                                                            conv_block(
                                                                cs[6],
                                                                cs[6],
                                                                cs[6],
                                                            ),
                                                            SelfAttention2d(
                                                                cs[6],
                                                                cs[6] // 64,
                                                            ),
                                                            conv_block(
                                                                cs[6],
                                                                cs[6],
                                                                cs[6],
                                                            ),
                                                            SelfAttention2d(
                                                                cs[6],
                                                                cs[6] // 64,
                                                            ),
                                                            conv_block(
                                                                cs[6],
                                                                cs[6],
                                                                cs[6],
                                                            ),
                                                            SelfAttention2d(
                                                                cs[6],
                                                                cs[6] // 64,
                                                            ),
                                                            conv_block(
                                                                cs[6],
                                                                cs[6],
                                                                cs[6],
                                                            ),
                                                            SelfAttention2d(
                                                                cs[6],
                                                                cs[6] // 64,
                                                            ),
                                                            conv_block(
                                                                cs[6],
                                                                cs[6],
                                                                cs[6],
                                                            ),
                                                            SelfAttention2d(
                                                                cs[6],
                                                                cs[6] // 64,
                                                            ),
                                                            conv_block(
                                                                cs[6],
                                                                cs[6],
                                                                cs[6],
                                                            ),
                                                            SelfAttention2d(
                                                                cs[6],
                                                                cs[6] // 64,
                                                            ),
                                                            conv_block(
                                                                cs[6],
                                                                cs[6],
                                                                cs[5],
                                                            ),
                                                            SelfAttention2d(
                                                                cs[5],
                                                                cs[5] // 64,
                                                            ),
                                                            self.up,
                                                        ]
                                                    ),
                                                    conv_block(
                                                        cs[5] * 2, cs[5], cs[5]
                                                    ),
                                                    SelfAttention2d(
                                                        cs[5], cs[5] // 64
                                                    ),
                                                    conv_block(
                                                        cs[5], cs[5], cs[5]
                                                    ),
                                                    SelfAttention2d(
                                                        cs[5], cs[5] // 64
                                                    ),
                                                    conv_block(
                                                        cs[5], cs[5], cs[5]
                                                    ),
                                                    SelfAttention2d(
                                                        cs[5], cs[5] // 64
                                                    ),
                                                    conv_block(
                                                        cs[5], cs[5], cs[4]
                                                    ),
                                                    SelfAttention2d(
                                                        cs[4], cs[4] // 64
                                                    ),
                                                    self.up,
                                                ]
                                            ),
                                            conv_block(
                                                cs[4] * 2, cs[4], cs[4]
                                            ),
                                            SelfAttention2d(
                                                cs[4], cs[4] // 64
                                            ),
                                            conv_block(cs[4], cs[4], cs[4]),
                                            SelfAttention2d(
                                                cs[4], cs[4] // 64
                                            ),
                                            conv_block(cs[4], cs[4], cs[4]),
                                            SelfAttention2d(
                                                cs[4], cs[4] // 64
                                            ),
                                            conv_block(cs[4], cs[4], cs[3]),
                                            SelfAttention2d(
                                                cs[3], cs[3] // 64
                                            ),
                                            self.up,
                                        ]
                                    ),
                                    conv_block(cs[3] * 2, cs[3], cs[3]),
                                    conv_block(cs[3], cs[3], cs[3]),
                                    conv_block(cs[3], cs[3], cs[3]),
                                    conv_block(cs[3], cs[3], cs[2]),
                                    self.up,
                                ]
                            ),
                            conv_block(cs[2] * 2, cs[2], cs[2]),
                            conv_block(cs[2], cs[2], cs[2]),
                            conv_block(cs[2], cs[2], cs[2]),
                            conv_block(cs[2], cs[2], cs[1]),
                            self.up,
                        ]
                    ),
                    conv_block(cs[1] * 2, cs[1], cs[1]),
                    conv_block(cs[1], cs[1], cs[1]),
                    conv_block(cs[1], cs[1], cs[1]),
                    conv_block(cs[1], cs[1], cs[0]),
                    self.up,
                ]
            ),
            conv_block(cs[0] * 2, cs[0], cs[0]),
            conv_block(cs[0], cs[0], cs[0]),
            conv_block(cs[0], cs[0], cs[0]),
            conv_block(cs[0], cs[0], 3, is_last=True),
        )

        with torch.no_grad():
            for param in self.net.parameters():
                param *= 0.5**0.5

    def forward(self, input, timestep_embed, selfcond):
        self.state["cond"] = selfcond
        out = self.net(torch.cat([input, timestep_embed], dim=1))
        self.state.clear()
        return out
