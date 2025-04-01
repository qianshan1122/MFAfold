import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from itertools import product


class SelfAttention(nn.Module):
    def __init__(self, dim_q, dim_k):
        super(SelfAttention, self).__init__()
        self.dim_q = dim_q
        self.dim_k = dim_k
        # 定义线性变换函数
        self.linear_q = nn.Linear(dim_q, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_q, dim_k, bias=False)

        self._norm_fact = 1 / math.sqrt(dim_k)

    def forward(self, x):
        # x: batch, n, dim_q
        # 根据文本获得相应的维度
        batch, n, dim_q = x.shape
        # 如果条件为 True，则程序继续执行；如果条件为 False，则程序抛出一个 AssertionError 异常，并停止执行。
        assert dim_q == self.dim_q  # 确保输入维度与初始化时的dim_q一致

        q = self.linear_q(x)  # batch, n, dim_k
        k = self.linear_k(x)  # batch, n, dim_k

        # q*k的转置并除以开根号后的dim_k
        dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact
        # 归一化获得attention的相关系数
        # dist = F.softmax(dist, dim=-1)  # batch, n, n
        # attention系数和v相乘，获得最终的得分
        return dist.unsqueeze(1)


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道维度计算权重
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        channel_weights = (avg_out + max_out).view(x.size(0), x.size(1), 1, 1)
        return x * channel_weights  # 通道加权


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 空间维度计算权重
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        spatial_weights = self.sigmoid(self.conv(combined))
        return x * spatial_weights  # 空间加权


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super().__init__()
        self.channel_attn = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attn = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attn(x)  # 先通道注意力
        x = self.spatial_attn(x)  # 后空间注意力
        return x


class ResNet2DBlock(nn.Module):
    def __init__(self, embed_dim, bias=False):
        super().__init__()

        # Bottleneck architecture
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim * 2, kernel_size=3, bias=bias, padding="same"),
            nn.InstanceNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=embed_dim * 2, out_channels=embed_dim * 2, kernel_size=5, bias=bias,
                      padding="same"),
            nn.InstanceNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=embed_dim * 2, out_channels=embed_dim, kernel_size=3, bias=bias, padding="same"),
            nn.InstanceNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.cbam = CBAM(embed_dim)

    def forward(self, x):
        residual = x

        x = self.conv_net(x)
        x = self.cbam(x)
        x = x + residual

        return x


class ResNet2D(nn.Module):
    def __init__(self, embed_dim, num_blocks, bias=False):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                ResNet2DBlock(embed_dim, bias=bias) for _ in range(num_blocks)
            ]
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        return x


class StructurePrediction(nn.Module):
    def __init__(self, hidden_size=32, num_blocks=3, img_ch=20, conv_dim=64, kernel_size=3):
        super().__init__()

        self.conv1 = conv_block(ch_in=img_ch, ch_out=conv_dim)
        self.cnn1 = nn.Conv1d(in_channels=4, out_channels=hidden_size, kernel_size=kernel_size,
                              padding=kernel_size // 2)
        self.cnn2 = nn.Conv1d(in_channels=11, out_channels=hidden_size, kernel_size=kernel_size,
                              padding=kernel_size // 2)
        self.bigru = nn.GRU(hidden_size, hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        self.resnet = ResNet2D(conv_dim, num_blocks)
        self.self_attention = SelfAttention(conv_dim, hidden_size)
        self.conv_out = nn.Conv2d(conv_dim, 1, kernel_size=kernel_size, padding="same")

    def forward(self, one_hot, dpcp, ncp, input1):
        x1 = self.cnn1(one_hot.transpose(1, 2)).transpose(1, 2)
        x1, _ = self.bigru(x1)
        x1 = self.self_attention(x1)

        x2 = self.cnn2(dpcp.transpose(1, 2)).transpose(1, 2)
        x2, _ = self.bigru(x2)
        x2 = self.self_attention(x2)

        x3 = self.cnn1(ncp.transpose(1, 2)).transpose(1, 2)
        x3, _ = self.bigru(x3)
        x3 = self.self_attention(x3)

        x = torch.cat((input1, x1, x2, x3), dim=1)

        x = self.conv1(x)
        x = self.resnet(x)
        x = self.conv_out(x)

        x = x.squeeze(-3)  # B x 1 x L x L => B x L x L

        # Symmetrize the output
        x = torch.triu(x, diagonal=1)
        x = x + x.transpose(-1, -2)

        return x
