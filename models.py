import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-2)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        query = self.query_conv(x).view(
            batch_size, -1, width*height).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, width*height)
        attention = self.softmax(torch.bmm(query, key))
        value = self.value_conv(x).view(batch_size, -1, width*height)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        out = self.gamma * out + x
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.res = nn.Sequential(
            nn.Conv2d(in_channels, in_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        return x + self.res(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_attention=False):
        super(DownBlock, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, stride=1, padding=4),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            ResidualBlock(out_channels),
            ResidualBlock(out_channels)
        )
        self.use_attention = use_attention
        if use_attention:
            self.attention = SelfAttention(out_channels)

    def forward(self, x):
        x = self.encoder(x)
        if self.use_attention:
            x = self.attention(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.inchannel = in_channels
        self.outchannel = out_channels
        super(UpBlock, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels,
                               kernel_size=3, stride=1, padding=4),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            ResidualBlock(out_channels),
            ResidualBlock(out_channels)
        )

    def forward(self, x1, x2):
        # print(x1.shape)
        # print(x2.shape)
        # print(self.inchannel)
        # print(self.outchannel)
        x1 = self.decoder(x1)
        x = torch.cat([x1, x2], dim=1)
        return x


class Generator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Generator, self).__init__()
        # print(" in: ", in_channels)
        # print("out: ", out_channels)
        self.enc1 = DownBlock(in_channels, 64)
        self.enc2 = DownBlock(64, 128, use_attention=True)
        self.enc3 = DownBlock(128, 256)
        self.enc4 = DownBlock(256, 512)

        self.dec1 = UpBlock(512, 256)
        self.dec2 = UpBlock(512, 128)
        self.dec3 = UpBlock(256, 64)
        self.final = nn.Sequential(
            nn.Conv2d(128, out_channels, kernel_size=7, stride=1, padding=0),
            nn.Tanh()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        d1 = self.dec1(e4, e3)
        d2 = self.dec2(d1, e2)
        d3 = self.dec3(d2, e1)
        d4 = self.final(d3)
        # print(f'e1:{e1.shape},e2:{e2.shape},e3:{e3.shape},e4:{e4.shape}')
        # print(f'd1:{d1.shape},d2:{d2.shape},d3:{d3.shape},d4:{d4.shape}')
        return d4


class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 6, 1, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # print(6)
        # print(x.shape)
        return self.main(x).squeeze()
