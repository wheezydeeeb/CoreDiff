import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from einops import rearrange


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class single_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class up(nn.Module):
    def __init__(self, in_ch):
        super(up, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        x = x2 + x1
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class adjust_net(nn.Module):
    def __init__(self, out_channels=64, middle_channels=32):
        super(adjust_net, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(2, middle_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),

            nn.Conv2d(middle_channels, middle_channels*2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),

            nn.Conv2d(middle_channels*2, middle_channels*4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),

            nn.Conv2d(middle_channels*4, out_channels*2, 1, padding=0)
        )

    def forward(self, x):
        out = self.model(x)
        out = F.adaptive_avg_pool2d(out, (1,1))
        out1 = out[:, :out.shape[1]//2]
        out2 = out[:, out.shape[1]//2:]
        return out1, out2


class res_fft_conv(nn.Module):
    def __init__(self, in_channels):
        super(res_fft_conv, self).__init__()
        self.img_conv = nn.Conv2d(in_channels,   in_channels,   kernel_size=3, stride=1, padding=1)
        self.fft_conv = nn.Conv2d(in_channels*2, in_channels*2, kernel_size=1, stride=1, padding=0)

    def forward(self, x, residue=None):

        # Handling residual
        x = self.img_conv(x)
        if residue is not None:
            x = x + residue
        x = F.gelu(x)

        # Frequency domain
        _, _, H, W = x.shape
        fft = torch.fft.rfft2(x, s=(H, W), dim=(2, 3), norm='ortho')
        fft = torch.cat([fft.real, fft.imag], dim=1)
        fft = F.relu(self.fft_conv(fft))
        fft_real, fft_imag = torch.chunk(fft, 2, dim=1)        
        fft = torch.complex(fft_real, fft_imag)
        fft = torch.fft.irfft2(fft, s=(H, W), dim=(2, 3), norm='ortho')

        # Image domain  
        x_conv = F.gelu(self.img_conv(x))

        # Mixing (residual, image, fourier)
        output = x + x_conv + fft
        return output  

 # The architecture of U-Net refers to "Toward Convolutional Blind Denoising of Real Photographs",
 # official MATLAB implementation: https://github.com/GuoShi28/CBDNet.
 # unofficial PyTorch implementation: https://github.com/IDKiro/CBDNet-pytorch/tree/master.
 # We improved it by adding time step embedding and EMM module, while removing the noise estimation network.
class UNet(nn.Module):
    # def __init__(self, in_channels=2, out_channels=1):
    #     super(UNet, self).__init__()

    #     dim = 32
    #     self.time_mlp = nn.Sequential(
    #         SinusoidalPosEmb(dim),
    #         nn.Linear(dim, dim * 4),
    #         nn.GELU(),
    #         nn.Linear(dim * 4, dim)
    #     )

    #     self.inc = nn.Sequential(
    #         single_conv(in_channels, 64),
    #         single_conv(64, 64)
    #     )

    #     self.down1 = nn.AvgPool2d(2)
    #     self.mlp1 = nn.Sequential(
    #         nn.GELU(),
    #         nn.Linear(dim, 64)
    #     )
    #     self.adjust1 = adjust_net(64)
    #     self.conv1 = nn.Sequential(
    #         single_conv(64, 128),
    #         single_conv(128, 128),
    #         single_conv(128, 128)
    #     )

    #     self.down2 = nn.AvgPool2d(2)
    #     self.mlp2 = nn.Sequential(
    #         nn.GELU(),
    #         nn.Linear(dim, 128)
    #     )
    #     self.adjust2 = adjust_net(128)
    #     self.conv2 = nn.Sequential(
    #         single_conv(128, 256),
    #         single_conv(256, 256),
    #         single_conv(256, 256),
    #         single_conv(256, 256),
    #         single_conv(256, 256),
    #         single_conv(256, 256)
    #     )

    #     self.up1 = up(256)
    #     self.mlp3 = nn.Sequential(
    #         nn.GELU(),
    #         nn.Linear(dim, 128)
    #     )
    #     self.adjust3 = adjust_net(128)
    #     self.conv3 = nn.Sequential(
    #         single_conv(128, 128),
    #         single_conv(128, 128),
    #         single_conv(128, 128)
    #     )

    #     self.up2 = up(128)
    #     self.mlp4 = nn.Sequential(
    #         nn.GELU(),
    #         nn.Linear(dim, 64)
    #     )
    #     self.adjust4 = adjust_net(64)
    #     self.conv4 = nn.Sequential(
    #         single_conv(64, 64),
    #         single_conv(64, 64)
    #     )

    #     self.gelu = nn.GELU()

    #     self.outc = outconv(64, out_channels)

    def __init__(self, in_channels=2, out_channels=1):
        super(UNet, self).__init__()

        dim = 32
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

        self.inc = nn.Sequential(
            single_conv(in_channels, 64),
            single_conv(64, 64)
        )

        self.down1 = nn.AvgPool2d(2)
        self.mlp1 = nn.Sequential(
            nn.GELU(),
            nn.Linear(dim, 64)
        )
        self.adjust1 = adjust_net(64)
        self.conv1 = nn.Sequential(
            single_conv(64, 128),
            single_conv(128, 128),
            single_conv(128, 128)
        )

        self.down2 = nn.AvgPool2d(2)
        self.mlp2 = nn.Sequential(
            nn.GELU(),
            nn.Linear(dim, 128)
        )
        self.adjust2 = adjust_net(128)
        self.conv2 = nn.Sequential(
            single_conv(128, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256)
        )

        self.up1 = up(256)
        self.mlp3 = nn.Sequential(
            nn.GELU(),
            nn.Linear(dim, 128)
        )
        self.adjust3 = adjust_net(128)
        self.conv3 = nn.Sequential(
            single_conv(128, 128),
            single_conv(128, 128),
            single_conv(128, 128)
        )

        self.up2 = up(128)
        self.mlp4 = nn.Sequential(
            nn.GELU(),
            nn.Linear(dim, 64)
        )
        self.adjust4 = adjust_net(64)
        self.conv4 = nn.Sequential(
            single_conv(64, 64),
            single_conv(64, 64)
        )

        self.gelu = nn.GELU()

        self.outc = nn.Sequential(
            single_conv(64, 64),
            outconv(64, out_channels)
        )

        self.resfftconv64 = res_fft_conv(64)
        self.resfftconv128 = res_fft_conv(128) 

    def forward(self, x, t, x_adjust, adjust):
        inx = self.inc(x)
        time_emb = self.time_mlp(t)

        # residual_x = 0
        # if not adjust:
        residual_x = x[:, 1].unsqueeze(1)

        # resfftconv Block 1
        res_inx = self.resfftconv64(inx)

        # Encoder Block 1
        # Downsample, adjust then increase channels in order
        down1 = self.down1(inx)
        condition1 = self.mlp1(time_emb)
        b, c = condition1.shape
        condition1 = rearrange(condition1, 'b c -> b c 1 1')
        if adjust:
            gamma1, beta1 = self.adjust1(x_adjust)
            down1 = down1 + gamma1 * condition1 + beta1
        else:
            down1 = down1 + condition1
        conv1 = self.conv1(down1)
        # resfftconv Block 2
        conv1 = self.resfftconv128(conv1)
        
        # Encoder Block 2
        down2 = self.down2(conv1)
        condition2 = self.mlp2(time_emb)
        b, c = condition2.shape
        condition2 = rearrange(condition2, 'b c -> b c 1 1')
        if adjust:
            gamma2, beta2 = self.adjust2(x_adjust)
            down2 = down2 + gamma2 * condition2 + beta2
        else:
            down2 = down2 + condition2
        conv2 = self.conv2(down2)

        # Decoder Block 1
        # Upsample, increase channels then adjust in order 
        up1 = self.up1(conv2, conv1)
        condition3 = self.mlp3(time_emb)
        b, c = condition3.shape
        condition3 = rearrange(condition3, 'b c -> b c 1 1')
        if adjust:
            gamma3, beta3 = self.adjust3(x_adjust)
            up1 = up1 + gamma3 * condition3 + beta3
        else:
            up1 = up1 + condition3
        up1 = self.resfftconv128(up1, residue=conv1)
        conv3 = self.conv3(up1)

        # Decoder Block 2
        up2 = self.up2(conv3, inx)
        condition4 = self.mlp4(time_emb)
        b, c = condition4.shape
        condition4 = rearrange(condition4, 'b c -> b c 1 1')
        if adjust:
            gamma4, beta4 = self.adjust4(x_adjust)
            up2 = up2 + gamma4 * condition4 + beta4
        else:
            up2 = up2 + condition4
        up2 = self.resfftconv64(up2, residue=res_inx)
        conv4 = self.conv4(up2)

        # Output conv
        out = self.outc(conv4)

        # residual_x connection with GELU
        # out = self.gelu(out + residual_x)
        out = out + residual_x
        return out

    # def forward(self, x, t, x_adjust, adjust):
    #     inx = self.inc(x)
    #     time_emb = self.time_mlp(t)
    #     residual_x = 0
    #     if not adjust:
    #         residual_x = x[:, 1].unsqueeze(1)

    #     # Encoder
    #     down1 = self.down1(inx)

    #     condition1 = self.mlp1(time_emb)
    #     b, c = condition1.shape
    #     condition1 = rearrange(condition1, 'b c -> b c 1 1')
    #     if adjust:
    #         gamma1, beta1 = self.adjust1(x_adjust)
    #         down1 = down1 + gamma1 * condition1 + beta1
    #     else:
    #         down1 = down1 + condition1
    #     conv1 = self.conv1(down1)
        
    #     down2 = self.down2(conv1)
        
    #     condition2 = self.mlp2(time_emb)
    #     b, c = condition2.shape
    #     condition2 = rearrange(condition2, 'b c -> b c 1 1')
    #     if adjust:
    #         gamma2, beta2 = self.adjust2(x_adjust)
    #         down2 = down2 + gamma2 * condition2 + beta2
    #     else:
    #         down2 = down2 + condition2
    #     conv2 = self.conv2(down2)

    #     # Decoder 
    #     up1 = self.up1(conv2, conv1)
        
    #     condition3 = self.mlp3(time_emb)
    #     b, c = condition3.shape
    #     condition3 = rearrange(condition3, 'b c -> b c 1 1')
    #     if adjust:
    #         gamma3, beta3 = self.adjust3(x_adjust)
    #         up1 = up1 + gamma3 * condition3 + beta3
    #     else:
    #         up1 = up1 + condition3
    #     conv3 = self.conv3(up1)

    #     up2 = self.up2(conv3, inx)
        
    #     condition4 = self.mlp4(time_emb)
    #     b, c = condition4.shape
    #     condition4 = rearrange(condition4, 'b c -> b c 1 1')
    #     if adjust:
    #         gamma4, beta4 = self.adjust4(x_adjust)
    #         up2 = up2 + gamma4 * condition4 + beta4
    #     else:
    #         up2 = up2 + condition4
    #     conv4 = self.conv4(up2)

    #     # Output conv
    #     out = self.outc(conv4)

    #     # residual_x connection with GELU
    #     # out = self.gelu(out + residual_x)
    #     out = self.gelu(out + residual_x)
    #     return out


class Network(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, context=True):
        super(Network, self).__init__()
        self.unet = UNet(in_channels=in_channels, out_channels=out_channels)
        self.context = context

    def forward(self, x, t, y, x_end, adjust=True):
        if self.context:
            x_middle = x[:, 1].unsqueeze(1)
        else:
            x_middle = x
        
        x_adjust = torch.cat((y, x_end), dim=1)
        out = self.unet(x, t, x_adjust, adjust=adjust) + x_middle

        return out


# WeightNet of the one-shot learning framework
class WeightNet(nn.Module):
    def __init__(self, weight_num=10):
        super(WeightNet, self).__init__()
        init = torch.ones([1, weight_num, 1, 1]) / weight_num
        self.weights = nn.Parameter(init)

    def forward(self, x):
        weights = F.softmax(self.weights, 1)
        out = weights * x
        out = out.sum(dim=1, keepdim=True)

        return out, weights
