"""
Channel and Spatial CSNet Network (CS-Net).
"""
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from laddernet.RSAN.DropBlack import DropBlock2D
from laddernet.networks.CBAM import ChannelAttention, SpatialAttention


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        if in_planes >= 16:
            ratio = 16
        else:
            ratio = 8

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttention(nn.Module):
    def __init__(self, out_channels, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class RCBAM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RCBAM, self).__init__()
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention(out_channels)
        self.conv1x1 = nn.Conv2d(3 * out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.ca(x) * x
        x2 = self.sa(x1) * x1
        return x2

class RCSE_M(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=8):
        super(RCSE_M, self).__init__()
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention(out_channels)
        self.conv1x1 = nn.Conv2d(3 * out_channels, out_channels, kernel_size=1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(out_channels, out_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels // reduction, out_channels, bias=False),
            nn.Sigmoid()
        )

        self.conv1x1 = nn.Conv2d(2 * out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.ca(x) * x
        x2 = self.sa(x1) * x1

        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        out1 = x * y.expand_as(x)
        out2 = x2

        out3 = out1 + out2

        return out3



class FilterResponseNormNd(nn.Module):

    def __init__(self, ndim, num_features, eps=1e-6,
                 learnable_eps=False):
        """
        Input Variables:
        ----------------
            ndim: An integer indicating the number of dimensions of the expected input tensor.
            num_features: An integer indicating the number of input feature dimensions.
            eps: A scalar constant or learnable variable.
            learnable_eps: A bool value indicating whether the eps is learnable.
        """
        assert ndim in [3, 4, 5], \
            'FilterResponseNorm only supports 3d, 4d or 5d inputs.'
        super(FilterResponseNormNd, self).__init__()
        shape = (1, num_features) + (1,) * (ndim - 2)
        self.eps = nn.Parameter(torch.ones(*shape) * eps)
        if not learnable_eps:
            self.eps.requires_grad_(False)
        # self.gamma = nn.Parameter(torch.Tensor(*shape))
        # self.beta = nn.Parameter(torch.Tensor(*shape))
        # self.tau = nn.Parameter(torch.Tensor(*shape))
        # self.reset_parameters()

    def forward(self, x):
        avg_dims = tuple(range(2, x.dim()))  # (2, 3)
        nu2 = torch.pow(x, 2).mean(dim=avg_dims, keepdim=True)
        x = x * torch.rsqrt(nu2 + torch.abs(self.eps))
        return x

    # def reset_parameters(self):
    #     nn.init.ones_(self.gamma)
    #     nn.init.zeros_(self.beta)
    #     nn.init.zeros_(self.tau)


class Mish_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.tanh(F.softplus(i))
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]

        v = 1. + i.exp()
        h = v.log()
        grad_gh = 1. / h.cosh().pow_(2)

        # Note that grad_hv * grad_vx = sigmoid(x)
        # grad_hv = 1./v
        # grad_vx = i.exp()

        grad_hx = i.sigmoid()

        grad_gx = grad_gh * grad_hx  # grad_hv * grad_vx

        grad_f = torch.tanh(F.softplus(i)) + i * grad_gx

        return grad_output * grad_f


class Mish(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        pass

    def forward(self, input_tensor):
        return Mish_func.apply(input_tensor)


def downsample():
    return nn.MaxPool2d(kernel_size=2, stride=2)


# class downsample(nn.Module):
#     def __init__(self, in_channels):
#         super(downsample, self).__init__()
#         self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
#     def forward(self, x):
#         out = self.conv(x)
#         return out


def deconv(in_channels, out_channels):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)


class ResEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResEncoder, self).__init__()
        self.conv = nn.Sequential(
            FilterResponseNormNd(4, in_channels),
            Mish(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            FilterResponseNormNd(4, out_channels),
            Mish(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Dropout(0.2),
            FilterResponseNormNd(4, out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
        )

        self.conv0 = nn.Sequential(
            FilterResponseNormNd(4, out_channels),
            Mish(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            FilterResponseNormNd(4, out_channels),
            Mish(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Dropout(0.2),
            FilterResponseNormNd(4, out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
        )
        self.conv1 = nn.Sequential(
            FilterResponseNormNd(4, out_channels),
            Mish(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            FilterResponseNormNd(4, out_channels),
            Mish(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Dropout(0.2),
            FilterResponseNormNd(4, out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
        )

        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.relu = Mish()

    def forward(self, x):
        residual = self.conv1x1(x)
        out1 = self.conv(x)
        out2 = self.conv0(out1)
        out3 = self.conv1(out2)
        out = out3 + residual
        out = self.relu(out)
        return out


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.conv = nn.Sequential(
            FilterResponseNormNd(4, in_channels),
            Mish(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            FilterResponseNormNd(4, out_channels),
            Mish(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Dropout(0.2),
            FilterResponseNormNd(4, out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
        )

        self.conv0 = nn.Sequential(
            FilterResponseNormNd(4, out_channels),
            Mish(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            FilterResponseNormNd(4, out_channels),
            Mish(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Dropout(0.2),
            FilterResponseNormNd(4, out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
        )

        self.conv1 = nn.Sequential(
            FilterResponseNormNd(4, out_channels),
            Mish(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            FilterResponseNormNd(4, out_channels),
            Mish(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Dropout(0.2),
            FilterResponseNormNd(4, out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
        )

    def forward(self, x):
        out = self.conv(x)
        out = self.conv0(out)
        out = self.conv1(out)
        return out


class ResEncoder1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResEncoder1, self).__init__()
        self.conv = nn.Sequential(
            FilterResponseNormNd(4, in_channels),
            Mish(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            FilterResponseNormNd(4, out_channels),
            Mish(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Dropout(0.2),
            FilterResponseNormNd(4, out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
        )

        self.conv0 = nn.Sequential(
            FilterResponseNormNd(4, out_channels),
            Mish(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            FilterResponseNormNd(4, out_channels),
            Mish(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Dropout(0.2),
            FilterResponseNormNd(4, out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
        )
        self.conv1 = nn.Sequential(
            FilterResponseNormNd(4, out_channels),
            Mish(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            FilterResponseNormNd(4, out_channels),
            Mish(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Dropout(0.2),
            FilterResponseNormNd(4, out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
        )

        self.conv2 = nn.Sequential(
            FilterResponseNormNd(4, out_channels),
            Mish(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            FilterResponseNormNd(4, out_channels),
            Mish(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Dropout(0.2),
            FilterResponseNormNd(4, out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
        )

        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.relu = Mish()
        self.F_M = nn.Sequential(
            FilterResponseNormNd(4, out_channels),
            Mish(),
        )

    def forward(self, x):
        residual = self.conv1x1(x)
        out1 = self.conv(x)
        out2 = self.conv0(out1)
        out3 = self.conv1(out2)
        out4 = self.conv2(out3)
        out = out4 + residual
        out = self.relu(out)
        return out


class Decoder1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder1, self).__init__()
        self.conv = nn.Sequential(
            FilterResponseNormNd(4, in_channels),
            Mish(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            FilterResponseNormNd(4, out_channels),
            Mish(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Dropout(0.2),
            FilterResponseNormNd(4, out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
        )

        self.conv0 = nn.Sequential(
            FilterResponseNormNd(4, out_channels),
            Mish(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            FilterResponseNormNd(4, out_channels),
            Mish(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Dropout(0.2),
            FilterResponseNormNd(4, out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
        )

        self.conv1 = nn.Sequential(
            FilterResponseNormNd(4, out_channels),
            Mish(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            FilterResponseNormNd(4, out_channels),
            Mish(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Dropout(0.2),
            FilterResponseNormNd(4, out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
        )
        self.conv2 = nn.Sequential(
            FilterResponseNormNd(4, out_channels),
            Mish(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            FilterResponseNormNd(4, out_channels),
            Mish(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Dropout(0.2),
            FilterResponseNormNd(4, out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
        )

    def forward(self, x):
        out = self.conv(x)
        out = self.conv0(out)
        out = self.conv1(out)
        out = self.conv2(out)
        return out


class SpatialAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttentionBlock, self).__init__()
        self.query = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 16, kernel_size=(1, 3), padding=(0, 1)),
            FilterResponseNormNd(4, in_channels // 16),
            Mish(),
        )
        self.key = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 16, kernel_size=(3, 1), padding=(1, 0)),
            FilterResponseNormNd(4, in_channels // 16),
            Mish(),
        )
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        :param x: input( BxCxHxW )
        :return: affinity value + x
        """
        B, C, H, W = x.size()
        # compress x: [B,C,H,W]-->[B,H*W,C], make a matrix transpose
        proj_query = self.query(x).view(B, -1, W * H).permute(0, 2, 1)
        proj_key = self.key(x).view(B, -1, W * H)
        affinity = torch.matmul(proj_query, proj_key)
        affinity = self.softmax(affinity)
        proj_value = self.value(x).view(B, -1, H * W)
        weights = torch.matmul(proj_value, affinity.permute(0, 2, 1))
        weights = weights.view(B, C, H, W)
        out = self.gamma * weights + x
        return out


class ChannelAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttentionBlock, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        :param x: input( BxCxHxW )
        :return: affinity value + x
        """
        B, C, H, W = x.size()
        proj_query = x.view(B, C, -1)
        proj_key = x.view(B, C, -1).permute(0, 2, 1)
        affinity = torch.matmul(proj_query, proj_key)
        affinity_new = torch.max(affinity, -1, keepdim=True)[0].expand_as(affinity) - affinity
        affinity_new = self.softmax(affinity_new)
        proj_value = x.view(B, C, -1)
        weights = torch.matmul(affinity_new, proj_value)
        weights = weights.view(B, C, H, W)
        out = self.gamma * weights + x
        return out


class AffinityAttention(nn.Module):
    """ Affinity attention module """

    def __init__(self, in_channels):
        super(AffinityAttention, self).__init__()
        self.sab = SpatialAttentionBlock(in_channels)
        self.cab = ChannelAttentionBlock(in_channels)
        # self.conv1x1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

    def forward(self, x):
        """
        sab: spatial attention block
        cab: channel att ention block
        :param x: input tensor
        :return: sab + cab
        """
        sab = self.sab(x)
        cab = self.cab(x)
        out = sab + cab
        return out


class BA_CS_FRN_MISH_CCC_MSS_3CS_4NetV1(nn.Module):
    def __init__(self, channels, classes, start_neurons=16):
        """
        :param classes: the object classes number.
        :param channels: the channels of the input image.
        """
        super(BA_CS_FRN_MISH_CCC_MSS_3CS_4NetV1, self).__init__()
        self.encoder0 = ResEncoder(channels, start_neurons * 2)
        self.encoder1 = ResEncoder(start_neurons * 2, start_neurons * 4)
        self.encoder2 = ResEncoder1(start_neurons * 4, start_neurons * 8)
        self.encoder3 = ResEncoder1(start_neurons * 8, start_neurons * 16)
        self.encoder4 = ResEncoder1(start_neurons * 16, start_neurons * 32)
        self.downsample = downsample()

        self.affinity_attention = AffinityAttention(start_neurons * 32)
        self.decoder4 = Decoder1(start_neurons * 32, start_neurons * 16)
        self.decoder3 = Decoder1(start_neurons * 16, start_neurons * 8)
        self.decoder2 = Decoder(start_neurons * 8, start_neurons * 4)
        self.decoder1 = Decoder(start_neurons * 4, start_neurons * 2)
        self.deconv4 = deconv(start_neurons * 32, start_neurons * 16)
        self.deconv3 = deconv(start_neurons * 16, start_neurons * 8)
        self.deconv2 = deconv(start_neurons * 8, start_neurons * 4)
        self.deconv1 = deconv(start_neurons * 4, start_neurons * 2)
        self.final = nn.Conv2d(start_neurons * 2, classes, kernel_size=1)

        self.resmlpd1 = RCSE_M(start_neurons * 2, start_neurons * 2)
        self.resmlpd2 = RCSE_M(start_neurons * 4, start_neurons * 4)
        self.resmlpd3 = RCSE_M(start_neurons * 8, start_neurons * 8)
        self.resmlpd4 = RCSE_M(start_neurons * 16, start_neurons * 16)
        #
        self.conv1x1_1 = nn.Conv2d(start_neurons * 4, start_neurons * 2, kernel_size=1)
        self.conv1x1_2 = nn.Conv2d(start_neurons * 8, start_neurons * 4, kernel_size=1)
        self.conv1x1_3 = nn.Conv2d(start_neurons * 16, start_neurons * 8, kernel_size=1)
        self.conv1x1_4 = nn.Conv2d(start_neurons * 32, start_neurons * 16, kernel_size=1)


        self.convUxU_1 = nn.Conv2d(40, 512, kernel_size=1)
        self.convUxU_2 = nn.Conv2d(80, 512, kernel_size=1)
        self.convUxU_3 = nn.Conv2d(160, 512, kernel_size=1)
        self.convUxU_4 = nn.Conv2d(1024, 512, kernel_size=1)


        self.convCxC_1 = nn.Conv2d(128, 32, kernel_size=1)
        self.convCxC_2 = nn.Conv2d(256, 64, kernel_size=1)
        self.convCxC_3 = nn.Conv2d(512, 128, kernel_size=1)


        self.deconv1_1 = deconv(start_neurons * 2, start_neurons * 2)
        self.deconv2_1 = deconv(start_neurons * 4, start_neurons * 4)
        self.deconv3_1 = deconv(start_neurons * 8, start_neurons * 8)


    def forward(self, x, R1, R2, R3):
        enc_input = self.encoder0(x)
        down0 = self.downsample(enc_input)

        enc1 = self.encoder1(down0)
        down2 = self.downsample(enc1)

        enc2 = self.encoder2(down2)
        down3 = self.downsample(enc2)

        enc3 = self.encoder3(down3)
        enc_input4 = self.resmlpd4(enc3)
        enc_input4 = torch.cat([enc_input4, enc3], dim=1)
        enc_input4 = self.conv1x1_4(enc_input4)
        down4 = self.downsample(enc3)

        input_feature = self.encoder4(down4)

        UU_1 = torch.cat([R1, enc_input],dim=1)
        UU_1 = self.downsample(UU_1)
        UU_1 = self.downsample(UU_1)
        UU_1 = self.downsample(UU_1)
        UU_1 = self.downsample(UU_1)
        UU_1 = self.convUxU_1(UU_1)

        UU_2 = torch.cat([R2, enc1], dim=1)
        UU_2 = self.downsample(UU_2)
        UU_2 = self.downsample(UU_2)
        UU_2 = self.downsample(UU_2)
        UU_2 = self.convUxU_2(UU_2)

        UU_3 = torch.cat([R3, enc2], dim=1)
        UU_3 = self.downsample(UU_3)
        UU_3 = self.downsample(UU_3)
        UU_3 = self.convUxU_3(UU_3)

        UU_123 = UU_1 + UU_2 + UU_3

        UU_4 = torch.cat([UU_123, input_feature], dim=1)
        UU_4 = self.convUxU_4(UU_4)


        attention = self.affinity_attention(UU_4)
        attention_fuse = UU_4 + attention

        CC_1 = self.convCxC_1(enc2)
        CC_1 = self.deconv1_1(CC_1)
        CC_1 = self.deconv1_1(CC_1)

        CC_2 = self.convCxC_2(enc3)
        CC_2 = self.deconv2_1(CC_2)
        CC_2 = self.deconv2_1(CC_2)

        CC_3 = self.convCxC_3(input_feature)
        CC_3 = self.deconv3_1(CC_3)
        CC_3 = self.deconv3_1(CC_3)

        enc_input1 = torch.cat([CC_1, enc_input], dim=1)
        enc_input1 = self.conv1x1_1(enc_input1)
        enc_input10 = self.resmlpd1(enc_input1)
        enc_input1 = torch.cat([enc_input10, enc_input], dim=1)
        enc_input1 = self.conv1x1_1(enc_input1)


        enc_input2 = torch.cat([CC_2, enc1], dim=1)
        enc_input2 = self.conv1x1_2(enc_input2)
        enc_input20 = self.resmlpd2(enc_input2)
        enc_input2 = torch.cat([enc_input20, enc1], dim=1)
        enc_input2 = self.conv1x1_2(enc_input2)


        enc_input3 = torch.cat([CC_3, enc2], dim=1)
        enc_input3 = self.conv1x1_3(enc_input3)
        enc_input30 = self.resmlpd3(enc_input3)
        enc_input3 = torch.cat([enc_input30, enc2], dim=1)
        enc_input3 = self.conv1x1_3(enc_input3)



        attention_fuse1 = self.deconv4(attention_fuse)
        attention_fuse1 = self.deconv3(attention_fuse1)
        attention_fuse1 = self.deconv2(attention_fuse1)
        attention_fuse1 = self.deconv1(attention_fuse1)
        attention_fuse1 = self.final(attention_fuse1)
        attention_fuse1 = nn.Sigmoid()(attention_fuse1)

        up4 = self.deconv4(attention_fuse)
        up4 = torch.cat([enc_input4, up4], dim=1)
        dec4 = self.decoder4(up4)
        dec41 = self.deconv3(dec4)
        dec41 = self.deconv2(dec41)
        dec41 = self.deconv1(dec41)
        dec41 = self.final(dec41)
        dec41 = nn.Sigmoid()(dec41)

        up3 = self.deconv3(dec4)
        up3 = torch.cat([enc_input3, up3], dim=1)
        dec3 = self.decoder3(up3)
        dec31 = self.deconv2(dec3)
        dec31 = self.deconv1(dec31)
        dec31 = self.final(dec31)
        dec31 = nn.Sigmoid()(dec31)

        up2 = self.deconv2(dec3)
        up2 = torch.cat([enc_input2, up2], dim=1)
        dec2 = self.decoder2(up2)
        dec21 = self.deconv1(dec2)
        dec21 = self.final(dec21)
        dec21 = nn.Sigmoid()(dec21)

        up1 = self.deconv1(dec2)
        up1 = torch.cat([enc_input1, up1], dim=1)
        dec1 = self.decoder1(up1)

        dec1 = self.final(dec1)
        dec1 = nn.Sigmoid()(dec1)

        final = attention_fuse1 * 0.1 + dec41 * 0.1 + dec31 * 0.1 + dec21 * 0.2 + dec1 * 0.5

        return final


class BA_CS_FRN_MISH_CCC_MSS_3CS_4NetV2(nn.Module):
    def __init__(self, channels, classes, start_neurons=4):
        """
        :param classes: the object classes number.
        :param channels: the channels of the input image.
        """
        super(BA_CS_FRN_MISH_CCC_MSS_3CS_4NetV2, self).__init__()
        self.encoder0 = ResEncoder1(channels, start_neurons * 2)
        self.encoder1 = ResEncoder1(start_neurons * 2, start_neurons * 4)
        self.encoder2 = ResEncoder1(start_neurons * 4, start_neurons * 8)
        self.downsample = downsample()
        self.decoder2 = Decoder1(start_neurons * 8, start_neurons * 4)
        self.decoder1 = Decoder1(start_neurons * 4, start_neurons * 2)
        self.deconv2 = deconv(start_neurons * 8, start_neurons * 4)
        self.deconv1 = deconv(start_neurons * 4, start_neurons * 2)
        self.final = nn.Conv2d(start_neurons * 2, classes, kernel_size=1)

        self.resmlpd1 = RCSE_M(start_neurons * 2, start_neurons * 2)
        self.resmlpd2 = RCSE_M(start_neurons * 4, start_neurons * 4)

        self.conv1x1_1 = nn.Conv2d(start_neurons * 4, start_neurons * 2, kernel_size=1)
        self.conv1x1_2 = nn.Conv2d(start_neurons * 8, start_neurons * 4, kernel_size=1)

    def forward(self, x):
        enc_input = self.encoder0(x)
        enc_input1 = self.resmlpd1(enc_input)
        enc_input1 = torch.cat([enc_input1, enc_input], dim=1)
        enc_input1 = self.conv1x1_1(enc_input1)
        down0 = self.downsample(enc_input)

        enc1 = self.encoder1(down0)
        enc_input2 = self.resmlpd2(enc1)
        enc_input2 = torch.cat([enc_input2, enc1], dim=1)
        enc_input2 = self.conv1x1_2(enc_input2)
        down2 = self.downsample(enc1)

        enc2 = self.encoder2(down2)

        up2 = self.deconv2(enc2)
        up2 = torch.cat([enc_input2, up2], dim=1)
        dec2 = self.decoder2(up2)
        dec21 = self.deconv1(dec2)
        dec21 = self.final(dec21)
        dec21 = nn.Sigmoid()(dec21)

        up1 = self.deconv1(dec2)
        up1 = torch.cat([enc_input1, up1], dim=1)
        dec1 = self.decoder1(up1)

        dec1 = self.final(dec1)
        dec1 = nn.Sigmoid()(dec1)

        final = dec1 * 0.6 + dec21 * 0.4

        return final, enc_input, enc1, enc2


class BA_CS_FRN_MISH_CCC_MSS_3CS_4NetVUU6_5(nn.Module):
    def __init__(self, channels, classes, start_neurons=4):
        """
        :param classes: the object classes number.
        :param channels: the channels of the input image.
        """
        super(BA_CS_FRN_MISH_CCC_MSS_3CS_4NetVUU6_5, self).__init__()
        self.v1 = BA_CS_FRN_MISH_CCC_MSS_3CS_4NetV1(channels, classes)
        self.v2 = BA_CS_FRN_MISH_CCC_MSS_3CS_4NetV2(channels, classes)

    def forward(self, x):
        v2, R1, R2, R3 = self.v2(x)
        v1 = self.v1(x, R1, R2, R3)
        v = v1 * 0.7 + v2 * 0.3

        return v, v1, v2


if __name__ == '__main__':
    net = BA_CS_FRN_MISH_CCC_MSS_3CS_4NetVUU6_5(1, 1)
    data_arr = torch.rand(size=(10, 1, 64, 64))
    outputs, outputs1, outputs2 = net(data_arr)
    print(outputs.size())

