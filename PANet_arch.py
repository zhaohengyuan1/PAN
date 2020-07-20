import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import arch_util as arch_util

class PA(nn.Module):
    '''PA is pixel attention'''
    def __init__(self, nf):

        super(PA, self).__init__()
        self.conv = nn.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        y = self.conv(x)
        y = self.sigmoid(y)
        out = torch.mul(x, y)

        return out
    
class PAConv(nn.Module):
    def __init__(self, nf, k_size=3):

        super(PAConv, self).__init__()
        self.k2 = nn.Conv2d(nf, nf, 1) # 1x1 convolution nf->nf
        self.sigmoid = nn.Sigmoid()
        self.k3 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) # 3x3 convolution
        self.k4 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) # 3x3 convolution

    def forward(self, x):

        y = self.k2(x)
        y = self.sigmoid(y)

        out = torch.mul(self.k3(x), y)
        out = self.k4(out)

        return out
    
class SCPA(nn.Module):
    
    """SCPA is modified from SCNet (Jiang-Jiang Liu et al. Improving Convolutional Networks with Self-Calibrated Convolutions. In CVPR, 2020)
        Github: https://github.com/MCG-NKU/SCNet
    """

    def __init__(self, nf, reduction=2, stride=1, dilation=1):
        super(SCPA, self).__init__()
        group_width = nf // reduction
        
        self.conv1_a = nn.Conv2d(nf, group_width, kernel_size=1, bias=False)
        self.conv1_b = nn.Conv2d(nf, group_width, kernel_size=1, bias=False)
        
        self.k1 = nn.Sequential(
                    nn.Conv2d(
                        group_width, group_width, kernel_size=3, stride=stride,
                        padding=dilation, dilation=dilation,
                        bias=False)
                    )
        
        self.PAConv = PAConv(group_width)
        
        self.conv3 = nn.Conv2d(
            group_width * reduction, nf, kernel_size=1, bias=False)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        residual = x

        out_a= self.conv1_a(x)
        out_b = self.conv1_b(x)
        out_a = self.lrelu(out_a)
        out_b = self.lrelu(out_b)

        out_a = self.k1(out_a)
        out_b = self.PAConv(out_b)
        out_a = self.lrelu(out_a)
        out_b = self.lrelu(out_b)

        out = self.conv3(torch.cat([out_a, out_b], dim=1))
        out += residual

        return out
    
class PANet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb):
        super(PANet, self).__init__()
        # SCPA
        SCPA_block_f = functools.partial(SCPA, nf=nf, reduction=2)
        
        ### first convolution
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        
        ### main blocks
        self.SCPA_trunk = arch_util.make_layer(SCPA_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, 24, 3, 1, 1, bias=True)
        self.att1 = PA(24)
        self.HRconv1 = nn.Conv2d(24, 24, 3, 1, 1, bias=True)
        
        self.upconv2 = nn.Conv2d(24, 24, 3, 1, 1, bias=True)
        self.att2 = PA(24)
        self.HRconv2 = nn.Conv2d(24, 24, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(24, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.SCPA_trunk(fea))
        fea = fea + trunk
        
        fea = self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest'))
        fea = self.lrelu(self.att1(fea))
        fea = self.lrelu(self.HRconv1(fea))
        fea = self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest'))
        fea = self.lrelu(self.att2(fea))
        fea = self.lrelu(self.HRconv2(fea))
        
        out = self.conv_last(fea)
        
        ILR = F.interpolate(x, scale_factor=4, mode='bilinear')
        out = out + ILR
        return out