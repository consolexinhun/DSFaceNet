import torch
from torch import nn
from torchsummary import summary
from config import *
import math
import torch.nn.functional as F
from torch.nn import Parameter
'''
SE block
'''
class SqueezeAndExcite(nn.Module):
    def __init__(self, in_channels, out_channels, divide=16):
        super(SqueezeAndExcite, self).__init__()
        mid_channels = in_channels // divide
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.SEblock = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=mid_channels),
            nn.ReLU6(inplace=False),
            nn.Linear(in_features=mid_channels, out_features=out_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        out = self.pool(x)
        out = out.view(b, -1)
        out = self.SEblock(out)
        out = out.view(b, c, 1, 1)
        return out * x

class Bottleneck(nn.Module):
    def __init__(self, inp, oup, stride, expansion):
        super(Bottleneck, self).__init__()
        self.connect = stride == 1 and inp == oup
        #
        self.conv = nn.Sequential(
            #pw
            nn.Conv2d(inp, inp * expansion, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),
            # nn.ReLU(inplace=True),

            #dw
            nn.Conv2d(inp * expansion, inp * expansion, 3, stride, 1, groups=inp * expansion, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),
            # nn.ReLU(inplace=True),

            #pw-linear
            nn.Conv2d(inp * expansion, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

        self.se = SqueezeAndExcite(inp, oup) # 只有inp = oup时才能够用SE block
        self.prelu = nn.PReLU()

    def forward(self, x):
        if self.connect:
            out = self.conv(x)
            out = self.se(out)
            out = out + x
            return self.prelu(out)
            # return x + self.conv(x)
        else:
            return self.conv(x)

class ConvBlock(nn.Module):
    def __init__(self, inp, oup, k, s, p, dw=False, linear=False):
        super(ConvBlock, self).__init__()
        self.linear = linear
        if dw:
            self.conv = nn.Conv2d(inp, oup, k, s, p, groups=inp, bias=False)
        else:
            self.conv = nn.Conv2d(inp, oup, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        if not linear:
            self.prelu = nn.PReLU(oup)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.linear:
            return x
        else:
            return self.prelu(x)

Mobilefacenet_bottleneck_setting = [
    # t, c , n ,s
    [2, 64, 5, 2],
    [4, 128, 1, 2],
    [2, 128, 6, 1],
    [4, 128, 1, 2],
    [2, 128, 2, 1]
]

class DSFaceNet(nn.Module):
    def __init__(self, bottleneck_setting=Mobilefacenet_bottleneck_setting):
        super(DSFaceNet, self).__init__()

        self.conv1 = ConvBlock(3, 64, 3, 2, 1)

        self.dw_conv1 = ConvBlock(64, 64, 3, 1, 1, dw=True)

        self.inplanes = 64
        block = Bottleneck
        self.blocks = self._make_layer(block, bottleneck_setting)

        self.conv2 = ConvBlock(128, 512, 1, 1, 0)

        self.linear7 = ConvBlock(512, 512, (7, 6), 1, 0, dw=True, linear=True)

        self.linear1 = ConvBlock(512, 128, 1, 1, 0, linear=True)

        # 模型参数初始化的技巧
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, setting):
        layers = []
        for t, c, n, s in setting:
            for i in range(n):
                if i == 0:
                    layers.append(block(self.inplanes, c, s, t))
                else:
                    layers.append(block(self.inplanes, c, 1, t))
                self.inplanes = c

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dw_conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)
        x = self.linear7(x)
        x = self.linear1(x)
        x = x.view(x.size(0), -1)

        return x



class ArcMarginProduct(nn.Module):
    def __init__(self, in_features=128, out_features=200, s=32.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m

        # 将不可训练的Tensor类型转变为可训练的Parameter，它能让变量在学习的过程中不断优化达到最优值
        # linear 的weight 和bias就是parameter类型，并且不能用tensor类型替换。与torch.tensor([1,2,3],requires_grad=True)的区别，这个只是将参数变成可训练的，并没有绑定在module的parameter列表中。
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)  # 初始化参数的方式
        # init.kaiming_uniform_()
        # self.weight.data.normal_(std=0.001)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label):
        # cosine = x * w = cos(theta)
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        # sin(theta) = 根号(1- cos^2)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        # cos(theta + m) = cos(theta) * cos(m) - sin(theta) * sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            # 如果cos(theta)>0，那么两个向量是相似的，选择phi，否则选择cosine
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # if theta + m > 180 ,使用cosface的计算方式
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size())  # one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot = one_hot.to(DEVICE)  # 这里必须要搬到CUDA上才能计算

        one_hot.scatter_(1, label.view(-1, 1).long(), 1)  # 这里是为了得到one-hot编码
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output

if __name__ == '__main__':
    model = DSFaceNet()

    device = DEVICE
    model = model.to(device)

    summary(model, (3, 112, 96)) # 必须开cuda，不需要传入batch_size






# input = torch.randn(1, 3, 112, 112)
# out = model(input)
# print(out.shape)