import torch
from torch import nn
from torchsummary import summary
from config import *
import math
import torch.nn.functional as F
from torch.nn import Parameter
'''
求Input的二范数，为其输入除以其模长
角度蒸馏Loss需要用到
'''
def l2_norm(input, axis=1):
    norm  = torch.norm(input, axis, keepdim=True) # 默认p=2
    output = torch.div(input, norm)
    return output


'''
变组卷积，S表示每个通道的channel数量
'''
def VarGConv(in_channels, out_channels, kernel_size, stride, S):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2, groups=in_channels//S, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.PReLU()
    )

'''
pointwise卷积，这里的kernelsize都是1，不过这里也要分组吗？？
'''
def PointConv(in_channels, out_channels, stride, S, isPReLU):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, stride, padding=0, groups=in_channels//S, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.PReLU() if isPReLU else nn.Sequential()
    )

'''
SE block
'''
class SqueezeAndExcite(nn.Module):
    def __init__(self, in_channels, out_channels, divide=4):
        super(SqueezeAndExcite, self).__init__()
        mid_channels = in_channels // divide
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.SEblock = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=mid_channels),
            # nn.ReLU6(inplace=True),
            nn.ReLU6(inplace=False),
            nn.Linear(in_features=mid_channels, out_features=out_channels),
            # nn.ReLU6(inplace=True), # 其实这里应该是sigmoid的
            nn.ReLU6(inplace=False)
        )

    def forward(self, x):
        b, c, h, w = x.size()
        out = self.pool(x)
        out = out.view(b, -1)
        out = self.SEblock(out)
        out = out.view(b, c, 1, 1)
        return out * x

'''
normal block
'''
class NormalBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, stride=1, S=8):
        super(NormalBlock, self).__init__()
        out_channels = 2 * in_channels
        self.vargconv1 = VarGConv(in_channels, out_channels, kernel_size, stride, S)
        self.pointconv1 = PointConv(out_channels, in_channels, stride, S, isPReLU=True)

        self.vargconv2 = VarGConv(in_channels, out_channels, kernel_size, stride, S)
        self.pointconv2 = PointConv(out_channels, in_channels, stride, S, isPReLU=False)

        self.se = SqueezeAndExcite(in_channels, in_channels)
        self.prelu = nn.PReLU()

    def forward(self, x):
        out = x
        x = self.pointconv1(self.vargconv1(x))
        x = self.pointconv2(self.vargconv2(x))
        x = self.se(x)
        # out += x
        out = out + x
        return self.prelu(out)

'''
downsampling block
'''

class DownSampling(nn.Module):
    def __init__(self, in_channels, kernel_size, stride=2, S=8):
        super(DownSampling, self).__init__()
        out_channels = 2 * in_channels


        self.branch1 = nn.Sequential(
            VarGConv(in_channels, out_channels, kernel_size, stride, S),
            PointConv(out_channels, out_channels, 1, S, isPReLU=True)
        )

        self.branch2 = nn.Sequential(
            VarGConv(in_channels, out_channels, kernel_size, stride, S),
            PointConv(out_channels, out_channels, 1, S, isPReLU=True)
        )

        self.block3 = nn.Sequential(
            VarGConv(out_channels, 2*out_channels, kernel_size, 1, S), # stride =1
            PointConv(2*out_channels, out_channels, 1, S, isPReLU=False)
        ) # 上面那个分支

        self.shortcut = nn.Sequential(
            VarGConv(in_channels, out_channels, kernel_size, stride, S),
            PointConv(out_channels, out_channels, 1, S, isPReLU=False)
        )

        self.prelu = nn.PReLU()

    def forward(self, x):
        out = self.shortcut(x)

        x1 = x2 = x
        x1 = self.branch1(x1)
        x2 = self.branch2(x2)
        x3 = x1+x2
        x3 = self.block3(x3)

        # out += x3
        out = out + x3
        return self.prelu(out)


class HeadSetting(nn.Module):
    def __init__(self, in_channels, kernel_size, S=8):
        super(HeadSetting, self).__init__()
        self.block = nn.Sequential(
            VarGConv(in_channels, in_channels, kernel_size, 2, S),
            PointConv(in_channels, in_channels, 1, S, isPReLU=True),
            VarGConv(in_channels, in_channels, kernel_size, 1, S),
            PointConv(in_channels, in_channels, 1, S, isPReLU=False)
        )

        self.short = nn.Sequential(
            VarGConv(in_channels, in_channels, kernel_size, 2, S),
            PointConv(in_channels, in_channels, 1, S, isPReLU=False),
        )

    def forward(self, x):
        out = self.short(x)
        x = self.block(x)
        # out += x
        out = out + x
        return out


class Embedding(nn.Module):
    def __init__(self, in_channels, out_channels=512, S=8):
        super(Embedding, self).__init__()
        self.embedding = nn.Sequential(
            nn.Conv2d(in_channels, 1024, kernel_size=1, stride=1,padding=0, bias=False),
            nn.BatchNorm2d(1024),
            # nn.ReLU6(inplace=True),
            nn.ReLU6(inplace=False),
            nn.Conv2d(1024, 1024, (7, 6), 1, padding=0, groups=1024//8, bias=False),
            nn.Conv2d(1024, 512, 1, 1, padding=0, groups=512, bias=False)
        )

        self.fc = nn.Linear(in_features=512, out_features=out_channels)



    def forward(self, x):
        x = self.embedding(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out

class VarGFaceNet(nn.Module):
    def __init__(self, num_classes=512):
        super(VarGFaceNet, self).__init__()
        S=8
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=40, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(40),
            # nn.ReLU6(inplace=True)
            nn.ReLU6(inplace=False)
        )
        self.head = HeadSetting(40, 3)
        self.stage2 = nn.Sequential( # 1 normal 2 down
            DownSampling(40, 3, 2),
            NormalBlock(80, 3, 1),
            NormalBlock(80, 3, 1)
        )

        self.stage3 = nn.Sequential(
            DownSampling(80, 3, 2),
            NormalBlock(160, 3, 1),
            NormalBlock(160, 3, 1),
            NormalBlock(160, 3, 1),
            NormalBlock(160, 3, 1),
            NormalBlock(160, 3, 1),
            NormalBlock(160, 3, 1),
        )

        self.stage4 = nn.Sequential(
            DownSampling(160, 3, 2),
            NormalBlock(320, 3, 1),
            NormalBlock(320, 3, 1),
            NormalBlock(320, 3, 1),
        )

        self.embedding = Embedding(320, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.head(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        out = self.embedding(x)
        return out


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
    model = VarGFaceNet(128)
    # input = torch.randn(1, 3, 112, 112)
    # out = model(input)
    # print(out.shape)

    device = DEVICE
    model = model.to(device)

    summary(model, (3, 112, 96)) # 必须开cuda，不需要传入batch_size
