import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .enhancer import Enhancer

def conv3x3(in_c, out_c, stride=1, padding=0, bias=False):
    """
    3x3 convolution with padding
    """
    return nn.Conv2d(in_c, out_c, stride=stride, kernel_size=3, padding=padding, bias = bias)

class BasicBlock(nn.Module):

    expansion = 1
    def __init__(self, inplanes, outplanes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv0 = conv3x3(inplanes, outplanes, stride=1, padding=1)
        self.bn0 = nn.BatchNorm2d(outplanes)
        self.conv1 = conv3x3(outplanes, outplanes, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.conv2 = conv3x3(outplanes, outplanes, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(outplanes)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.conv0(x)
        out = self.bn0(out)
        out = self.relu(out)

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        out = out+residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):

    def __init__(self, blocktype, layers, N=5, K=1, args=None):
        super(ResNet, self).__init__()
        dims = [64, 128, 256, 512]
        self.epos = args.epos
        self.inplanes = 64
        self.conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(64)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.layer0 = self._make_layer(blocktype, 64, layers[0], stride=2)
        self.layer1 = self._make_layer(blocktype, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(blocktype, 256, layers[2], stride=2)
        self.layer3 = self._make_layer(blocktype, 512, layers[3], stride=1)
        self.N = N
        self.K = K
        self.use_enhancer=args.use_enhancer
        if self.use_enhancer:
            self.enhancer = Enhancer(dims[self.epos], dims[self.epos], self.N, self.K, args)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, blocktype, planes, num_block, stride=1):
        downsample=None
        if stride != 1 or self.inplanes!=planes*blocktype.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes*blocktype.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes*blocktype.expansion)
            )
        layers = []
        layers.append(blocktype(self.inplanes, planes, stride, downsample)) # in each layer, only the first block change the feature map size
        self.inplanes=planes*blocktype.expansion

        for i in range(1, num_block):
            layers.append(blocktype(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        i = 0
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu(x)

        x = self.layer0(x)
        if self.use_enhancer and i==self.epos:
            x = self.enhancer(x)

        i+=1
        x = self.layer1(x)
        if self.use_enhancer and i==self.epos:
            x = self.enhancer(x)

        i+=1
        x = self.layer2(x)
        if self.use_enhancer and i==self.epos:
            x = self.enhancer(x)

        i+=1
        x = self.layer3(x)
        if self.use_enhancer and i==self.epos:
            x = self.enhancer(x)
        
        return x

def resnet12():
    model = ResNet(BasicBlock, [1, 1, 1, 1])
    return model