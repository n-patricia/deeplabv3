#
# https://github.com/jfzhang95/pytorch-deeplab-xception.git
#

import math

import torch
import torch.nn as nn
from torch.utils import model_zoo as model_zoo

from models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

import pdb

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, batch_norm=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = batch_norm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.bn2 = batch_norm(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = batch_norm(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, output_stride, batch_norm, pretrained=True):
        super(ResNet, self).__init__()
        self.inplanes = 64
        blocks = [1, 2, 4]
        if output_stride==16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride==8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = batch_norm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0], batch_norm=batch_norm)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1], batch_norm=batch_norm)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2], batch_norm=batch_norm)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[3], batch_norm=batch_norm)
        self.layer4 = self._make_MG_layer(block, 512, blocks=blocks, stride=strides[3], dilation=dilations[3], batch_norm=batch_norm)

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self._init_weight()
        # print(self)
        # if pretrained:
        #     self._load_pretrained_model()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        out = self.layer1(out)
        low_level_features = out
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        return out, low_level_features

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, batch_norm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                batch_norm(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample, batch_norm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, batch_norm=batch_norm))

        return nn.Sequential(*layers)

    def _make_MG_layer(self, block, planes, blocks, stride=1, dilation=1, batch_norm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                batch_norm(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=blocks[0]*dilation, downsample=downsample, batch_norm=batch_norm))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1, dilation=blocks[i]*dilation, batch_norm=batch_norm))

        return nn.Sequential(*layers)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    # def _load_pretrained_model(self):
    #     pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
    #     model_dict = {}
    #     state_dict = self.state_dict()
    #     # print(state_dict.keys())
    #
    #     for k, v in pretrain_dict.items():
    #         if k in state_dict:
    #             model_dict[k] = v
    #     state_dict.update(model_dict)
    #     self.load_state_dict(state_dict)


def _load_state_dict_from_url(arch, model):
    pretrain_dict = model_zoo.load_url(model_urls[arch])
    model_dict = {}
    state_dict = model.state_dict()

    for k, v in pretrain_dict.items():
        if k in state_dict:
            model_dict[k] = v
    state_dict.update(model_dict)
    model.load_state_dict(state_dict)
    return model


def _resnet(arch, block, layers, output_stride, batch_norm, pretrained):
    model = ResNet(block, layers, output_stride, batch_norm, pretrained)
    if pretrained:
        model = _load_state_dict_from_url(arch, model)
    return model


def ResNet50(output_stride, batch_norm, pretrained=True):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, return a model pre-trained on ImageNet
    """
    model = _resnet("resnet50", Bottleneck, [3, 4, 6, 3], output_stride, batch_norm, pretrained)
    return model


def ResNet101(output_stride, batch_norm, pretrained=True):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, return a model pre-trained on ImageNet
    """
    model = _resnet("resnet101", Bottleneck, [3, 4, 23, 3], output_stride, batch_norm, pretrained)
    return model


def ResNet152(output_stride, batch_norm, pretrained=True):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, return a model pre-trained on ImageNet
    """
    model = _resnet("resnet152", Bottleneck, [3, 8, 36, 3], output_stride, batch_norm, pretrained)
    return model


if __name__=='__main__':
    model = ResNet101(output_stride=8, batch_norm=nn.BatchNorm2d, pretrained=True)
    print(model)

    inp = torch.rand(1, 3, 512, 512)
    out, low_feat = model(inp)
    print(out.size())
    print(low_feat.size())
