#
# https://github.com/jfzhang95/pytorch-deeplab-xception.git
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.aspp import build_aspp
from models.decoder import build_decoder
from models.backbone import build_backbone
from models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

class DeepLab(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_class=21, sync_bn=True, freeze_bn=False):
        super(DeepLab, self).__init__()
        if sync_bn:
            batch_norm = SynchronizedBatchNorm2d
        else:
            batch_norm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, batch_norm)
        self.aspp = build_aspp(backbone, output_stride, batch_norm)
        self.decoder = build_decoder(num_class, backbone, batch_norm)
        self.freeze_bn = freeze_bn

    def forward(self, x):
        out, low_level_feat = self.backbone(x)
        out = self.aspp(out)
        out = self.decoder(out, low_level_feat)

        out = F.interpolate(out, size=x.size()[2:], mode="bilinear", align_corners=True)
        return out

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

if __name__=='__main__':
    model = DeepLab(backbone="mobilenet", output_stride=16)
    model.eval()
    inp = torch.rand(1, 3, 513, 513)
    oup = model(inp)
    print(oup.size())
