#
#

from . import mobilenet, resnet, xception

def build_backbone(backbone, output_stride, batch_norm):
    if backbone=='resnet':
        return resnet.ResNet101(output_stride, batch_norm)
    elif backbone=='xception':
        return xception.AlignedXception(output_stride, batch_norm)
    elif backbone=='mobilenet':
        return mobilenet.MobileNetV2(output_stride, batch_norm)
