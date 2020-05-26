#
#


import argparse
import numpy as np
import matplotlib.image as im
import torch
import torch.utils.data as data
from dataloaders.utils import decode_segmap
from dataloaders.datasets.pascal import VOCSegmentation
from models.deeplab import DeepLab

import pdb


def load_pretrained_model(model, pretrained_file):
    # pdb.set_trace()
    dict_model = torch.load("./pretrained/deeplab-resnet.pth", map_location=torch.device("cpu"))
    for k in dict_model["state_dict"]:
        print(model.state_dict()[k].size(), dict_model["state_dict"][k].size())

    return model



parser = argparse.ArgumentParser()
args = parser.parse_args()
args.base_dir = "/Volumes/CYRED/Google Drive/voc/VOCdevkit/VOC2012"
args.base_size = 513
args.crop_size = 513
args.dataset = "pascal"
pdb.set_trace()

model = DeepLab(output_stride=8, backbone="resnet") #, sync_bn=False)
dict_model = torch.load("./pretrained/deeplab-resnet.pth", map_location=torch.device("cpu"))
model.load_state_dict(dict_model["state_dict"])
# load_pretrained_model(model, "./pretrained/deeplab-resnet.pth")
# model.load_state_dict(torch.load("./pretrained/deeplab-resnet.pth", map_location=torch.device("cpu")))#, strict=False)
model.eval()

val = VOCSegmentation(args, base_dir=args.base_dir, split="val")
dl = data.DataLoader(val, batch_size=1, shuffle=False, num_workers=0)

# pdb.set_trace()
for i, sample in enumerate(dl):
    img, tar = sample["image"], sample["label"]
    # print(img.size())
    # print(tar.size())
    out = model(img)
    pred = torch.argmax(out.squeeze(), dim=0).detach().numpy()
    seg = decode_segmap(pred, "pascal")
    im.imsave("pred.png", seg)
    # for j in range(sample["image"].size()[0]):
    #     img = sample["image"].numpy()
    #     gt = sample["label"].numpy()
    #     tmp = np.array(gt[j]).astype(np.uint8)
    #     segmap = decode_segmap(tmp, dataset=args.dataset)
    #     img_tmp = np.transpose(img[j], axes=[1, 2, 0])
    #     img_tmp += (0.229, 0.224, 0.225)
    #     img_tmp += (0.485, 0.456, 0.406)
    #     img_tmp *= 255.0
    #     img_tmp = img_tmp.astype(np.uint8)
    #     im.imsave("sgmap.jpg", segmap)
    #     print(img_tmp.size)
