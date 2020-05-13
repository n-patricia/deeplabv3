#
#


import argparse
import numpy as np
import matplotlib.image as im
import torch.utils.data as data
from dataloaders.utils import decode_segmap
from dataloaders.datasets.pascal import VOCSegmentation

import pdb

parser = argparse.ArgumentParser()
args = parser.parse_args()
args.base_dir = "/Users/nopz/Google Drive/voc/VOCdevkit/VOC2012"
args.base_size = 513
args.crop_size = 513
args.dataset = "pascal"

voc_train = VOCSegmentation(args, base_dir=args.base_dir, split="train")
dl = data.DataLoader(voc_train, batch_size=5, shuffle=True, num_workers=0)

pdb.set_trace()
for i, sample in enumerate(dl):
    for j in range(sample["image"].size()[0]):
        img = sample["image"].numpy()
        gt = sample["label"].numpy()
        tmp = np.array(gt[j]).astype(np.uint8)
        segmap = decode_segmap(tmp, dataset=args.dataset)
        img_tmp = np.transpose(img[j], axes=[1, 2, 0])
        img_tmp += (0.229, 0.224, 0.225)
        img_tmp += (0.485, 0.456, 0.406)
        img_tmp *= 255.0
        img_tmp = img_tmp.astype(np.uint8)
        im.imsave("sgmap.jpg", segmap)
        print(img_tmp.size)

    if i==1:
        break
