#
#

from __future__ import print_function, division

import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import dataloaders.custom_transforms as tf

class VOCSegmentation(data.Dataset):
    NUM_CLASS = 21

    def __init__(self, args, base_dir="./", split="train"):
        super().__init__()
        self.base_dir = base_dir
        self.image_dir = os.path.join(self.base_dir, "JPEGImages")
        self.categ_dir = os.path.join(self.base_dir, "SegmentationClass")

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.args = args
        splits_dir = os.path.join(self.base_dir, "ImageSets", "Segmentation")
        self.im_ids = []
        self.images = []
        self.categories = []

        for s in self.split:
            with open(os.path.join(splits_dir, s+".txt"), "r") as f:
                lines = f.read().splitlines()

            for ii, line in enumerate(lines):
                img = os.path.join(self.image_dir, line+".jpg")
                cat = os.path.join(self.categ_dir, line+".png")
                assert os.path.isfile(img)
                assert os.path.isfile(cat)
                self.im_ids.append(line)
                self.images.append(img)
                self.categories.append(cat)

        assert (len(self.images) == len(self.categories))

        print("Number of images in {}: {:d}".format(split, len(self.images)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image, label = self._make_img_gt_point_pair(index)
        sample = {"image": image, "label": label}

        for split in self.split:
            if split=="train":
                return self._transform_train(sample)
            elif split=="val":
                return self._transform_val(sample)


    def _make_img_gt_point_pair(self, index):
        image = Image.open(self.images[index]).convert("RGB")
        target = Image.open(self.categories[index])

        return image, target


    def _transform_train(self, sample):
        composed_transform = transforms.Compose([
            tf.RandomHorizontalFlip(),
            tf.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            tf.RandomGaussianBlur(),
            tf.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tf.ToTensor()])

        return composed_transform(sample)


    def _transform_val(self, sample):
        composed_transform = transforms.Compose([
            tf.FixScaleCrop(crop_size=self.args.crop_size),
            tf.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tf.ToTensor()])

        return composed_transform(sample)

    def __str__(self):
        return f"VOC2012(split={self.split})"


if __name__=="__main__":
    import argparse
    import numpy as np
    import torch.utils.data as data
    from .dataloaders.utils import decode_segmap

    import pdb

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513
    args.dataset = "pascal"

    voc_train = VOCSegmentation(args, split="train")
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

            print(img_tmp)

        if i==1:
            break
