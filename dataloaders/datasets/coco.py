#
#

from __future__ import print_function, division

import os
import numpy as np
from PIL import Image
from tqdm import trange

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from pycocotools.coco import COCO
from pycocotools import mask

import dataloaders.custom_transforms as tf


class COCOSegmentation(data.Dataset):
    CAT_LIST = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4, 1, 64, 20, 63, 7, 72]

    def __init__(self, args, base_dir="./", split="train", year="2017"):
        super().__init__()

        self.base_dir = base_dir
        ann_file = os.path.join(self.base_dir, "annotations/instances_{}{}.json".format(split, year))
        ids_file = os.path.join(self.base_dir, "annotations/{}_ids_{}.pth".format(split, year))
        self.img_dir = os.path.join(self.base_dir, "images/{}{}".format(split, year))
        self.split = split
        self.coco = COCO(ann_file)
        self.coco_mask = mask
        if os.path.exists(ids_file):
            self.ids = torch.load(ids_file)
        else:
            ids = list(self.coco.imgs.keys())
            self.ids = self._preprocess(ids, ids_file)
        self.args = args

    def __getitem__(self, index):
        img, target = self._make_img_gt_point_pair(index)
        sample = {"image": img, "label": target}

        if self.split == "train":
            return self._transform_tr(sample)
        elif self.split == "val":
            return self._transform_val(sample)

    def _make_img_gt_point_pair(self, index):
        coco = self.coco
        img_id = self.ids[index]
        img_metadata = coco.loadImgs(img_id)[0]
        path = img_metadata["file_name"]
        img = Image.open(os.path.join(self.img_dir, path)).convert("RGB")
        cocotarget = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        target = Image.fromarray(self._gen_seg_mask(cocotarget, img_metadata["height"], img_metadata["width"]))

        return img, target

    def _preprocess(self, ids, ids_file):
        print("Preprocessing mask, this will take a while. " + \
              "But don't worry, it only run once for each split.")
        tbar = trange(len(ids))
        new_ids = []
        for i in tbar:
            img_id = ids[i]
            cocotarget = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
            img_metadata = self.coco.loadImgs(img_id)[0]
            mask = self._gen_seg_mask(cocotarget, img_metadata["height"], img_metadata["width"])
            # more than 1k pixels
            if (mask > 0).sum() > 1000:
                new_ids.append(img_id)
            tbar.set_description("Doing: {}/{}, got {} qualified images".format(i, len(ids), len(new_ids)))
        print("Found number of qualified images: ", len(new_ids))
        torch.save(new_ids, ids_file)
        return new_ids

    def _gen_seg_mask(self, target, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        coco_mask = self.coco_mask
        for instance in target:
            rle = coco_mask.frPyObjects(instance["segmentation"], h, w)
            m = coco_mask.decode(rle)
            cat = instance["category_id"]
            if cat in self.CAT_LIST:
                c = self.CAT_LIST.index(cat)
            else:
                continue
            if len(m.shape) < 3:
                mask[:, :] += (mask == 0) * (m * c)
            else:
                mask[:, :] += (mask == 0) * (((np.sum(m, axis=2)) > 0) * c).astype(np.uint8)
        return mask

    def _transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tf.RandomHorizontalFlip(),
            tf.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            tf.RandomGaussianBlur(),
            tf.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tf.ToTensor()])

        return composed_transforms(sample)

    def _transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tf.FixScaleCrop(crop_size=self.args.crop_size),
            tf.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tf.ToTensor()])

        return composed_transforms(sample)

    def __len__(self):
        return len(self.ids)


if __name__ == "__main__":
    from .dataloaders.utils import decode_segmap
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513

    coco_val = COCOSegmentation(args, split='val', year='2017')

    dl = data.DataLoader(coco_val, batch_size=4, shuffle=True, num_workers=0)

    for i, sample in enumerate(dl):
        for j in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[j]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='coco')
            img_tmp = np.transpose(img[j], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)

            print(img_tmp)

        if i==1:
            break
