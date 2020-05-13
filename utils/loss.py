import torch
import torch.nn as nn

class SegmentationLoss(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode="ce"):
        """mode: ce, focal"""
        if mode=="ce":
            return self.CrossEntropyLoss
        elif mode=="focal":
            return self.FocalLoss

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index, size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()
        loss = criterion(logit, target.long())
        if self.batch_average:
            loss = loss / n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index, size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()
        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt = logpt * alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss = loss / n

        return loss


if __name__=="__main__":
    loss = SegmentationLoss()
    a = torch.rand(1, 3, 7, 7)
    b = torch.rand(1, 7, 7)

    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b).item())