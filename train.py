import os
import argparse
import numpy as np
import tqdm.tqdm as tqdm

from dataloaders import make_data_loader
from models.sync_batchnorm.replicate import patch_replication_callback
from models.deeplab import *
from utils.loss import SegmentationLoss
from utils.calculate_weights import calculate_weights_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator

class Trainer(object):
    def __init__(self, args):
        self.args = args

        # define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()
        # define DataLoader



def main():
    parser = argparse.ArgumentParser(description="PyTorch DeepLabV3Plus Mobilenet, Xception")
    parser.add_argument("save_directory", type="str", help="save experiments dir name")
