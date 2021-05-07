import torch
import argparse
import os
import sys
import cv2
import time


class Configuration():
    def __init__(self):
        self.EXP_NAME = 'res2net_base'

        self.DIR_ROOT = './'
        self.DIR_DATA = os.path.join(self.DIR_ROOT, 'data')
        self.DIR_RESULT = os.path.join(self.DIR_ROOT, 'result', self.EXP_NAME)
        self.DIR_CKPT = os.path.join(self.DIR_RESULT, 'ckpt')
        self.DIR_LOG = os.path.join(self.DIR_RESULT, 'log')
        self.DIR_IMG_LOG = os.path.join(self.DIR_RESULT, 'log', 'img')
        self.DIR_EVALUATION = os.path.join(self.DIR_RESULT, 'eval')
        self.DIR_VECOLOR = os.path.join(self.DIR_ROOT, 'data')

        self.DATASETS = ['vecolor']
        self.DATA_WORKERS = 4
        self.DATA_MIN_SCALE_FACTOR = 1.
        self.DATA_MAX_SCALE_FACTOR = 1.3
        self.DATA_SHORT_EDGE_LEN = 480
        self.DATA_RANDOMCROP = (480, 480)
        self.DATA_RANDOMFLIP = 0.5

        self.COLOR_LIST = ["black", "blue", "cyan", "gray", "green", "red", "white", "yellow"]

        self.PRETRAIN = True

        self.MODEL_NUM_CLASSES = len(self.COLOR_LIST)
        self.MODEL_GCT_BETA_WD = True

        self.TRAIN_TOTAL_STEPS = 50000
        self.TRAIN_START_STEP = 0
        self.TRAIN_EVAL_STEPS = 5000
        self.TRAIN_LR = 0.01
        self.TRAIN_MOMENTUM = 0.9
        self.TRAIN_COSINE_DECAY = False
        self.TRAIN_WARM_UP_STEPS = 1000
        self.TRAIN_WEIGHT_DECAY = 15e-5
        self.TRAIN_POWER = 0.9
        self.TRAIN_GPUS = 2
        self.TRAIN_BATCH_SIZE = 8
        self.TRAIN_LOG_STEP = 20
        self.TRAIN_CLIP_GRAD_NORM = 5.
        self.TRAIN_SAVE_STEP = 5000
        self.TRAIN_RESUME = False
        self.TRAIN_RESUME_CKPT = None
        self.TRAIN_RESUME_STEP = 0
        self.TRAIN_AUTO_RESUME = True

        # self.TEST_GPU_ID = 0
        # self.TEST_DATASET = 'youtubevos'
        # self.TEST_DATASET_FULL_RESOLUTION = False
        # self.TEST_DATASET_SPLIT = ['val']
        # self.TEST_CKPT_PATH = None
        # self.TEST_CKPT_STEP = None  # if "None", evaluate the latest checkpoint.
        # self.TEST_FLIP = False
        # self.TEST_MULTISCALE = [1]
        # self.TEST_MIN_SIZE = None
        # self.TEST_MAX_SIZE = 800 * 1.3 if self.TEST_MULTISCALE == [1.] else 800
        # self.TEST_WORKERS = 4

        # dist
        self.DIST_ENABLE = True
        self.DIST_BACKEND = "gloo"
        self.DIST_URL = "file:///tmp/sharefile"
        self.DIST_START_GPU = 0

        self.__check()

    def __check(self):
        if not torch.cuda.is_available():
                raise ValueError('config.py: cuda is not avalable')
        if self.TRAIN_GPUS == 0:
                raise ValueError('config.py: the number of GPU is 0')
        for path in [self.DIR_RESULT, self.DIR_CKPT, self.DIR_LOG, self.DIR_EVALUATION, self.DIR_IMG_LOG]:
            if not os.path.isdir(path):
                os.makedirs(path)



cfg = Configuration()
