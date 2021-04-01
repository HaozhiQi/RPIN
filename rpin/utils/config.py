# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from yacs.config import CfgNode

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CfgNode()

_C.DATA_ROOT = './data/'
_C.DATASET_ABS = 'Phys'
_C.PHYRE_PROTOCAL = 'within'
_C.PHYRE_FOLD = 0
# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CfgNode()
_C.INPUT.PRELOAD_TO_MEMORY = False
_C.INPUT.IMAGE_MEAN = [0, 0, 0]
_C.INPUT.IMAGE_STD = [1.0, 1.0, 1.0]
_C.INPUT.IMAGE_CHANNEL = 3
# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CfgNode()

_C.SOLVER.BASE_LR = 0.001
_C.SOLVER.LR_GAMMA = 0.1
_C.SOLVER.VAL_INTERVAL = 16000
_C.SOLVER.WEIGHT_DECAY = 0.0
_C.SOLVER.WARMUP_ITERS = -1
_C.SOLVER.LR_MILESTONES = [12000000, 18000000]
_C.SOLVER.MAX_ITERS = 20000000
_C.SOLVER.BATCH_SIZE = 128
_C.SOLVER.SCHEDULER = 'step'

# ---------------------------------------------------------------------------- #
# Intuitive Physics models
# ---------------------------------------------------------------------------- #
_C.RPIN = CfgNode()
_C.RPIN.ARCH = ''
_C.RPIN.BACKBONE = ''
_C.RPIN.HORIZONTAL_FLIP = False
_C.RPIN.VERTICAL_FLIP = False
# prediction setting
_C.RPIN.INPUT_SIZE = 4
_C.RPIN.PRED_SIZE_TRAIN = 20
_C.RPIN.PRED_SIZE_TEST = 40
# input for mixed dataset
_C.RPIN.IMAGE_EXT = '.jpg'
_C.RPIN.INPUT_HEIGHT = 360
_C.RPIN.INPUT_WIDTH = 640
# training setting
_C.RPIN.MAX_NUM_OBJS = 3
_C.RPIN.POSITION_LOSS_WEIGHT = 1
_C.RPIN.MASK_LOSS_WEIGHT = 0.0
_C.RPIN.MASK_SIZE = 14
# additional input
_C.RPIN.IMAGE_FEATURE = True
# ROI POOLING
_C.RPIN.ROI_POOL_SIZE = 1
_C.RPIN.ROI_POOL_SPATIAL_SCALE = 0.25
_C.RPIN.ROI_POOL_SAMPLE_R = 1
# parameter
_C.RPIN.VE_FEAT_DIM = 32
_C.RPIN.IN_FEAT_DIM = 64
# uncertainty modeling, usually just for ShapeStacks
_C.RPIN.VAE = False
_C.RPIN.VAE_KL_LOSS_WEIGHT = 0.001
# DISCOUNT
_C.RPIN.DISCOUNT_TAU = 0.01
# EXPLORE ARCH
_C.RPIN.N_EXTRA_ROI_F = 0
_C.RPIN.N_EXTRA_PRED_F = 0
_C.RPIN.N_EXTRA_SELFD_F = 0
_C.RPIN.N_EXTRA_RELD_F = 0
_C.RPIN.N_EXTRA_AFFECTOR_F = 0
_C.RPIN.N_EXTRA_AGGREGATOR_F = 0
_C.RPIN.EXTRA_F_KERNEL = 3
_C.RPIN.EXTRA_F_PADDING = 1
_C.RPIN.SEQ_CLS_LOSS_WEIGHT = 0.0

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = './outputs/default'
