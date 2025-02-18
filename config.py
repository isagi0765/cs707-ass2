import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, adjusted for road segmentation
_C.DATA.BATCH_SIZE = 4
# Path to dataset
_C.DATA.DATA_PATH = './data_road'
# Dataset name
_C.DATA.DATASET = 'road_segmentation'
# Input image size (resize images to this size)
_C.DATA.IMG_SIZE = 256
# Interpolation to resize image (options: random, bilinear, bicubic)
_C.DATA.INTERPOLATION = 'bilinear'
# Use zipped dataset instead of folder dataset
_C.DATA.ZIP_MODE = False
# Cache Data in Memory
_C.DATA.CACHE_MODE = 'part'
# Pin CPU memory in DataLoader for efficient transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 2

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type (e.g., Swin Transformer)
_C.MODEL.TYPE = 'swin'
# Model name
_C.MODEL.NAME = 'swin_tiny_patch4_window7_224'
# Pretrained checkpoint path
_C.MODEL.PRETRAIN_CKPT = './pretrained_ckpt/swin_tiny_patch4_window7_224.pth'
_C.MODEL.RESUME = ''
# Number of classes (binary segmentation: road vs non-road)
_C.MODEL.NUM_CLASSES = 1
# Dropout rate
_C.MODEL.DROP_RATE = 0.0
# Drop path rate
_C.MODEL.DROP_PATH_RATE = 0.1

# Swin Transformer parameters (adjusted for segmentation)
_C.MODEL.SWIN.PATCH_SIZE = 4
_C.MODEL.SWIN.IN_CHANS = 3  # RGB input channels
_C.MODEL.SWIN.EMBED_DIM = 96
_C.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN.WINDOW_SIZE = 7
_C.MODEL.SWIN.MLP_RATIO = 4.0

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 50  # Adjusted for road segmentation dataset size
_C.TRAIN.WARMUP_EPOCHS = 5
_C.TRAIN.WEIGHT_DECAY = 0.01
_C.TRAIN.BASE_LR = 1e-4
_C.TRAIN.WARMUP_LR = 1e-6

# Optimizer settings (AdamW)
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)

# -----------------------------------------------------------------------------
# Miscellaneous settings
# -----------------------------------------------------------------------------
# Output directory for checkpoints and logs
_C.OUTPUT_DIR = './output'
