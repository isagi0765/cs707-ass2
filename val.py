#! /data/cxli/miniconda3/envs/th200/bin/python
import argparse
import os
from glob import glob
import random
import numpy as np

import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from collections import OrderedDict

import archs
from dataset import RoadSegmentationDataset  # Changed to your custom dataset
from metrics import iou_score, indicators
from utils import AverageMeter
from albumentations import RandomRotate90, Resize
import time

from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default=None, help='model name')
    parser.add_argument('--output_dir', default='outputs', help='output dir')
    return parser.parse_args()

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main():
    seed_torch()
    args = parse_args()

    with open(f'{args.output_dir}/{args.name}/config.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-'*20)
    for key in config.keys():
        print(f'{key}: {str(config[key])}')
    print('-'*20)

    cudnn.benchmark = True

    # Initialize model
    model = archs.__dict__[config['arch']](
        config['num_classes'], 
        config['input_channels'], 
        config['deep_supervision'], 
        embed_dims=config['input_list']
    ).cuda()

    # Load trained weights
    ckpt = torch.load(f'{args.output_dir}/{args.name}/model.pth')
    model.load_state_dict(ckpt)
    model.eval()

    # Data configuration
    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    # Initialize dataset and loader
    val_dataset = RoadSegmentationDataset(  # Using your custom dataset
        img_dir=os.path.join(config['data_dir'], 'training', 'image_2'),
        mask_dir=os.path.join(config['data_dir'], 'training', 'gt_image_2'),
        transform=val_transform
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False
    )

    # Metrics trackers
    metrics = {
        'iou': AverageMeter(),
        'dice': AverageMeter(),
        'hd95': AverageMeter(),
        'recall': AverageMeter(),
        'specificity': AverageMeter(),
        'precision': AverageMeter()
    }

    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()

            # Forward pass
            output = model(input)
            
            # Calculate metrics
            iou, dice, hd95 = iou_score(output, target)
            iou_, dice_, hd_, hd95_, recall_, specificity_, precision_ = indicators(output, target)

            # Update metrics
            metrics['iou'].update(iou, input.size(0))
            metrics['dice'].update(dice, input.size(0))
            metrics['hd95'].update(hd95, input.size(0))
            metrics['recall'].update(recall_, input.size(0))
            metrics['specificity'].update(specificity_, input.size(0))
            metrics['precision'].update(precision_, input.size(0))

            # Save predictions
            output = (torch.sigmoid(output) > 0.5).float().cpu().numpy()
            os.makedirs(os.path.join(args.output_dir, config['name'], 'predictions'), exist_ok=True)
            for pred, img_id in zip(output, meta['img_id']):
                cv2.imwrite(
                    os.path.join(args.output_dir, config['name'], 'predictions', f'{img_id}.png'),
                    (pred[0] * 255).astype(np.uint8)
                )

    # Print final metrics
    print(f"\nValidation Metrics for {config['name']}:")
    print(f"IoU: {metrics['iou'].avg:.4f}")
    print(f"Dice: {metrics['dice'].avg:.4f}")
    print(f"HD95: {metrics['hd95'].avg:.4f}")
    print(f"Recall: {metrics['recall'].avg:.4f}")
    print(f"Specificity: {metrics['specificity'].avg:.4f}")
    print(f"Precision: {metrics['precision'].avg:.4f}")

if __name__ == '__main__':
    main()
