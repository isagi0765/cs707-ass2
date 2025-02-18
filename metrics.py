import numpy as np
import torch
import torch.nn.functional as F

from medpy.metric.binary import jc, dc, hd, hd95, recall, specificity, precision


def iou_score(output, target):
    """
    Calculate IoU and Dice score from output and target.

    Args:
        output (torch.Tensor): Predicted logits.
        target (torch.Tensor): Ground truth masks.

    Returns:
        tuple: IoU, Dice score, and Hausdorff distance 95%.
    """
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)
    dice = (2 * iou) / (iou + 1)

    try:
        hd95_ = hd95(output_, target_)
    except:
        hd95_ = 0

    return iou, dice, hd95_


def dice_coef(output, target):
    """
    Calculate Dice coefficient from output and target.

    Args:
        output (torch.Tensor): Predicted logits.
        target (torch.Tensor): Ground truth masks.

    Returns:
        float: Dice coefficient.
    """
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / (output.sum() + target.sum() + smooth)


def indicators(output, target):
    """
    Calculate various segmentation metrics.

    Args:
        output (torch.Tensor): Predicted logits.
        target (torch.Tensor): Ground truth masks.

    Returns:
        tuple: IoU, Dice coefficient, Hausdorff distance, Hausdorff distance 95%,
               recall, specificity, precision.
    """
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5

    iou_ = jc(output_, target_)
    dice_ = dc(output_, target_)
    hd_ = hd(output_, target_)
    hd95_ = hd95(output_, target_)
    recall_ = recall(output_, target_)
    specificity_ = specificity(output_, target_)
    precision_ = precision(output_, target_)

    return iou_, dice_, hd_, hd95_, recall_, specificity_, precision_
