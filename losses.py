import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

__all__ = ['BCEDiceLoss', 'LovaszHingeLoss']

class BCEDiceLoss(nn.Module):
    """
    Combines Binary Cross-Entropy (BCE) loss and Dice loss.
    
    BCE measures the difference between predicted probabilities and true labels.
    Dice loss measures the overlap between predicted and true masks.
    """
    def __init__(self, alpha=0.5):
        """
        Initialize the loss function with a weighting factor for BCE and Dice loss.
        
        Args:
            alpha (float, optional): Weighting factor for BCE loss. Defaults to 0.5.
        """
        super().__init__()
        self.alpha = alpha

    def forward(self, input, target):
        """
        Compute the combined loss.
        
        Args:
            input (torch.Tensor): Predicted logits.
            target (torch.Tensor): Ground truth masks.
        
        Returns:
            torch.Tensor: Combined BCE and Dice loss.
        """
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice_loss = 1 - dice.mean()  # Use mean instead of sum for consistency
        return self.alpha * bce + (1 - self.alpha) * dice_loss


class LovaszHingeLoss(nn.Module):
    """
    Lovasz Hinge loss for binary segmentation tasks.
    
    This loss is based on the Lovasz extension of the hinge loss.
    """
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        """
        Compute the Lovasz Hinge loss.
        
        Args:
            input (torch.Tensor): Predicted logits.
            target (torch.Tensor): Ground truth masks.
        
        Returns:
            torch.Tensor: Lovasz Hinge loss.
        """
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss.mean()  # Use mean for consistency across batches
