import argparse
import torch.nn as nn

class qkv_transform(nn.Conv1d):
    """1D convolution layer for query/key/value transformation in attention mechanisms.
    
    Inherits from `nn.Conv1d` and can be used as a standard Conv1d layer.
    Typically used in transformer architectures for processing sequence data.
    """
    pass  # Add custom implementation if needed

def str2bool(v):
    """Convert string input to boolean for argument parsing.
    
    Args:
        v (str): Input value to convert
        
    Returns:
        bool: True for 'true', '1', 'yes'; False for 'false', '0', 'no'
        
    Raises:
        argparse.ArgumentTypeError: If input cannot be converted to bool
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', '1'):
        return True
    elif v.lower() in ('no', 'false', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected. Got: %s' % v)

def count_params(model):
    """Count the number of trainable parameters in a model.
    
    Args:
        model (nn.Module): PyTorch model
        
    Returns:
        int: Total number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class AverageMeter(object):
    """Computes and stores running averages for metrics.
    
    Attributes:
        val: Current value
        avg: Running average
        sum: Cumulative sum
        count: Total number of updates
    """
    
    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics to zero."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Update metrics with new value.
        
        Args:
            val (float): Value to add to metrics
            n (int): Number of samples this value represents
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
