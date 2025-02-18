import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class RoadSegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_ids, transform=None):
        """
        Args:
            img_dir: Path to image directory
            mask_dir: Path to mask directory
            img_ids: List of image IDs from train.txt/val.txt
            transform: Albumentations transforms
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ids = [x.strip() for x in img_ids]  # Clean whitespace
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        
        # Build paths with .png extension
        img_path = os.path.join(self.img_dir, img_id)
        mask_path = os.path.join(self.mask_dir, img_id)

        # Read images with existence checks
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Missing image: {img_path}")
            
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Missing mask: {mask_path}")

        # Normalize and process
        img = img.astype(np.float32) / 255.0
        mask = (mask > 128).astype(np.float32)  # Threshold masks

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        # Convert to tensors
        img = torch.tensor(img.transpose(2, 0, 1))  # HWC to CHW
        mask = torch.tensor(mask).unsqueeze(0)      # Add channel dim
        
        return img, mask, {'img_id': img_id}
