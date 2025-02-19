import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class RoadSegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_ids, transform=None):
        """
        Args:
            img_dir: Path to image directory (training/image_2)
            mask_dir: Path to mask directory (training/gt_image_2)
            img_ids: List of image IDs from train.txt/val.txt
            transform: Albumentations transforms
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ids = [x.strip() for x in img_ids]
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]  # e.g., "uu_000041.png"
        
        # Convert image ID to mask ID format: "uu_road_000041.png"
        parts = img_id.replace('.png', '').split('_')
        if len(parts) >= 3:  # Handle lane vs road masks
            mask_id = f"{parts[0]}_road_{'_'.join(parts[2:])}.png"
        else:
            mask_id = img_id.replace('_', '_road_')

        # Build paths
        img_path = os.path.join(self.img_dir, img_id)
        mask_path = os.path.join(self.mask_dir, mask_id)

        # Verify file existence
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Missing image: {img_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Missing mask: {mask_path}")

        # Read and process images
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Normalize and threshold
        img = img.astype(np.float32) / 255.0
        mask = (mask > 128).astype(np.float32)  # Binary mask

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        # Convert to tensors
        img = torch.tensor(img.transpose(2, 0, 1))  # HWC to CHW
        mask = torch.tensor(mask).unsqueeze(0)      # Add channel dimension
        
        return img, mask, {'img_id': img_id}
