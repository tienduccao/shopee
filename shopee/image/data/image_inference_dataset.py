import os
from typing import Tuple

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset

from .augmentations import DEFAULT_SIZE, get_inference_augmentations


class ImageInferenceDataset(Dataset):
    """
    Image dataset for inference

    Args:
        data: shopee dataframe (test/train)
        images_root_dir: path to directory containing images
        size: size for resizing images

    Attributes:
        data:
        images_root_dir:
        augmentations: albumentations augmentations for inference obtained
            from get_inference_augmentations
    """
    
    def __init__(
        self, 
        data: pd.DataFrame, 
        images_root_dir: str, 
        size: Tuple[int, int] = DEFAULT_SIZE,
    ):
        self.images = list(data.image)
        self.augmentations = get_inference_augmentations(size)
        self.images_root_dir = images_root_dir
        
    def __len__(self) -> int:
        """Length of the Dataset"""
        return len(self.images)
    
    def __getitem__(self, index: int) -> torch.Tensor:
        """Return image corresponding to row index"""
        path = os.path.join(self.images_root_dir, self.images[index])
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        image = self.augmentations(image=image)["image"]
            
        return image

