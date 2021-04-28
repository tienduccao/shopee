from typing import Tuple

import albumentations
from albumentations.pytorch.transforms import ToTensorV2


DEFAULT_SIZE: Tuple[int, int] = (224, 224)

def get_inference_augmentations(size: Tuple[int, int] = DEFAULT_SIZE):
    """Return data augmentations pipeline for inference"""
    # TODO: check and use the following instead of resizing (for keeping aspect
    # ratio of non square image, if any):
    # https://albumentations.ai/docs/api_reference/augmentations/geometric/resize/#albumentations.augmentations.geometric.resize.LongestMaxSize
    # https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.PadIfNeeded
    return albumentations.Compose([
            albumentations.Resize(size[0], size[1], always_apply=True),
            albumentations.Normalize(always_apply=True),
            ToTensorV2(always_apply=True)
        ])