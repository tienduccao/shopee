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
    return albumentations.Compose(
        [
            albumentations.Resize(size[0], size[1], always_apply=True),
            albumentations.Normalize(always_apply=True),
            ToTensorV2(always_apply=True),
        ]
    )


# https://www.kaggle.com/parthdhameliya77/shopee-pytorch-eca-nfnet-l0-image-training
def get_train_augmentations(size: Tuple[int, int] = DEFAULT_SIZE):
    return albumentations.Compose(
        [   
            albumentations.Resize(size[0], size[1],always_apply=True),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.Rotate(limit=120, p=0.8),
            albumentations.RandomBrightness(limit=(0.09, 0.6), p=0.5),
            albumentations.Normalize(always_apply=True),
            ToTensorV2(always_apply=True),
        ]
    )

