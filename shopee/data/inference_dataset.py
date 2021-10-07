import random
from typing import Any, List, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

# from .augmentations import get_inference_augmentations


class InferenceDataset(Dataset):
#     """
#     Generate triplets of images and corresponding groups id as labels

#     Data augmentations can be distinct between anchor, positive and negative
#     images. For instance, applying strong data augmentation to the positive
#     image can increase its distance to the anchor for generating 'hard
#     positive' examples):
#     - anchor and negative have identical augmentation
#     - positive can be flipped, cropped, resized, etc, for increasing its
#         difference with anchor image (especially useful if anchor == positive)


#     Args:
#         images_path: path to directory containing images
#         images: list of images filenames
#         labels: list of group id corresponding to each image
#         anchor_augmentations: augmentation to apply on anchor images, default
#             to get_inference_augmentations output
#         positive_augmentations: augmentation to apply on positive images,
#             default to anchor_augmentations
#         negative_augmentations: augmentation to apply on negative images,
#             default to anchor_augmentations
#     """

    def __init__(
        self,
        image_embeddings,
        text_embeddings,
#         labels: np.ndarray,
    ):
        super().__init__()

        self.image_embeddings = image_embeddings
        self.text_embeddings = text_embeddings
#         self.labels = labels

    def __len__(self) -> int:
        return len(self.image_embeddings)

    def __getitem__(
        self, index: int
    ) -> Tuple[Tensor, Tensor]:
        return self.image_embeddings[index], self.text_embeddings[index]
