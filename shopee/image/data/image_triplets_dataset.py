import os
import random
from typing import Any, List, Tuple

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .augmentations import get_inference_augmentations


class ImageTripletsDataset(Dataset):
    """
    Generate triplets of images and corresponding groups id as labels

    Data augmentations can be distinct between anchor, positive and negative
    images. For instance, applying strong data augmentation to the positive
    image can increase its distance to the anchor for generating 'hard
    positive' examples):
    - anchor and negative have identical augmentation
    - positive can be flipped, cropped, resized, etc, for increasing its
        difference with anchor image (especially useful if anchor == positive)


    Args:
        images_path: path to directory containing images
        images: list of images filenames
        labels: list of group id corresponding to each image
        anchor_augmentations: augmentation to apply on anchor images, default
            to get_inference_augmentations output
        positive_augmentations: augmentation to apply on positive images,
            default to anchor_augmentations
        negative_augmentations: augmentation to apply on negative images,
            default to anchor_augmentations
    """

    def __init__(
        self,
        images_path: str,
        images: List[str],
        labels: List[int],
        anchor_augmentations=None,
        positive_augmentations=None,
        negative_augmentations=None,
    ):
        super().__init__()

        self.images_path = images_path
        self.images = images
        self.labels = np.array(labels)

        if anchor_augmentations is None:
            anchor_augmentations = get_inference_augmentations()
        self.anchor_augmentations = anchor_augmentations
        if positive_augmentations is None:
            positive_augmentations = anchor_augmentations
        if negative_augmentations is None:
            negative_augmentations = anchor_augmentations
        self.positive_augmentations = positive_augmentations
        self.negative_augmentations = negative_augmentations

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(
        self, index: int
    ) -> Tuple[Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor, Tensor],]:
        """
        Return triplet of tensor images and the corresponding triplet of label

        Triplet consists of:
            - the anchor image corresponding to index
            - one positive image randomly sampled from the group of the anchor
                image (same label), anchor is removed if the group has at least
                two elements
            - one negative image randomly sampled from an other group than the
                anchor (different label)
        """
        # find a positive sample for current anchor
        positive_candidates = np.argwhere(self.labels == self.labels[index])[
            :, 0
        ].tolist()
        # remove anchor if there exist at least one duplicate
        if len(positive_candidates) > 1:
            positive_candidates.remove(index)
        positive_index = random.choice(positive_candidates)

        # find a negative sample for current anchor
        negative_candidates = np.argwhere(self.labels != self.labels[index])[
            :, 0
        ].tolist()
        negative_index = random.choice(negative_candidates)

        triplet_indices = [index, positive_index, negative_index]

        anchor, positive, negative = [
            cv2.cvtColor(
                cv2.imread(os.path.join(self.images_path, self.images[i])),
                cv2.COLOR_BGR2RGB,
            )
            for i in triplet_indices
        ]

        if self.anchor_augmentations is not None:
            anchor = self.anchor_augmentations(image=anchor)["image"]
        if self.positive_augmentations is not None:
            positive = self.positive_augmentations(image=positive)["image"]
        if self.negative_augmentations is not None:
            negative = self.negative_augmentations(image=negative)["image"]
        triplet = (anchor, positive, negative)

        label = tuple([torch.tensor(self.labels[i]) for i in triplet_indices])

        return triplet, label
