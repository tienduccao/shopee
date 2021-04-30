# Adapted from:
# https://www.kaggle.com/tanulsingh077/pytorch-metric-learning-pipeline-only-images#Metric-Learning-Losses
# https://www.kaggle.com/ragnar123/shopee-efficientnetb3-arcmarginproduct
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .hardest_triplet import hardest_triplet


class FaceTripletLoss(nn.modules.loss._Loss):
    """
    ArcFace loss with triplet. Classification weights are replaced with
    anchor embeddings for each triplet.

    Args:
        theta_margin: the margin used in ArcFace such that Cosine Similarity
            between anchor and positive
            cos(theta) = cosine_similarity(anchor, positive)
            is replaced with cos(theta + theta_margin)
        cos_margin: the margin used in CosFace (additive margin),
            cos(theta) is replaced with cos(theta) + cos_margin
        scale: the scaling used in ArcFace for adapting the sphere radius
        easy_margin: whether to use easy_margin (not in the paper?)
        size_average: for compatibility with Pytorch Loss
        reduce: same
        reduction: reduction to use for loss

    """

    def __init__(
        self,
        theta_margin: float = 0.5,
        cos_margin: float = 0.0,
        scale: float = 30.0,
        easy_margin: bool = False,
        #         ls_eps: float = 0.0, # TODO: add label smoothing for triplet?
        size_average=None,
        reduce=None,
        reduction: str = "mean",
    ):
        super(FaceTripletLoss, self).__init__(size_average, reduce, reduction)
        self.theta_margin = theta_margin
        self.cos_margin = cos_margin

        # used for computing cos(theta + theta_margin)
        self.cos_m = math.cos(theta_margin)
        self.sin_m = math.sin(theta_margin)

        # used for easy margin
        self.easy_margin = easy_margin
        self.th = math.cos(math.pi - theta_margin)
        self.mm = math.sin(math.pi - theta_margin) * theta_margin

        self.scale = scale

    #         self.ls_eps = ls_eps  # label smoothing

    def forward(
        self,
        triplet: Tuple[Tensor, Tensor, Tensor],
        # anchor: torch.Tensor,
        # positive: torch.Tensor,
        # negative: torch.Tensor,
        label: Optional[Tensor] = None,
    ) -> torch.Tensor:
        """
        Return loss value for input triplet

        Args:
            triplet: the triplet (anchor, positive, negative)
            label: group label of the triplet (anchor, positive, negative).
                If provided, label are used for replacing each negative sample
                with hardest negative extracted from the whole batch by using
                hardest_triplet function

        Returns:
            the loss value
        """
        anchor, positive, negative = triplet
        if label is None:
            cos_p = F.cosine_similarity(anchor, positive)
            cos_n = F.cosine_similarity(anchor, negative)
        else:
            triplets = (anchor, positive, negative)
            cos_p, cos_n = hardest_triplet(
                triplets, label, F.cosine_similarity, mode="max"
            )

        sin_p = torch.sqrt(1.0 - torch.pow(cos_p, 2))
        phi = cos_p * self.cos_m - sin_p * self.sin_m

        if self.easy_margin:
            phi = torch.where(cos_p > 0, phi, cos_p)
        else:
            phi = torch.where(cos_p > self.th, phi, cos_p - self.mm)

        logits = phi - self.cos_margin - cos_n
        logits *= self.scale
        label = torch.ones_like(logits)

        return F.binary_cross_entropy_with_logits(
            logits, label, reduction=self.reduction
        )
