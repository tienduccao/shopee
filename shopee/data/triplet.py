import random
from typing import Any, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def sample_triplets(groups: List[List[Any]]) -> List[List[Any]]:
    """Generate samples of triplets from groups of samples

    Args:
        groups: list of groups of samples from which to select triplets

    Returns:
        a list of triplets samples [anchor, positive, negative] where anchor
            and positive are sampled from the same group while negative is
            sampled from a distinct group
    """
    anchor_groups = [group for group in groups if len(group) > 1]
    n_triplets = min(len(groups) // 2, len(anchor_groups))
    anchor_groups = random.sample(anchor_groups, n_triplets)

    negative_groups = random.sample(
        [group for group in groups if group not in anchor_groups],
        n_triplets,
    )

    # generate_triplets
    triplets = []
    for anchor_group, negative_group in zip(anchor_groups, negative_groups):
        anchor, positive = random.sample(anchor_group, 2)
        negative = random.choice(negative_group)
        triplets.append([anchor, positive, negative])

    return triplets


def show_triplet(
    triplet: Tuple[np.ndarray, np.ndarray, np.ndarray],
    title: Optional[str] = None,
    figure_scale: int = 5,
):
    """Plot a triplet of images (anchor, positive, negative)"""
    plots = zip(triplet, ["anchor", "positive", "negative"])

    plt.figure(figsize=(3 * figure_scale, figure_scale))
    for j, (image, subtitle) in enumerate(plots):
        plt.subplot(1, 3, j + 1)
        plt.imshow(image)
        plt.title(f"{subtitle}, shape: {image.shape[:2]}")
        plt.axis("off")

    if title:
        plt.suptitle(title)
    plt.show()
