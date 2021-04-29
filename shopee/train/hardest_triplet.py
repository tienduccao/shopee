from typing import Callable, Tuple

from einops import rearrange
import torch
from torch import Tensor

DEBUG = False  # activate debug mode when needed


def hardest_triplet(
    triplet: Tuple[Tensor, Tensor, Tensor],
    label: Tuple[Tensor, Tensor, Tensor],
    distance: Callable[[Tensor, Tensor], Tensor],
    mode: str = "min",
) -> Tuple[Tensor, Tensor]:
    """Return hardest negative distance of a batch for each anchor

    Args:
        triplet: the triplet input embedding (anchor, positive, negative) of
            shape (B, C) where C is embedding dimension
        label: the corresponding triplet of label, each of shape (B)
        distance: the distance to be used
        mode: if "min" hardest negative are supposed to have minimal distance
            to anchor, else maximum distance

    Returns:
        positive_distance: the distance between anchor and positive samples
            (shape B)
        hardest_negative_distance: the hardest distance between anchor and
            any other sample not belonging to the same label group
    """
    # triplet is a 3-tuple of anchor, positive, negative samples
    # with shape (B, C) and corresponding labels have shape (B)
    anchor = triplet[0]
    anchor_label = label[0]
    triplet = torch.cat(triplet)  # shape (3B, C)
    label = torch.cat(label)  # shape (3B)

    # compute pairwise distance between anchor and triplet
    anchor = rearrange(anchor, "b c -> b c 1")
    triplet = rearrange(triplet, "n c -> 1 c n")  # N == 3B
    dist = distance(anchor, triplet)  # shape (B, 3B)

    # compute pairwise positive distance
    positive_distance = torch.diag(
        rearrange(dist, "b (t b1) -> b t b1", t=3)[:, 1]  # B1 == B
    )

    # compute pairwise boolean for identifying samples belonging to same group
    anchor_label = rearrange(anchor_label, "b -> b 1")
    label = rearrange(label, "n -> 1 n")  # N == 3B
    same_label = anchor_label == label  # shape (B, 3B)

    # find hardest negative distance
    inf = torch.tensor(float("inf")).to(anchor.device)
    if mode == "min":
        updated_distance = torch.where(same_label, inf, dist)
        hardest_negative_distance = torch.min(updated_distance, dim=1)[0]
    else:
        updated_distance = torch.where(same_label, -inf, dist)
        hardest_negative_distance = torch.max(updated_distance, dim=1)[0]

    # used for checking
    if DEBUG:
        print(f"Original distance:\n{dist}")
        print(f"Distance after filtering positives:\n{updated_distance}")
        print(f"Positive distance:\n{positive_distance}")
        print(f"Hardest negative distance:\n{hardest_negative_distance}")

    return positive_distance, hardest_negative_distance


if __name__ == "__main__":
    # check return values on a simple case
    import torch.nn.functional as F

    device = "cuda" if torch.cuda.is_available() else "cpu"
    DEBUG = True

    B = 2
    C = 256
    triplet = tuple(torch.randn((B, C)).to(device) for i in range(3))
    label = tuple(
        torch.tensor(_label).to(device) for _label in [[0, 1], [0, 1], [2, 0]]
    )
    distance = lambda x, y: 1 - F.cosine_similarity(x, y)

    hardest_triplet(triplet, label, distance, mode="min")
