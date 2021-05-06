from collections import defaultdict
from typing import Optional, Tuple

from einops import rearrange, reduce
import numpy as np

from .compute_knn import compute_knn


def compute_f1_score(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_neighbors: int = 50,
    thresholds: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """
    Compute F1 score for each threshold by using cosine distance

     Args:
        embeddings: array of shape (N, C) where N is number of samples and C
            embeddings dimension
        labels: array of shape (N) containing label_group for each sample
        n_neighbors: number of neighbors to compute
        thresholds: thresholds for which F1 score should be computed

    Returns:
        scores: F1 score values for each threshold for these embeddings
    """
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, num=101)
    knn_scores, knn_indices = compute_knn(embeddings, n_neighbors)
    knn_labels = labels[knn_indices]

    counts = defaultdict(lambda: 0)
    for label in labels:
        counts[label] += 1
    labels_counts = np.array([counts[label] for label in labels])

    scores = _compute_f1_score(
        knn_scores,
        knn_labels,
        labels,
        labels_counts,
        thresholds,
    )

    # best = np.argmax(scores)
    # best_score = scores[best]
    # best_threshold = thresholds[best]

    return scores


def _compute_f1_score(
    knn_scores, knn_labels, labels, labels_counts, thresholds, verbose=False
):
    """
    Compute f1 score for each combination of sample x nearest_neighbor x threshold

    Args:
        knn_scores: tensor of shape (N, K) containing the score for each
            combination sample x nearest neighbour
            TODO: currently lower score means sample and neighbor are
                closer, will add option mode with 'min'/'max' choices
        knn_labels: tensor of shape (N, K) providing label group for each
            nearest neighbour of a combination sample x nearest neighbour
        labels_counts: tensor of shape (N) giving the size of the group
            each sample belongs to
        thresholds: tensor of shape (T) corresponding to the thresholds
            values to be evaluated
        verbose: if True, show intermediate values (for debugging)
    """
    # reshape arrays for computing each combination
    # sample x nearest_neighbor x threshold (shape (n, k, t))
    thresholds = rearrange(thresholds, "t -> 1 1 t")
    knn_scores = rearrange(knn_scores, "n k -> n k 1")
    knn_labels = rearrange(knn_labels, "n k -> n k 1")
    labels = rearrange(labels, "n -> n 1 1")
    labels_counts = rearrange(labels_counts, "n -> n 1")

    # compute f1 score for each couple sample x threshold
    # TODO: add mode "max" for reverting comparaison
    preds = knn_scores <= thresholds  # shape (n, k, t)
    preds_counts = reduce(preds, "n k t -> n t", "sum")
    corrects = knn_labels == labels  # shape (n, k, 1)
    intersection_counts = reduce(preds & corrects, "n k t -> n t", "sum")
    sample_f1_scores = (
        2 * intersection_counts / (labels_counts + preds_counts)
    )  # shape (n, t)

    # average over samples
    f1_scores = reduce(sample_f1_scores, "n t -> t", "mean")

    if verbose:
        print(
            knn_scores.shape,
            knn_labels.shape,
            labels.shape,
            thresholds.shape,
        )
        print(f"preds\n{1*preds}")
        print(f"corrects\n{1*corrects}")
        print(f"preds_count\n{preds_counts}")
        print(f"intersection\n{intersection_counts}")
        print(f"f1\n{sample_f1_scores}")

    return f1_scores


if __name__ == "__main__":

    import numpy as np

    labels = [0, 1, 1]
    knn_scores = [
        [0, 0.5, 0.3],
        [0.5, 0, 0.2],
        [0.6, 0.1, 0],
    ]
    knn_labels = [labels] * 3
    labels_counts = [0] * 2
    for label in labels:
        labels_counts[label] += 1

    labels = np.array(labels)
    knn_scores = np.array(knn_scores)
    knn_labels = np.array(knn_labels)
    labels_counts = np.array(labels_counts)
    labels_counts = labels_counts[labels]
    # thresholds = np.linspace(0, 1, num=101)
    thresholds = np.sort(np.unique(knn_scores))

    scores = _compute_f1_score(
        knn_scores, knn_labels, labels, labels_counts, thresholds, verbose=True
    )
