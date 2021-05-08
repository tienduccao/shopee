from typing import Tuple

from cuml.neighbors import NearestNeighbors
import numpy as np


def compute_knn(
    embeddings: np.ndarray, n_neighbors: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute KNN from embeddings by using chunks of rows and by using cosine
    distance.

    Args:
        embeddings: array of shape (N, C) where N is number of samples and C
            embeddings dimension
        n_neighbors: number of neighbors to compute

    Returns:
        knn_scores: array of shape (N, n_neighbors) containing the KNN
            distances (in increasing order, sample included)
        knn_indices: array of shape (N, n_neighbors) containing the indices of
            the KNN
    """

    model = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
    model.fit(embeddings)

    n_embeddings = embeddings.shape[0]

    knn_scores = np.zeros((n_embeddings, n_neighbors), dtype=np.float32)
    knn_indices = np.zeros((n_embeddings, n_neighbors), dtype=np.int32)

    chunks = 1024
    for i in range(0, n_embeddings, chunks):
        (
            knn_scores[i : i + chunks],
            knn_indices[i : i + chunks],
        ) = model.kneighbors(embeddings[i : i + chunks])

    return knn_scores, knn_indices
