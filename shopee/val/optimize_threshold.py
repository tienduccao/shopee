import random
import sys
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from .compute_f1_score import compute_f1_score


def optimize_threshold(
    data: pd.DataFrame,
    embeddings: np.ndarray,
    dataset_size: int = 70_000,
    output_figure: Optional[str] = None,
) -> float:
    """
    Estimate optimal threshold for a given dataset_size.

    Args:
        data: shopee dataframe
        embeddings: array of shape (N, C) where N is number of samples and C
            embeddings dimension
        dataset_size: target size
        output_figure: if provided, path to the file where to save the figure

    Returns:
        optimal threshold value (TODO: could add support for np.ndarray giving
            sizes and returning several thresholds at once)
    """
    # sort by label_group for avoiding splitting the groups
    # when subsampling the dataset (only the last group is split
    # at worse)
    data = data.sort_values(by=["label_group"])
    labels = np.array(data.label_group)
    embeddings = embeddings[data.index]

    thresholds = np.linspace(0.0, 1.0, num=1001)

    # dataset_sizes to be evaluated
    n_samples = len(embeddings)
    start = 8000
    step = 2000
    dataset_sizes = list(range(start, n_samples, step))

    # compute best F1 score and threshold for each dataset_size
    best_thresholds = []
    best_scores = []
    for size in dataset_sizes:
        print(f"Computing best F1 score for {size} samples")
        indices = list(range(size))
        scores = compute_f1_score(
            embeddings[indices], labels[indices], thresholds=thresholds
        )
        best = np.argmax(scores)
        best_thresholds.append(thresholds[best])
        best_scores.append(scores[best])

    # fit linear regression
    x = dataset_sizes
    y = best_thresholds
    x = np.array(x).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)

    reg = LinearRegression()
    reg.fit(x, y)
    r2 = reg.score(x, y)
    expected_value = float(reg.predict(np.array([[dataset_size]])))

    if output_figure is not None:

        fig = plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(x, y, ".")
        plt.plot(x, reg.predict(x), "r")
        plt.legend(
            [
                "points",
                f"reg:\ncoeff = {float(reg.coef_):0.4e},\n"
                + f"intercept = {float(reg.intercept_):0.2f}\n"
                + f"r2 = {r2:0.4f}\n"
                + f"expected at {dataset_size}: {expected_value:0.2f}",
            ]
        )
        plt.xlabel("dataset size")
        plt.ylabel("best threshold")

        plt.subplot(1, 2, 2)
        plt.plot(dataset_sizes, best_scores, ".")
        plt.xlabel("dataset size")
        plt.ylabel("best F1 score")

        plt.savefig(output_figure)

    return expected_value


if __name__ == "__main__":

    data = pd.read_csv("../shopee-product-matching/train.csv")

    path = "/opt/jsk/share/shopee-models/image/embeddings/image_embeddings_efficientnet_v2s.npy"
    path = "/opt/jsk/share/shopee-models/image/embeddings/image_embeddings_efficientnet_v2s_finetuned_arcface_triplet_model_00.npy"
    embeddings = np.load(path)

    output_figure = "f1.png"

    optimize_threshold(data, embeddings, output_figure=output_figure)