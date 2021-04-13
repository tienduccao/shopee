import gc

import numpy as np
from cuml.neighbors import NearestNeighbors


def nearest_neighbors(test, embeddings, distance_threshold, KNN=50):
    if len(test) == 3:
        KNN = 2
    model = NearestNeighbors(n_neighbors=KNN, metric="cosine")
    model.fit(embeddings)

    preds = []

    CHUNK = 1024 * 4

    print("Finding similar products...")
    CTS = len(embeddings) // CHUNK
    if len(embeddings) % CHUNK != 0:
        CTS += 1

    for j in range(CTS):

        a = j * CHUNK
        b = (j + 1) * CHUNK
        b = min(b, len(embeddings))
        print("chunk", a, "to", b)
        distances, indices = model.kneighbors(
            embeddings[
                a:b,
            ]
        )

        for k in range(b - a):
            IDX = np.where(
                distances[
                    k,
                ]
                <= distance_threshold
            )[0]
            IDS = indices[k, IDX]
            o = test.iloc[IDS].posting_id.values
            preds.append(o)

    del model, distances, indices, embeddings
    _ = gc.collect()

    return preds


def perfect_nearest_neighbors(test, embeddings, distance_threshold, KNN, num_neighbors, CHUNK=1024 * 4):
    model = NearestNeighbors(n_neighbors=KNN, metric="cosine")
    model.fit(embeddings)

    preds = []

    print("Finding similar products...")
    CTS = len(embeddings) // CHUNK
    if len(embeddings) % CHUNK != 0:
        CTS += 1

    idx = 0
    for j in range(CTS):

        a = j * CHUNK
        b = (j + 1) * CHUNK
        b = min(b, len(embeddings))
        print("chunk", a, "to", b)
        distances, indices = model.kneighbors(
            embeddings[
                a:b,
            ]
        )

        for k in range(b - a):
            IDX = np.where(
                distances[
                    k,
                ][: num_neighbors[idx]]
                <= distance_threshold
            )[0]
            idx += 1
            IDS = indices[k, IDX]
            o = test.iloc[IDS].posting_id.values
            preds.append(o)

    del model, distances, indices, embeddings
    _ = gc.collect()

    return preds
