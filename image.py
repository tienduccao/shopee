import gc

import cv2
import numpy as np
import tensorflow as tf


class DataGenerator(tf.keras.utils.Sequence):
    "Generates data for Keras"

    def __init__(self, df, img_size=256, batch_size=32, path=""):
        self.df = df
        self.img_size = img_size
        self.batch_size = batch_size
        self.path = path
        self.indexes = np.arange(len(self.df))

    def __len__(self):
        "Denotes the number of batches per epoch"
        ct = len(self.df) // self.batch_size
        ct += int(((len(self.df)) % self.batch_size) != 0)
        return ct

    def __getitem__(self, index):
        "Generate one batch of data"
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        X = self.__data_generation(indexes)
        return X

    def __data_generation(self, indexes):
        "Generates data containing batch_size samples"
        X = np.zeros((len(indexes), self.img_size, self.img_size, 3), dtype="float32")
        df = self.df.iloc[indexes]
        for i, (index, row) in enumerate(df.iterrows()):
            img = cv2.imread(self.path + row.image)
            X[i,] = cv2.resize(
                img, (self.img_size, self.img_size)
            )  # /128.0 - 1.0
        return X


def get_embeddings(test, model, CHUNK=4096, batch_size=32):
    BASE = "../input/shopee-product-matching/test_images/"
    if COMPUTE_CV:
        BASE = "../input/shopee-product-matching/train_images/"

    # WGT = "../input/effnetb0/efficientnetb0_notop.h5"
    # model = EfficientNetB0(weights=WGT, include_top=False, pooling="avg", input_shape=None)

    embeds = []

    print("Computing image embeddings...")
    CTS = len(test) // CHUNK
    if len(test) % CHUNK != 0:
        CTS += 1

    for i, j in enumerate(range(CTS)):

        a = j * CHUNK
        b = (j + 1) * CHUNK
        b = min(b, len(test))
        print("chunk", a, "to", b)

        test_gen = DataGenerator(test.iloc[a:b], batch_size=batch_size, path=BASE)
        image_embeddings = model.predict(
            test_gen, verbose=1, use_multiprocessing=True, workers=4
        )
        embeds.append(image_embeddings)

        # if i>=1: break

    del model, embeds
    _ = gc.collect()
    image_embeddings = np.concatenate(embeds)
    print("image embeddings shape", image_embeddings.shape)

    return image_embeddings
