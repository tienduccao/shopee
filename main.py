import gc

import text, image, search
from data import COMPUTE_CV, test

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0


LIMIT = 1
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [
                tf.config.experimental.VirtualDeviceConfiguration(
                    memory_limit=1024 * LIMIT
                )
            ],
        )
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
    except RuntimeError as e:
        print(e)


COMPUTE_CV = True
TEST_MEMORY_ERROR = True
QUICK_TEST = False
TEST_DATA_SIZE = 1024 * 4


if COMPUTE_CV:
    test = pd.read_csv('../input/shopee-product-matching/train.csv')

    if QUICK_TEST:
        test = test[:TEST_DATA_SIZE]

    if TEST_MEMORY_ERROR:
        test = pd.concat((test, test))
#         test.title = test.title.apply(lambda title: title + ' ' + " text " * 100)

    print('Using train as test to compute CV (since commit notebook). Shape is', test.shape )
else:
    test = pd.read_csv('../input/shopee-product-matching/test.csv')
    print('Test shape is', test.shape )


def getMetric(col):
    def f1score(row):
        n = len(np.intersect1d(row.target, row[col]))
        return 2 * n / (len(row.target) + len(row[col]))

    return f1score


def combine_for_sub(row):
    x = np.concatenate([row.preds, row.preds2, row.preds3])
    return " ".join(np.unique(x))


def combine_for_cv(row):
    x = np.concatenate([row.preds, row.preds2, row.preds3])
    return np.unique(x)


def generate_submission(test):
    test["matches"] = test.apply(combine_for_sub, axis=1)
    test[["posting_id", "matches"]].to_csv("submission.csv", index=False)


def pipeline():
    WGT = "../input/effnetb0/efficientnetb0_notop.h5"
    model = EfficientNetB0(weights=WGT, include_top=False, pooling="avg", input_shape=None)
    image_embeddings = image.get_embeddings(test, model)
    preds = search.nearest_neighbors(test, image_embeddings, 0.1)
    test['preds2'] = preds
    del preds
    gc.collect()

    from transformers import DistilBertTokenizer, DistilBertModel
    model_name='../input/distilbert-base-indonesian'
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    bert_model = DistilBertModel.from_pretrained(model_name).cuda()
    text_embeddings = text.get_embeddings(test, bert_model, tokenizer, 20)
    preds = search.nearest_neighbors(test, text_embeddings, 0.3)
    test['preds'] = preds
    del preds
    gc.collect()

    tmp = test.groupby('image_phash').posting_id.agg('unique').to_dict()
    test['preds3'] = test.image_phash.map(tmp)

    if COMPUTE_CV:
        tmp = test.groupby("label_group").posting_id.agg("unique").to_dict()
        test["target"] = test.label_group.map(tmp)
        test["oof"] = test.apply(combine_for_cv, axis=1)
        test["f1"] = test.apply(getMetric("oof"), axis=1)
        print("CV Score =", test.f1.mean())

    generate_submission()
