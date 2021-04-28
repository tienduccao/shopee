import gc

import shopee_text_module as text, shopee_image_module as image, shopee_search_module as search

import torch
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from memory_profiler import memory_usage


MAX_MEMORY = 13 * 1024


def mem_check(func):
    def wrapper(*arg, **kwargs):
        max_mem, function_value = memory_usage(proc=(func, arg, kwargs), max_usage=True, retval=True)
        print(f"Max memory usage of {func.__name__} is {max_mem} MB")
        if max_mem > MAX_MEMORY:
            raise Exception(f'Max Memory Error {max_mem} vs {MAX_MEMORY}')
        return function_value

    return wrapper


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
TEST_MEMORY_ERROR = False
QUICK_TEST = False
TEST_DATA_SIZE = 1024 * 4


def getMetric(col):
    def f1score(row):
        n = len(np.intersect1d(row.target, row[col]))
        return 2 * n / (len(row.target) + len(row[col]))

    return f1score


def combine_for_sub(row):
    x = np.concatenate([row.preds, row.preds2, row.preds3])
    return " ".join(np.unique(x))


def combine_for_cv(row):
    x = np.concatenate([row.preds, row.preds3])
    return np.unique(x)


def generate_submission(test):
    test["matches"] = test.apply(combine_for_sub, axis=1)
    test[["posting_id", "matches"]].to_csv("submission.csv", index=False)


@mem_check
def predict_images(test):
    WGT = "../input/effnetb0/efficientnetb0_notop.h5"
    model = EfficientNetB0(weights=WGT, include_top=False, pooling="avg", input_shape=None)
    BASE = "../input/shopee-product-matching/test_images/"
    if COMPUTE_CV:
        BASE = "../input/shopee-product-matching/train_images/"
    image_embeddings = image.get_embeddings(BASE, test, model)
    preds = search.perfect_nearest_neighbors(test, image_embeddings, 0.2, 50, test.num_neighbors)
    test['preds2'] = preds
    del preds
    gc.collect()

    return test


@mem_check
def predict_texts(test):
    # model_name = '../input/fine-tuned-titles/fine_tuned_titles'
    # model = torch.load(model_name)
    # model.eval()
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('distiluse-base-multilingual-cased-v2').cuda()
    model.eval()
    text_embeddings = text.get_sentence_transformer_embeddings(test, model, 128)
    preds = search.perfect_nearest_neighbors(test, text_embeddings, 0.2, 50, test.num_neighbors)
    test['preds'] = preds
    del preds
    gc.collect()

    return test


@mem_check
def predict_images_and_texts(test):
    WGT = "../input/effnetb0/efficientnetb0_notop.h5"
    model = EfficientNetB0(weights=WGT, include_top=False, pooling="avg", input_shape=None)
    BASE = "../input/shopee-product-matching/test_images/"
    if COMPUTE_CV:
        BASE = "../input/shopee-product-matching/train_images/"
    image_embeddings = image.get_embeddings(BASE, test, model)
    del model
    gc.collect()

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('distiluse-base-multilingual-cased-v2').cuda()
    model.eval()
    text_embeddings = text.get_sentence_transformer_embeddings(test, model, 128)
    del model
    gc.collect()

    embeddings = np.concatenate((text_embeddings, image_embeddings), axis=1)
    del text_embeddings, image_embeddings
    gc.collect()

    preds = search.perfect_nearest_neighbors(test, embeddings, 0.2, 50, test.num_neighbors)
    test['preds'] = preds
    del preds, embeddings
    gc.collect()

    return test



BATCH = 5
def num_to_range(num):
    _range = num // BATCH
    if BATCH * _range < num:
        _range += 1
    return BATCH * _range


def pipeline(test):
    if COMPUTE_CV:
        tmp = test.groupby("label_group").posting_id.agg("unique").to_dict()
        test["target"] = test.label_group.map(tmp)

    # test["num_neighbors"] = test.target.apply(lambda target: len(target))
    test["num_neighbors"] = test.target.apply(lambda target: num_to_range(len(target)))

    # test = predict_images(test)
    # test = predict_texts(test)
    test = predict_images_and_texts(test)

    tmp = test.groupby('image_phash').posting_id.agg('unique').to_dict()
    test['preds3'] = test.image_phash.map(tmp)

    if COMPUTE_CV:
        test["oof"] = test.apply(combine_for_cv, axis=1)
        test["f1"] = test.apply(getMetric("oof"), axis=1)
        print("CV Score =", test.f1.mean())

    # generate_submission(test)


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

pipeline(test)
