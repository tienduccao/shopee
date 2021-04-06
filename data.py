import pandas as pd


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
