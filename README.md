# References
+ Competition link https://www.kaggle.com/c/shopee-product-matching
+ Trello board https://trello.com/b/EaxLChyV/shopee
+ Baselines from other participants
    + [RAPIDS tf-idf + EfficientNetB0](https://www.kaggle.com/cdeotte/part-2-rapids-tfidfvectorizer-cv-0-700)
    + [BERT + EfficientNetB3](https://www.kaggle.com/ragnar123/unsupervised-baseline-arcface/)
+ EDA from other participants
    + https://www.kaggle.com/ruchi798/shopee-eda-rapids-preprocessing

# Workflow
## Install the kaggle command line
+ https://github.com/Kaggle/kaggle-api
## Local
+ Add new functions to `shopee_text.py`, `shopee_image.py`, `shopee_search.py`
+ Modify `main.py` in order incorporate new functions if needed
    + Set `COMPUTE_CV = True` if you want to compute the score on the training set
    + Set `TEST_MEMORY_ERROR = True` to test your code with the training set x2
    + Set `QUICK_TEST = True` to test your code with a dataset of 4096 rows
+ Commit and push your changes with git
## Push new functions to Kaggle
+ `kaggle kernels push -p kaggle_scripts/config/text` if your made only changes to `shopee_text_module.py`.
+ `kaggle kernels push -p kaggle_scripts/config/image` if your made only changes to `shopee_image_module.py`.
+ `kaggle kernels push -p kaggle_scripts/config/search` if your made only changes to `shopee_search_module.py`.
+ `./upload_utility_scripts.sh` if you made changes to 3 modules.
+ You should wait a bit (max 30 seconds) to have your new versions available in Kaggle.
## Push your new main.py to Kaggle
+ Edit `main.py`
    + Set `COMPUTE_VC = False`
    + Set `TEST_MEMORY_ERROR = False`
    + Set `QUICK_TEST = False`
+ Edit `kernel-metadata.json`
    + Set an appropriate name for `id`, replace `duccao` with your username
    + `title` is the same as `id` (but having `<username>/` removed)
    + Add new datasets to `dataset_sources` if needed
+ `kaggle kernels push`

# Resources
+ Indonesian pre-trained models https://www.kaggle.com/liuhh02/datasets

# Ideas
## Unsupervisied
+ FAISS index
+ Title clusters using SentenceTransformers

## Supervised
+ Study multimodal models
+ Identify suitable loss function(s)

## Pre-trained
+ Indonesian pre-trained language model (need to verify whether most of the titles are Indonesian)
    + English, Indonesian, and Malaysian https://trello.com/c/9t1IRWsw/2-identify-the-main-languages-from-product-titles

## Fine-tuning
+ SentenceTransformers on pairs of titles from the same label_group

## Experiment management
+ MLFlow 

# EDA

## Train set labels
+ [Remarks]
    + **/!\\** 53% of target have length equal to 51 -> there are 50 duplicates + row itself
+ Baseline scores:
    + [Each row in its own group]
    ```
    F1 = 0.4608
    Precision:  1.0000
    Recall:  0.3216
    ```
    + [Duplicates based on phash]
    ```
    F1 = 0.5531
    Precision:  0.9941
    Recall:  0.4222
    ```
    phash can be used as a first filter (<1% False Positives)

## Text 
+ [TEXT]
    + [BIG SALE] Timbangan Dapur Digital Kitchen Scale SF-400 Murah
    + [Mizan] The Montessori Toddler
    + [ORIGINAL] MS Glow Paket Wajah
+ Product models, e.g., "Tempered Glass OPPO A37/F1S/A3S/A5S/A7/A1K/A5 2020/A9 2020/A83/A57/A39/A71/F5/F7/F9/F11/F11 PRO"
+ Brands, e.g., Xiaomi in "Xiaomi Redmi Note 9 9s 8 7 10X Pro 5G 4G Camera Lens Tempered Glass Screen Protector"
+ Units, e.g., 50 Sheets in "Bantex Loose Leaf Paper B5 80 gsm 50 Sheets - 26 Holes 8600 00"
