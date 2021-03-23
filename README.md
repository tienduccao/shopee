# References
+ Competition link https://www.kaggle.com/c/shopee-product-matching
+ Trello board https://trello.com/b/EaxLChyV/shopee
+ Baselines from other participants
    + [RAPIDS tf-idf + EfficientNetB0](https://www.kaggle.com/cdeotte/part-2-rapids-tfidfvectorizer-cv-0-700)
    + [BERT + EfficientNetB3](https://www.kaggle.com/ragnar123/unsupervised-baseline-arcface/)
+ EDA from other participants
    + https://www.kaggle.com/ruchi798/shopee-eda-rapids-preprocessing

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
## Text 
+ [TEXT]
    + [BIG SALE] Timbangan Dapur Digital Kitchen Scale SF-400 Murah
    + [Mizan] The Montessori Toddler
    + [ORIGINAL] MS Glow Paket Wajah
+ Product models, e.g., "Tempered Glass OPPO A37/F1S/A3S/A5S/A7/A1K/A5 2020/A9 2020/A83/A57/A39/A71/F5/F7/F9/F11/F11 PRO"
+ Brands, e.g., Xiaomi in "Xiaomi Redmi Note 9 9s 8 7 10X Pro 5G 4G Camera Lens Tempered Glass Screen Protector"
+ Units, e.g., 50 Sheets in "Bantex Loose Leaf Paper B5 80 gsm 50 Sheets - 26 Holes 8600 00"
