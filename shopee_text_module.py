import os
os.system('pip install sentence_transformers')

import gc
import itertools
import random
random.seed(0)

import torch
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util, InputExample, losses
from torch.utils.data import DataLoader


def get_sentence_transformer_embeddings(test, model, batch_size):
    num_batches = test.shape[0] // batch_size
    if num_batches * batch_size < test.shape[0]:
        num_batches += 1

    with torch.no_grad():
        list_text_embeddings = list()
        titles = test.title.fillna("").tolist()
        for batch_id in tqdm(
            range(num_batches), total=num_batches, desc="Text embeddings"
        ):
            batch = titles[batch_id * batch_size : (batch_id + 1) * batch_size]
            output = model.encode(batch, convert_to_tensor=True, show_progress_bar=False).detach().cpu().numpy().tolist()
            list_text_embeddings.extend(output)

    text_embeddings = np.asarray(list_text_embeddings)

    del list_text_embeddings, model
    gc.collect()

    torch.cuda.empty_cache()

    print("text embeddings shape", text_embeddings.shape)

    return text_embeddings


def get_embeddings(test, transformer_model, tokenizer, max_length):
    batch_size = 1
    num_batches = test.shape[0] // batch_size
    if num_batches * batch_size < test.shape[0]:
        num_batches += 1

    with torch.no_grad():
        list_text_embeddings = list()
        titles = test.title.fillna("").tolist()
        for batch_id in tqdm(
            range(num_batches), total=num_batches, desc="Text embeddings"
        ):
            batch = titles[batch_id * batch_size : (batch_id + 1) * batch_size]
            encoded_input = tokenizer(
                batch, return_tensors="pt", max_length=max_length, truncation=True
            )
            output = transformer_model(**encoded_input.to("cuda"))
            list_text_embeddings.extend(
                output.last_hidden_state.cpu()
                .detach()
                .numpy()[0]
                .mean(0)
                .reshape(1, 768)
            )

    text_embeddings = np.asarray(list_text_embeddings)

    del list_text_embeddings, transformer_model
    gc.collect()

    torch.cuda.empty_cache()

    print("text embeddings shape", text_embeddings.shape)

    return text_embeddings


def similarities(model, pairs):
    s1 = [pair[0].lower() for pair in pairs]
    s2 = [pair[1].lower() for pair in pairs]
    embeddings1 = model.encode(s1, convert_to_tensor=True)
    embeddings2 = model.encode(s2, convert_to_tensor=True)

    for e1, e2 in tqdm(zip(embeddings1, embeddings2)):
        yield util.pytorch_cos_sim(e1, e2).cpu().numpy()[0][0]


def evaluate(model, s1, s2):
    embeddings1 = model.encode(s1, convert_to_tensor=True)
    embeddings2 = model.encode(s2, convert_to_tensor=True)

    for e1, e2 in tqdm(zip(embeddings1, embeddings2)):
        yield util.pytorch_cos_sim(e1, e2).cpu().numpy()[0][0]


def fine_tune_sentence_transformers(train_df):
    model = SentenceTransformer("distiluse-base-multilingual-cased-v2").cuda()

    similar_titles = train_df.groupby("label_group")["title"].size()

    examples = list()
    for size in range(2, 8 + 1):
        titles = (
            train_df[
                train_df.label_group.isin(similar_titles[similar_titles == size].index)
            ]
            .groupby("label_group")
            .title.apply(list)
            .reset_index()
            .title
        )
        title_pairs = list()
        for title in titles:
            title_pairs.extend(list(itertools.combinations(title, 2)))

        sim = list(similarities(model, title_pairs))
        threshold = np.quantile(sim, 0.3)

        for s, pair in zip(sim, title_pairs):
            if s <= threshold:
                examples.append(InputExample(texts=pair, label=1))

    random.shuffle(examples)
    cut_id = int(len(examples) * 0.8)
    train_examples = examples[: cut_id]
    val_examples = examples[cut_id: ]

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = losses.ContrastiveLoss(model)

    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=5, warmup_steps=100)

    s1, s2 = list(), list()
    for e in val_examples:
        s1.append(e.texts[0])
        s2.append(e.texts[1])

    new_sim = list(evaluate(model, s1, s2))
    return np.min(new_sim), np.median(new_sim)

