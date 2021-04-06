import gc

import cupy
import torch
from tqdm import tqdm


def get_embeddings(test, transformer_model, tokenizer, max_length):
    batch_size = 1
    num_batches = test.shape[0] // batch_size
    if num_batches * batch_size < test.shape[0]:
        num_batches += 1

    with torch.no_grad():
        list_text_embeddings = list()
        titles = test.title.fillna('').tolist()
        for batch_id in tqdm(range(num_batches), total=num_batches, desc='Text embeddings'):
            batch = titles[batch_id * batch_size: (batch_id + 1) * batch_size]
            encoded_input = tokenizer(batch, return_tensors='pt', max_length=max_length, truncation=True)
            output = transformer_model(**encoded_input.to('cuda'))
            list_text_embeddings.extend(output.last_hidden_state.cpu().detach().numpy()[0].mean(0).reshape(1, 768))

    text_embeddings = cupy.asarray(list_text_embeddings)

    del list_text_embeddings, transformer_model
    gc.collect()

    torch.cuda.empty_cache()

    print('text embeddings shape',text_embeddings.shape)

    return text_embeddings
