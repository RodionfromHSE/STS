import torch
from torch.nn.functional import cosine_similarity


def get_embeddings(model, tokenizer, sentences, no_grad=True):
    # tokenize
    tokenized = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

    with torch.set_grad_enabled(not no_grad):
        embeddings = model(**tokenized).last_hidden_state.mean(dim=1)
    return embeddings


def get_score(model, tokenizer, sample):
    emb_a, emb_b = get_embeddings(model, tokenizer, [sample["text1"], sample["text2"]])
    return cosine_similarity(emb_a.unsqueeze(0), emb_b.unsqueeze(0)).item()
