
import torch
from transformers import AutoTokenizer, AutoModel

def load_model():
    tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-uncased")
    model = AutoModel.from_pretrained("dbmdz/bert-base-turkish-uncased")
    return tokenizer, model

def embed_text(chunks: list) -> list:
    tokenizer, model = load_model()
    embeddings = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        embeddings.append(embedding)
    return embeddings
