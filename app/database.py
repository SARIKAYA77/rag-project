import redis
import numpy as np
from numpy.linalg import norm

redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)


def index_embeddings(chunks: list, embeddings: list):
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        redis_client.set(f"chunk_{i}", chunk)
        redis_client.set(f"embedding_{i}", embedding.astype(np.float32).tobytes())


def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))


def search_embeddings(chunks: list, embeddings: list, query_embedding: np.ndarray):
    similarities = []
    for i, embedding in enumerate(embeddings):
        embedding = np.frombuffer(embedding, dtype=np.float32)
        similarity = cosine_similarity(embedding, query_embedding)
        similarities.append((similarity, chunks[i]))
    similarities.sort(reverse=True, key=lambda x: x[0])

    results = [
        {"chunk": chunk, "distance": 1 - similarity}
        for similarity, chunk in similarities[:5]
    ]

    return results
