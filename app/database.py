import redis
import numpy as np
from numpy.linalg import norm

# Redis bağlantısı
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)


def index_embeddings(chunks: list, embeddings: list):
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        redis_client.set(f"chunk_{i}", chunk)
        redis_client.set(f"embedding_{i}", embedding.tobytes())  # Embedding'i byte dizisine çeviriyoruz


def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))


def search_embeddings(chunks: list, embeddings: list, query_embedding: np.ndarray):
    similarities = []

    for i, embedding in enumerate(embeddings):
        embedding = np.frombuffer(embedding, dtype=np.float32)  # Redis'ten byte olarak alınan embedding'i geri yükle
        similarity = cosine_similarity(embedding, query_embedding)
        similarities.append((similarity, chunks[i]))

    # Benzerliklere göre sıralayın ve en iyi 5 sonucu döndürün
    similarities.sort(reverse=True, key=lambda x: x[0])

    results = [
        {"chunk": chunk, "distance": 1 - similarity}
        # Distance = 1 - similarity (1'e ne kadar yakınsa o kadar az benzer)
        for similarity, chunk in similarities[:5]
    ]

    return results
