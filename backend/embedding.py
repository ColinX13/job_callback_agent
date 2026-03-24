from fastembed import TextEmbedding
import numpy as np

# CPU-only model, no pytorch/CUDA required
model = TextEmbedding(model_name='BAAI/bge-small-en-v1.5')

def embed_text(text: str):
    if not text:
        return np.zeros((384, )).tolist()  # fallback for empty text
    embedding = next(model.embed([text]))
    return embedding.tolist()

def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))