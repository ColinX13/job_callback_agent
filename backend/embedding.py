from sentence_transformers import SentenceTransformer
import numpy as np

# free local model since no Groq embedding model available
model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_text(text: str):
    if not text:
        return np.zeros((384, ))  # fallback for empty text
    embedding = model.encode(text)
    return embedding.tolist()

def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))