from openai import OpenAI
import numpy as np


client = OpenAI()


def embed_text(text: str):
    response = client.embeddings.create(
        model="text-embedding-3-large", 
        input=text
    )    
    return response.data[0].embedding


def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))