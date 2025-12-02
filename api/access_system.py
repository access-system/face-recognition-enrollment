import requests
import numpy as np

url = "http://localhost:8081/api/v1/"

def validate_embedding(embedding: np.ndarray):
    if embedding is None:
        raise ValueError("Embedding is None.")

    embedding_list = embedding.tolist()

    data = {
        "vector": embedding_list
    }
    response = requests.post(url + 'embedding/validate', json=data)

    if response.status_code == 200:
        return True, response.text
    else:
        return False, response.text


def add_embedding(embedding: np.ndarray, name: str) -> int:
    if embedding is None:
        raise ValueError("Embedding is None.")

    embedding_list = embedding.tolist()

    if len(embedding_list) != 512:
        raise ValueError("Embedding must be a list of 512 floats.")

    data = {
        "name": name,
        "vector": embedding_list
    }
    response = requests.post(url + "embedding", json=data)

    return response.status_code
