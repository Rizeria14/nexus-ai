import faiss
import numpy as np
import pickle
import os


class VectorStoreFAISS:
    def __init__(self, dim):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.texts = []

    def add(self, embeddings, texts):
        vectors = np.array(embeddings).astype("float32")
        self.index.add(vectors)
        self.texts.extend(texts)

    def search(self, query_embedding, k=5):
        if len(self.texts) == 0:
            return []

        query = np.array([query_embedding]).astype("float32")
        distances, indices = self.index.search(query, k)

        results = []
        for i in indices[0]:
            if i < len(self.texts):  # safety check
                results.append(self.texts[i])

        return results

    def save(self, path):
        os.makedirs(path, exist_ok=True)

        faiss.write_index(self.index, os.path.join(path, "faiss.index"))

        with open(os.path.join(path, "texts.pkl"), "wb") as f:
            pickle.dump(self.texts, f)

    def load(self, path):
        index_path = os.path.join(path, "faiss.index")
        text_path = os.path.join(path, "texts.pkl")

        if not os.path.exists(index_path) or not os.path.exists(text_path):
            raise FileNotFoundError("Vector store files not found")

        self.index = faiss.read_index(index_path)

        with open(text_path, "rb") as f:
            self.texts = pickle.load(f)