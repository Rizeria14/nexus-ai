import faiss
import numpy as np

class VectorStore:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)
        self.text_chunks = []

    def add(self, embeddings, chunks):
        self.index.add(np.array(embeddings).astype("float32"))
        self.text_chunks.extend(chunks)

    def search(self, query_embedding, k=6):
        distances, indices = self.index.search(
            np.array([query_embedding]).astype("float32"), k
        )

        results = []
        for i in indices[0]:
            if i < len(self.text_chunks):
                results.append(self.text_chunks[i])

        return results