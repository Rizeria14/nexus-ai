from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
import shutil, os, json, requests

from pdf_utils import extract_text_from_pdf
from chunking import chunk_text
from embeddings import get_embedding
from vector_store_faiss import VectorStoreFAISS

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = "data"
CHAT_DIR = "chats"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHAT_DIR, exist_ok=True)


# =====================
# HELPERS
# =====================

def sanitize(name):
    return name.replace(".pdf", "").replace(" ", "_")

def get_store(doc_id):
    path = os.path.join(DATA_DIR, doc_id)
    store = VectorStoreFAISS(dim=384)
    if os.path.exists(os.path.join(path, "faiss.index")):
        store.load(path)
    return store


# =====================
# DOCUMENT APIs
# =====================

@app.get("/documents")
def list_docs():
    return {"documents": os.listdir(DATA_DIR)}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    doc_id = sanitize(file.filename)
    path = os.path.join(DATA_DIR, doc_id)
    os.makedirs(path, exist_ok=True)

    file_path = os.path.join(path, "doc.pdf")

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    text = extract_text_from_pdf(file_path)

    chunks = chunk_text(text, 600, 150)
    embeddings = [get_embedding(c) for c in chunks]

    store = VectorStoreFAISS(dim=384)
    store.add(embeddings, chunks)
    store.save(path)

    return {"doc_id": doc_id}


@app.delete("/document")
def delete_doc(doc_id: str):
    path = os.path.join(DATA_DIR, doc_id)
    if not os.path.exists(path):
        return {"error": "Not found"}
    shutil.rmtree(path)
    return {"message": "deleted"}


@app.put("/document")
def rename_doc(old_id: str, new_id: str):
    os.rename(
        os.path.join(DATA_DIR, old_id),
        os.path.join(DATA_DIR, new_id)
    )
    return {"message": "renamed"}


# =====================
# CHAT APIs
# =====================

@app.post("/chat")
def create_chat(doc_id: str):
    chat_id = f"chat_{len(os.listdir(CHAT_DIR)) + 1}"

    data = {
        "chat_id": chat_id,
        "doc_id": doc_id,
        "messages": []
    }

    with open(f"{CHAT_DIR}/{chat_id}.json", "w") as f:
        json.dump(data, f)

    return data


@app.get("/chats")
def list_chats():
    return {"chats": os.listdir(CHAT_DIR)}


@app.get("/chat")
def load_chat(chat_id: str):
    with open(f"{CHAT_DIR}/{chat_id}") as f:
        return json.load(f)


# =====================
# ASK (CORE)
# =====================

@app.get("/ask")
def ask(query: str, doc_id: str = "", mode: str = "single", chat_id: str = ""):

    all_chunks = []

    if mode == "multi":
        for doc in os.listdir(DATA_DIR):
            store = get_store(doc)
            emb = get_embedding(query)
            results = store.search(emb, k=5)
            all_chunks.extend(results)
    else:
        store = get_store(doc_id)
        emb = get_embedding(query)
        all_chunks = store.search(emb, k=10)

    # simple rerank
    def score(q, t):
        return sum(word in t.lower() for word in q.lower().split())

    all_chunks = sorted(all_chunks, key=lambda x: score(query, x), reverse=True)[:6]

    context = "\n\n---\n\n".join(all_chunks)

    prompt = f"""
You are an expert AI assistant.

Answer ONLY from context.
If not found, say: Not found in document.

Context:
{context}

Question:
{query}

Answer:
"""

    res = requests.post(
        "http://127.0.0.1:11434/api/generate",
        json={"model": "mistral", "prompt": prompt, "stream": False}
    )

    answer = res.json().get("response", "")

    # save chat
    if chat_id:
        file = f"{CHAT_DIR}/{chat_id}.json"
        if os.path.exists(file):
            with open(file) as f:
                chat = json.load(f)

            chat["messages"].append({"role": "user", "content": query})
            chat["messages"].append({"role": "bot", "content": answer})

            with open(file, "w") as f:
                json.dump(chat, f)

    return {"answer": answer}