import sys
# ChromaDB SQLite compatibility monkeypatch for Linux environments (like Streamlit Cloud)
if sys.platform == "linux":
    try:
        import pysqlite3
        sys.modules["sqlite3"] = pysqlite3
    except ImportError:
        pass

import os
import hashlib
import chromadb

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.path.join(BASE_DIR, "chroma_db")

# Ensure DB directory exists
os.makedirs(DB_DIR, exist_ok=True)

# Initialize ChromaDB persistent client
client = chromadb.PersistentClient(path=DB_DIR)

class HashingEmbeddingFunction(chromadb.EmbeddingFunction):
    """
    A lightweight, pure-Python, zero-dependency embedding function.
    It hashes words into a 128-dimensional space using MD5 and normalizes the vectors.
    This prevents memory exhaustion (OOM) and model download timeouts on Streamlit Cloud.
    """
    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        embeddings = []
        for text in input:
            vector = [0.0] * 128
            words = text.lower().split()
            if not words:
                embeddings.append(vector)
                continue
            
            for word in words:
                # Hash each word into an index between 0 and 127
                h = int(hashlib.md5(word.encode('utf-8')).hexdigest(), 16)
                index = h % 128
                vector[index] += 1.0
                
            # Normalize vector (L2 norm)
            norm = sum(x**2 for x in vector) ** 0.5
            if norm > 0:
                vector = [x / norm for x in vector]
                
            embeddings.append(vector)
        return embeddings

# Initialize hashing embedding function
embedding_fn = HashingEmbeddingFunction()

# Get or create the collection (using v2 to prevent dimension mismatch with older runs)
collection = client.get_or_create_collection(
    name="autoresearch_docs_v2",
    embedding_function=embedding_fn
)

def add_chunks(chunks: list[str], metadatas: list[dict], ids: list[str]):
    """
    Adds text chunks, metadata, and unique IDs to the ChromaDB collection.
    
    Args:
        chunks (list[str]): List of text chunks to add.
        metadatas (list[dict]): List of metadata dictionaries per chunk.
        ids (list[str]): List of unique string IDs for the chunks.
    """
    if not chunks:
        return
    collection.add(
        documents=chunks,
        metadatas=metadatas,
        ids=ids
    )

def retrieve(query: str, n: int = 5) -> list[dict]:
    """
    Queries the ChromaDB collection and returns top N matching documents.
    
    Args:
        query (str): The search query.
        n (int): The number of chunks to return.
        
    Returns:
        list[dict]: List of dictionaries containing content, metadata, id, and distance.
    """
    if not query:
        return []
        
    # Get total document count to prevent querying more than available
    count = collection.count()
    if count == 0:
        return []
    
    n_results = min(n, count)
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    
    retrieved = []
    if results and results.get("documents") and len(results["documents"]) > 0:
        documents = results["documents"][0]
        metadatas = results["metadatas"][0] if results.get("metadatas") else [{} for _ in documents]
        ids = results["ids"][0] if results.get("ids") else [str(i) for i, _ in enumerate(documents)]
        distances = results["distances"][0] if results.get("distances") else [0.0 for _ in documents]
        
        for doc, meta, idx, dist in zip(documents, metadatas, ids, distances):
            retrieved.append({
                "content": doc,
                "metadata": meta,
                "id": idx,
                "distance": dist
            })
            
    return retrieved

def clear_collection():
    """
    Clears all items in the autoresearch_docs_v2 collection.
    """
    global collection
    try:
        client.delete_collection("autoresearch_docs_v2")
    except Exception:
        pass
    collection = client.get_or_create_collection(
        name="autoresearch_docs_v2",
        embedding_function=embedding_fn
    )
