import os
import chromadb
from chromadb.utils import embedding_functions

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.path.join(BASE_DIR, "chroma_db")

# Ensure DB directory exists
os.makedirs(DB_DIR, exist_ok=True)

# Initialize ChromaDB persistent client
client = chromadb.PersistentClient(path=DB_DIR)

# Use default embedding function (sentence-transformers / onnx)
try:
    embedding_fn = embedding_functions.DefaultEmbeddingFunction()
except Exception as e:
    print(f"Warning: Failed to load DefaultEmbeddingFunction ({e}). Falling back to simple embedding.")
    # Fallback embedding function that returns dummy or simple random embeddings if system dependencies fail
    class FallbackEmbeddingFunction(chromadb.EmbeddingFunction):
        def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
            # Simple dummy embedding: list of 384-dim zero vectors
            return [[0.0] * 384 for _ in input]
    embedding_fn = FallbackEmbeddingFunction()

# Get or create the collection
collection = client.get_or_create_collection(
    name="autoresearch_docs",
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
    Clears all items in the autoresearch_docs collection.
    """
    global collection
    try:
        client.delete_collection("autoresearch_docs")
    except Exception:
        pass
    collection = client.get_or_create_collection(
        name="autoresearch_docs",
        embedding_function=embedding_fn
    )
