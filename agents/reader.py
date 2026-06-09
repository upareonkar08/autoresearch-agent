import uuid
import logging
import fitz  # PyMuPDF
from core.vectordb import add_chunks, retrieve as chroma_retrieve, clear_collection

logger = logging.getLogger(__name__)

def chunk_text(text: str, chunk_size_tokens: int = 500) -> list[str]:
    """
    Splits text content into chunks of roughly 500 tokens (approx 375 words),
    with a small word overlap of 10% to preserve context across boundaries.
    
    Args:
        text (str): Input text content.
        chunk_size_tokens (int): Target chunk size in tokens.
        
    Returns:
        list[str]: A list of text chunks.
    """
    if not text:
        return []
        
    words = text.split()
    chunk_size_words = int(chunk_size_tokens * 0.75)  # 1 token ~= 0.75 words
    overlap_words = max(1, int(chunk_size_words * 0.1))  # 10% overlap
    
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i : i + chunk_size_words]
        chunks.append(" ".join(chunk_words))
        i += (chunk_size_words - overlap_words)
        if len(chunk_words) < chunk_size_words:
            break
            
    return chunks

def process_search_results(results: list[dict], original_query: str):
    """
    Accepts search results, chunks the content, and stores chunks in ChromaDB
    with corresponding metadata: source_url, title, query.
    
    Args:
        results (list[dict]): A list of search results.
        original_query (str): The primary user query.
    """
    chunks_to_add = []
    metadatas = []
    ids = []
    
    for r in results:
        url = r.get("url", "")
        title = r.get("title", "No Title")
        query_searched = r.get("query", original_query)
        content = r.get("content", "")
        
        if not content:
            continue
            
        text_chunks = chunk_text(content, chunk_size_tokens=500)
        for chunk in text_chunks:
            chunks_to_add.append(chunk)
            metadatas.append({
                "source_url": url,
                "title": title,
                "query": query_searched
            })
            ids.append(str(uuid.uuid4()))
            
    if chunks_to_add:
        try:
            logger.info(f"Storing {len(chunks_to_add)} chunks in ChromaDB from search results.")
            add_chunks(chunks=chunks_to_add, metadatas=metadatas, ids=ids)
        except Exception as e:
            logger.error(f"Failed to store search chunks in ChromaDB: {e}")

def process_pdf(pdf_bytes: bytes, filename: str, original_query: str):
    """
    Accepts PDF file bytes, extracts text using PyMuPDF (fitz), chunks the content,
    and stores chunks in ChromaDB.
    
    Args:
        pdf_bytes (bytes): The raw PDF file bytes.
        filename (str): The name of the uploaded PDF file.
        original_query (str): The primary user query.
    """
    try:
        logger.info(f"Extracting text from PDF: {filename}")
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
            
        text_chunks = chunk_text(text, chunk_size_tokens=500)
        chunks_to_add = []
        metadatas = []
        ids = []
        
        for chunk in text_chunks:
            chunks_to_add.append(chunk)
            metadatas.append({
                "source_url": f"uploaded_file://{filename}",
                "title": filename,
                "query": original_query
            })
            ids.append(str(uuid.uuid4()))
            
        if chunks_to_add:
            logger.info(f"Storing {len(chunks_to_add)} chunks in ChromaDB from PDF.")
            add_chunks(chunks=chunks_to_add, metadatas=metadatas, ids=ids)
            
    except Exception as e:
        logger.error(f"Failed to parse or store PDF '{filename}': {e}")

def retrieve(query: str, n: int = 5) -> list[dict]:
    """
    Fetches top-N relevant chunks from ChromaDB for the specified query.
    
    Args:
        query (str): The user query to search against ChromaDB.
        n (int): Number of matching chunks to retrieve.
        
    Returns:
        list[dict]: List of retrieved chunks with metadata.
    """
    try:
        return chroma_retrieve(query, n=n)
    except Exception as e:
        logger.error(f"ChromaDB retrieval failed: {e}")
        return []

def reset_db():
    """
    Resets the ChromaDB collection to start a fresh research session.
    """
    try:
        clear_collection()
        logger.info("ChromaDB collection cleared successfully.")
    except Exception as e:
        logger.error(f"Failed to clear ChromaDB collection: {e}")
