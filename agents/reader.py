import re
import uuid
import base64
import logging
from youtube_transcript_api import YouTubeTranscriptApi

from core.vectordb import add_chunks, retrieve as chroma_retrieve, clear_collection
from core.llm import get_llm, MockChatAnthropic

logger = logging.getLogger(__name__)

def extract_youtube_id(url: str) -> str:
    """
    Extracts the 11-character video ID from a YouTube URL.
    
    Args:
        url (str): The video URL.
        
    Returns:
        str: The video ID if found, otherwise None.
    """
    pattern = r'(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, url)
    return match.group(1) if match else None

def get_youtube_transcript(video_id: str) -> str:
    """
    Fetches the caption transcript for a YouTube video using youtube-transcript-api.
    
    Args:
        video_id (str): The YouTube video ID.
        
    Returns:
        str: The combined transcript text, or empty string on failure.
    """
    try:
        logger.info(f"Fetching YouTube transcript for video ID: {video_id}")
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join([t.get("text", "") for t in transcript_list])
        return text
    except Exception as e:
        logger.warning(f"Failed to fetch YouTube transcript for {video_id}: {e}")
        return ""

def chunk_text(text: str, chunk_size_tokens: int = 500) -> list[str]:
    """
    Splits text content into chunks of roughly 500 tokens (approx 375 words),
    with a small word overlap of 10% to preserve context across boundaries.
    """
    if not text:
        return []
        
    words = text.split()
    chunk_size_words = int(chunk_size_tokens * 0.75)
    overlap_words = max(1, int(chunk_size_words * 0.1))
    
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
    Accepts search results, detects YouTube video links to extract captions,
    chunks all content, and stores chunks in ChromaDB.
    """
    chunks_to_add = []
    metadatas = []
    ids = []
    
    for r in results:
        url = r.get("url", "")
        title = r.get("title", "No Title")
        query_searched = r.get("query", original_query)
        content = r.get("content", "")
        
        # Check if URL is a YouTube link
        yt_id = extract_youtube_id(url)
        if yt_id:
            transcript = get_youtube_transcript(yt_id)
            if transcript:
                content = f"[YouTube Video Transcript for {title}]\n{transcript}"
                title = f"YouTube Video: {title}"
        
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
    Extracts text from PDF bytes using PyMuPDF (fitz), chunks it, and indexes in ChromaDB.
    """
    try:
        import fitz
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
            add_chunks(chunks=chunks_to_add, metadatas=metadatas, ids=ids)
            
    except Exception as e:
        logger.error(f"Failed to parse or store PDF '{filename}': {e}")

def process_image(image_bytes: bytes, filename: str, original_query: str):
    """
    Processes an image/chart by base64-encoding it, prompting Claude Vision
    to extract data points and trends, chunking the description, and storing in ChromaDB.
    
    Args:
        image_bytes (bytes): The raw image file bytes.
        filename (str): The name of the uploaded image.
        original_query (str): The research question.
    """
    try:
        llm = get_llm(max_tokens=2000)
        
        # Check if we are running in Mock mode
        if isinstance(llm, MockChatAnthropic):
            logger.info(f"Mock Vision: Analysing chart image {filename}...")
            # Generate a realistic mock chart description
            description = (
                f"[Analysis of Uploaded Chart: {filename}]\n"
                f"This data visualization represents key metric statistics for the query '{original_query}'. "
                f"The timeline axis covers 2024 to 2026. The values indicate a significant, steady growth "
                f"trend over the quarters, highlighting a 2.5x increase in activity by Q4 2025. Key indicators "
                f"are marked at 42% growth rate in 2025, beating earlier forecasts of 30%."
            )
        else:
            logger.info(f"Claude Vision: Analyzing chart image {filename}...")
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            
            # Format vision message content for ChatAnthropic
            from langchain_core.messages import HumanMessage
            content_payload = [
                {
                    "type": "text",
                    "text": (
                        "You are an expert vision intelligence bot. Analyze this image (which may be a chart, "
                        "graph, table, or infographic) in detail to help answer the research query: "
                        f"'{original_query}'. Extract all text, numbers, data points, trends, and labels. "
                        "Provide a comprehensive textual description of its contents."
                    )
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",  # Standard base64 media block
                        "data": image_base64
                    }
                }
            ]
            message = HumanMessage(content=content_payload)
            response = llm.invoke([message])
            description = response.content.strip()

        text_chunks = chunk_text(description, chunk_size_tokens=500)
        chunks_to_add = []
        metadatas = []
        ids = []
        
        for chunk in text_chunks:
            chunks_to_add.append(chunk)
            metadatas.append({
                "source_url": f"uploaded_image://{filename}",
                "title": f"Image Analysis: {filename}",
                "query": original_query
            })
            ids.append(str(uuid.uuid4()))
            
        if chunks_to_add:
            logger.info(f"Storing {len(chunks_to_add)} vision-derived chunks in ChromaDB.")
            add_chunks(chunks=chunks_to_add, metadatas=metadatas, ids=ids)
            
    except Exception as e:
        logger.error(f"Failed to process or store image '{filename}': {e}")

def retrieve(query: str, n: int = 5) -> list[dict]:
    """
    Fetches top-N relevant chunks from ChromaDB for the specified query.
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
