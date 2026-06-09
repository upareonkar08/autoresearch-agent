import os
import logging
from dotenv import load_dotenv
from tavily import TavilyClient

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

def search_sub_questions(sub_questions: list[str]) -> list[dict]:
    """
    Accepts a list of sub-questions, executes Tavily search queries for each,
    and returns a structured list of findings.
    
    Args:
        sub_questions (list[str]): List of sub-questions to search.
        
    Returns:
        list[dict]: A list of results containing:
            - "query": The sub-question searched.
            - "title": The title of the page.
            - "url": The source URL.
            - "content": The page snippet or contents.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        logger.error("TAVILY_API_KEY is not set in environment or .env file.")
        return []
        
    try:
        client = TavilyClient(api_key=api_key)
    except Exception as e:
        logger.error(f"Failed to initialize TavilyClient: {e}")
        return []
        
    all_results = []
    
    for sq in sub_questions:
        sq = sq.strip()
        if not sq:
            continue
        try:
            logger.info(f"Searching for: {sq}")
            # Request top 3 results per sub-question
            search_response = client.search(query=sq, search_depth="basic", max_results=3)
            
            results = search_response.get("results", [])
            for r in results:
                all_results.append({
                    "query": sq,
                    "title": r.get("title", "No Title"),
                    "url": r.get("url", ""),
                    "content": r.get("content", "")
                })
        except Exception as e:
            logger.error(f"Tavily search failed for sub-question '{sq}': {e}")
            # Continue to the next sub-question even if one fails
            continue
            
    return all_results
