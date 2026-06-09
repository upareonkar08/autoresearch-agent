import os
import sys
import logging
from dotenv import load_dotenv

# Ensure root folder is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging to see node transitions
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load env variables from .env
load_dotenv()

from core.graph import research_graph

def run_test():
    """
    Runs a test research query through the LangGraph workflow and prints the resulting report.
    """
    # Check keys
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")
    
    if not anthropic_key or "YOUR_ANTHROPIC" in anthropic_key or not anthropic_key.strip():
        logger.error("Error: ANTHROPIC_API_KEY is not set in your .env file!")
        sys.exit(1)
        
    if not tavily_key or "YOUR_TAVILY" in tavily_key or not tavily_key.strip():
        logger.error("Error: TAVILY_API_KEY is not set in your .env file!")
        sys.exit(1)
        
    query = "What are the latest trends in generative AI in 2025?"
    logger.info(f"Starting research for query: '{query}'")
    
    initial_state = {
        "original_query": query,
        "sub_questions": [],
        "search_results": [],
        "chunks": [],
        "reasoning": {},
        "final_report": "",
        "feedback": None,
        "pdf_bytes": None,
        "pdf_filename": None,
        "errors": []
    }
    
    try:
        result = research_graph.invoke(initial_state)
        report = result.get("final_report", "")
        
        print("\n" + "="*80)
        print("GENERATED RESEARCH REPORT:")
        print("="*80 + "\n")
        print(report)
        print("\n" + "="*80)
        
    except Exception as e:
        logger.error(f"Test run failed: {e}", exc_info=True)

if __name__ == "__main__":
    run_test()
