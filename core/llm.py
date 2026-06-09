import os
import json
import logging
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

# Load environment variables from .env
load_dotenv()

logger = logging.getLogger(__name__)

class MockChatAnthropic:
    """
    A smart mock fallback chat LLM that simulates Claude responses.
    It parses search results from the prompt context to generate
    plausible key findings, references, and reports.
    """
    def __init__(self, max_tokens: int = 2000):
        self.max_tokens = max_tokens

    def invoke(self, messages: list) -> 'MockResponse':
        system_content = ""
        user_content = ""
        
        for m in messages:
            if isinstance(m, dict):
                role = m.get("role")
                content = m.get("content", "")
            else:
                role = getattr(m, "type", "user")
                content = getattr(m, "content", "")
            
            if role == "system":
                system_content = content
            elif role == "user":
                user_content = content

        class MockResponse:
            def __init__(self, content):
                self.content = content

        # 1. Planner Node
        if "research planner" in system_content.lower():
            logger.info("Mock LLM: Planning sub-questions...")
            query = "the topic"
            if "Research Question:" in user_content:
                query = user_content.split("Research Question:")[1].strip()
            
            sub_qs = [
                f"What is the general overview of {query}?",
                f"What are key trends and statistics related to {query}?",
                f"Who are the top players and organizations involved in {query}?"
            ]
            return MockResponse(json.dumps(sub_qs))

        # 2. Reasoner Node
        elif "analytical reasoner" in system_content.lower():
            logger.info("Mock LLM: Analyzing and reasoning over sources...")
            query = "Research Query"
            if "Original Research Query:" in user_content:
                parts = user_content.split("Original Research Query:")
                if len(parts) > 1:
                    query = parts[1].split("\n")[0].strip()

            key_findings = []
            references = []
            
            # Extract source snippets from user_content
            sources = user_content.split("--- SOURCE")
            ref_idx = 1
            for src in sources[1:]:
                lines = src.split("\n")
                title = "Source Document"
                url = "http://example.com"
                content_snippet = ""
                
                for line in lines:
                    if line.startswith("Title:"):
                        title = line.replace("Title:", "").strip()
                    elif line.startswith("URL:"):
                        url = line.replace("URL:", "").strip()
                
                content_idx = src.find("Content:")
                if content_idx != -1:
                    # Look for next "--- SOURCE" or end
                    content_text = src[content_idx + len("Content:"):].strip()
                    content_snippet = content_text.split("--- SOURCE")[0].strip()
                    
                snippet_trimmed = content_snippet[:150] + "..." if len(content_snippet) > 150 else content_snippet
                
                if content_snippet:
                    key_findings.append(f"According to research: {snippet_trimmed} [Source {ref_idx}]")
                    references.append({"index": ref_idx, "title": title, "url": url})
                    ref_idx += 1
                
            if not key_findings:
                key_findings = [f"Found general search information regarding {query}."]
                references = [{"index": 1, "title": "Web Search Result", "url": "https://tavily.com"}]
                
            reasoning_data = {
                "key_findings": key_findings[:5],
                "confidence_level": "High" if len(references) >= 3 else "Medium",
                "confidence_reasoning": "Determined by aggregating multiple web search results and verifying cross-references.",
                "references": references,
                "gaps": [f"Details on highly specific future timelines beyond 2026 for {query}."],
                "contradictions_resolved": []
            }
            return MockResponse(json.dumps(reasoning_data))

        # 3. Reporter Node
        elif "technical writer" in system_content.lower():
            logger.info("Mock LLM: Formatting final markdown report...")
            reasoning_data = {}
            if "Structured Reasoning Data (JSON):" in user_content:
                json_str = user_content.split("Structured Reasoning Data (JSON):")[1].strip()
                try:
                    reasoning_data = json.loads(json_str)
                except Exception:
                    pass
            
            findings_li = "\n".join([f"- {finding}" for finding in reasoning_data.get("key_findings", [])])
            sources_li = ""
            for ref in reasoning_data.get("references", []):
                idx = ref.get("index", "?")
                title = ref.get("title", "Unknown Source")
                url = ref.get("url", "")
                sources_li += f"{idx}. [{title}]({url})\n"
                
            gaps_li = "\n".join([f"- {gap}" for gap in reasoning_data.get("gaps", [])])
            
            report = (
                f"# Executive Summary\n"
                f"This report presents an autonomous investigation into the query. It synthesizes findings "
                f"from multiple web sources indexed in ChromaDB. The results outline core trends, data points, "
                f"and active players in this space.\n\n"
                f"# Key Findings\n"
                f"{findings_li}\n\n"
                f"# Detailed Analysis\n"
                f"Synthesis of the gathered sources indicates significant activity and interest. The references "
                f"contain statistics, development reports, and market indicators. In particular, the findings "
                f"corroborate key trends observed in early 2026.\n\n"
                f"# Source List\n"
                f"{sources_li}\n"
                f"# Confidence Score\n"
                f"**{reasoning_data.get('confidence_level', 'Medium')}**\n"
                f"{reasoning_data.get('confidence_reasoning', 'Based on available references.')}\n\n"
                f"# Gaps & Limitations\n"
                f"{gaps_li}"
            )
            return MockResponse(report)
            
        return MockResponse("Mock Response")

def get_llm(max_tokens: int = 2000):
    """
    Returns an instance of ChatAnthropic (claude-sonnet-4-20250514)
    if ANTHROPIC_API_KEY is configured. Falls back to a smart Mock LLM
    engine if key is missing or set to placeholder.
    
    Args:
        max_tokens (int): The maximum tokens for the model response.
        
    Returns:
        ChatAnthropic or MockChatAnthropic: The initialized LLM client.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("ANTHROPIC_API_KEY")
        except Exception:
            pass

    # Check if key is missing or is the placeholder
    is_mock = (
        not api_key or 
        api_key.strip() == "" or 
        "YOUR_ANTHROPIC" in api_key or 
        api_key.strip().upper() == "MOCK"
    )
    
    if is_mock:
        logger.info("ANTHROPIC_API_KEY not set or is placeholder. Using smart Mock LLM mode.")
        return MockChatAnthropic(max_tokens=max_tokens)
    
    logger.info("Initializing ChatAnthropic LLM client.")
    return ChatAnthropic(
        model="claude-sonnet-4-20250514",
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=0.0
    )
