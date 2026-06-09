import json
import logging
from core.llm import get_llm

logger = logging.getLogger(__name__)

def generate_sub_questions(query: str) -> list[str]:
    """
    Accepts a raw research question, queries Claude to break it down
    into 3-5 focused sub-questions, and returns them as a Python list.
    
    Args:
        query (str): The original research question.
        
    Returns:
        list[str]: A list of 3-5 sub-questions.
    """
    if not query:
        return []
        
    system_prompt = (
        "You are an expert research planner. Your task is to break down a raw research question "
        "into 3-5 highly focused, specific sub-questions that can be searched on the web. "
        "Each sub-question should target a distinct facet of the main question (e.g., market size, "
        "key players, regional policies, future trends). "
        "Return the sub-questions strictly as a JSON list of strings. Do not include any markdown "
        "formatting, explainers, or text outside the JSON list. "
        "Example output: [\"Question 1\", \"Question 2\", \"Question 3\"]"
    )
    
    try:
        llm = get_llm(max_tokens=2000)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Research Question: {query}"}
        ]
        
        response = llm.invoke(messages)
        content = response.content.strip()
        
        # Clean up markdown code block if present
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        sub_questions = json.loads(content)
        if isinstance(sub_questions, list):
            cleaned_questions = [str(q).strip() for q in sub_questions if q]
            if not cleaned_questions:
                raise ValueError("Parsed JSON list is empty.")
            return cleaned_questions
        else:
            raise ValueError(f"Expected list, got {type(sub_questions)}")
            
    except Exception as e:
        logger.error(f"Error in Planner Agent: {e}")
        # Graceful fallback: return a default set of sub-questions
        fallback_query = query.strip()
        return [
            f"What is the general overview of {fallback_query}?",
            f"What are the key statistics and recent developments in {fallback_query}?",
            f"Who are the major competitors or key entities in {fallback_query}?"
        ]
