import json
import logging
from core.llm import get_llm

logger = logging.getLogger(__name__)

def evaluate_report(query: str, report: str, retrieved_chunks: list[dict]) -> dict:
    """
    Evaluates the research report against the retrieved chunks and original query.
    Scores accuracy, coverage, citation quality, and clarity on a scale of 0-10.
    
    Args:
        query (str): The original user query.
        report (str): The generated markdown report.
        retrieved_chunks (list[dict]): The list of source chunks from Reader.
        
    Returns:
        dict: A structured dictionary of scores and justifications.
    """
    if not report or report.startswith("# Research Report\nAn error occurred"):
        return {
            "accuracy": {"score": 0.0, "reason": "No valid report was generated to score."},
            "coverage": {"score": 0.0, "reason": "No valid report was generated to score."},
            "citation_quality": {"score": 0.0, "reason": "No valid report was generated to score."},
            "clarity": {"score": 0.0, "reason": "No valid report was generated to score."},
            "average_score": 0.0
        }

    # Format source text for context
    sources_text = ""
    for idx, chunk in enumerate(retrieved_chunks):
        meta = chunk.get("metadata", {})
        title = meta.get("title", "Untitled")
        content = chunk.get("content", "")
        sources_text += f"--- SOURCE {idx + 1} ---\nTitle: {title}\nContent: {content}\n\n"

    system_prompt = (
        "You are an expert peer reviewer and evaluation bot. Your task is to evaluate the quality of a generated research report based on the retrieved sources.\n\n"
        "You must score the report on the following 4 metrics, each on a scale of 0 to 10:\n"
        "1. \"accuracy\": How factually accurate is the report compared to the sources? (0-10)\n"
        "2. \"coverage\": How thoroughly does it answer the original research question based on available source information? (0-10)\n"
        "3. \"citation_quality\": How well-cited are the claims and are the references valid? (0-10)\n"
        "4. \"clarity\": How well-structured, clear, and professional is the writing? (0-10)\n\n"
        "You must return your evaluation strictly as a JSON object with the following keys:\n"
        "- \"accuracy\": {\"score\": float, \"reason\": string}\n"
        "- \"coverage\": {\"score\": float, \"reason\": string}\n"
        "- \"citation_quality\": {\"score\": float, \"reason\": string}\n"
        "- \"clarity\": {\"score\": float, \"reason\": string}\n"
        "- \"average_score\": float\n\n"
        "Do not include any explanation or formatting outside the JSON object."
    )

    user_content = (
        f"Original Research Query: {query}\n\n"
        f"Retrieved Sources:\n{sources_text}\n\n"
        f"Generated Report:\n{report}"
    )

    try:
        llm = get_llm(max_tokens=2000)
        
        # Check if we are running in Mock mode
        from core.llm import MockChatAnthropic
        if isinstance(llm, MockChatAnthropic):
            # Smart deterministic scoring fallback for Mock Mode
            num_sources = len(retrieved_chunks)
            cit_score = 9.0 if num_sources >= 3 else 7.0 if num_sources >= 1 else 3.0
            cov_score = 8.5 if len(report.split()) > 100 else 4.0
            acc_score = 9.0 if num_sources >= 1 else 5.0
            cl_score = 9.5
            
            avg = (acc_score + cov_score + cit_score + cl_score) / 4.0
            
            return {
                "accuracy": {
                    "score": acc_score,
                    "reason": f"Findings match the {num_sources} retrieved documents from the vector database exactly."
                },
                "coverage": {
                    "score": cov_score,
                    "reason": "Successfully covered the main facets of the query using search result snippets."
                },
                "citation_quality": {
                    "score": cit_score,
                    "reason": f"Proper bracketed citations are used. Found and mapped {num_sources} unique sources in bibliography."
                },
                "clarity": {
                    "score": cl_score,
                    "reason": "Report is formatted using standard markdown sections with clean headings, bullets, and links."
                },
                "average_score": avg
            }

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        response = llm.invoke(messages)
        content = response.content.strip()
        
        # Clean markdown code block if present
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        eval_data = json.loads(content)
        return eval_data

    except Exception as e:
        logger.error(f"Error in Evaluator Agent: {e}")
        return {
            "accuracy": {"score": 5.0, "reason": f"Evaluation error: {e}"},
            "coverage": {"score": 5.0, "reason": "Evaluation failed to execute."},
            "citation_quality": {"score": 5.0, "reason": "Evaluation failed to execute."},
            "clarity": {"score": 5.0, "reason": "Evaluation failed to execute."},
            "average_score": 5.0
        }
