import json
import logging
from core.llm import get_llm

logger = logging.getLogger(__name__)

def generate_report(query: str, reasoning_data: dict) -> str:
    """
    Accepts the original research question and the structured reasoning data,
    uses Claude to generate a clean, professional markdown report with specific sections,
    and returns it as a string.
    
    Args:
        query (str): The original user query.
        reasoning_data (dict): The structured dictionary returned by the Reasoner.
        
    Returns:
        str: The final markdown report.
    """
    system_prompt = (
        "You are an expert technical writer and researcher. Your task is to take a structured reasoning output "
        "about a research query and compile it into a clean, professional, and publication-ready markdown report. "
        "The report MUST have the following sections, formatted exactly as described:\n\n"
        "1. # Executive Summary\n"
        "   Provide a 3-5 sentence synthesis of the query, the core answer, and the overall context.\n\n"
        "2. # Key Findings\n"
        "   Present the key findings as bullet points. Each bullet point MUST end with a bracketed citation referring to the source list (e.g., [1] or [2]).\n\n"
        "3. # Detailed Analysis\n"
        "   Provide a comprehensive 2-3 paragraph breakdown of the findings, detailing context, examples, and nuances.\n\n"
        "4. # Source List\n"
        "   List the references in a numbered list, including their Title and clickable URL. Format: [Index] [Title](URL)\n\n"
        "5. # Confidence Score\n"
        "   State the confidence score: **Low** / **Medium** / **High**.\n"
        "   Explain the reasoning behind this confidence level based on source credibility, completeness, and consistency.\n\n"
        "6. # Gaps & Limitations\n"
        "   Discuss any unresolved gaps in information, contradictions encountered, and potential limitations of the sources.\n\n"
        "Keep the tone professional, objective, and clear. Return only the markdown string, starting directly with the 'Executive Summary' section (do not prefix it with code block quotes or introductory notes)."
    )
    
    user_content = (
        f"Original Research Query: {query}\n\n"
        f"Structured Reasoning Data (JSON):\n{json.dumps(reasoning_data, indent=2)}"
    )
    
    try:
        # Request 3000 tokens as specified for the reporting task
        llm = get_llm(max_tokens=3000)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        response = llm.invoke(messages)
        report_markdown = response.content.strip()
        
        # Clean up markdown code block wrapping if present
        if report_markdown.startswith("```markdown"):
            report_markdown = report_markdown[11:]
        elif report_markdown.startswith("```"):
            report_markdown = report_markdown[3:]
        if report_markdown.endswith("```"):
            report_markdown = report_markdown[:-3]
        report_markdown = report_markdown.strip()
        
        return report_markdown
        
    except Exception as e:
        logger.error(f"Error in Reporter Agent: {e}")
        # Fallback manual formatting if LLM fails
        findings_li = "\n".join([f"- {finding}" for finding in reasoning_data.get("key_findings", [])])
        
        sources_li = ""
        for ref in reasoning_data.get("references", []):
            idx = ref.get("index", "?")
            title = ref.get("title", "Unknown Title")
            url = ref.get("url", "")
            sources_li += f"{idx}. [{title}]({url})\n"
            
        gaps_li = "\n".join([f"- {gap}" for gap in reasoning_data.get("gaps", [])])
        
        fallback_report = (
            f"# Executive Summary\n"
            f"This is an automated fallback report generated in response to the query: '{query}'. "
            f"An error occurred during report styling, but the raw data has been compiled below.\n\n"
            f"# Key Findings\n"
            f"{findings_li}\n\n"
            f"# Detailed Analysis\n"
            f"Detailed LLM reporting was not available. Refer to the key findings and source list for details.\n\n"
            f"# Source List\n"
            f"{sources_li}\n"
            f"# Confidence Score\n"
            f"**{reasoning_data.get('confidence_level', 'Low')}**\n"
            f"{reasoning_data.get('confidence_reasoning', 'N/A')}\n\n"
            f"# Gaps & Limitations\n"
            f"{gaps_li}"
        )
        return fallback_report
