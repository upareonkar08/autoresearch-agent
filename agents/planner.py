import json
import logging
from core.llm import get_llm

logger = logging.getLogger(__name__)

def generate_sub_questions(query: str, domain: str = "General", memory_history: str = "") -> list[str]:
    """
    Accepts a raw research question, selected domain, and user memory history,
    queries Claude to break it down into 3-5 domain-specialized sub-questions,
    and returns them as a Python list.
    
    Args:
        query (str): The original research question.
        domain (str): The selected domain (Medical, Legal, Tech, Finance, General).
        memory_history (str): Summary of recent research sessions.
        
    Returns:
        list[str]: A list of 3-5 specialized sub-questions.
    """
    if not query:
        return []
        
    # Domain-specific instructions
    domain_guidelines = {
        "General": (
            "Break down the query into general web-searchable sub-questions covering overview, statistics, and players."
        ),
        "Medical": (
            "Break down the query with a focus on clinical trials, medical literature, peer-reviewed journals (like PubMed/Lancet), "
            "government healthcare policies, FDA guidelines, and clinical efficacy metrics."
        ),
        "Legal": (
            "Break down the query with a focus on case law precedents, statutory provisions, regulations, legal commentary, "
            "and legal rulings."
        ),
        "Tech": (
            "Break down the query with a focus on developer benchmarks, software documentation, GitHub repositories, system architectures, "
            "and tech hardware/software specifications."
        ),
        "Finance": (
            "Break down the query with a focus on company earnings reports, SEC filings (10-K/10-Q), market valuation multiples, "
            "industry benchmarks, and macroeconomic indicators."
        )
    }
    
    guideline = domain_guidelines.get(domain, domain_guidelines["General"])
    
    system_prompt = (
        "You are an expert research planner. Your task is to break down a raw research question "
        "into 3-5 highly focused, specific sub-questions that can be searched on the web.\n\n"
        f"Selected Domain: {domain}\n"
        f"Domain-Specific Guidelines: {guideline}\n\n"
        "You must also consider the user's recent research history (if any) to avoid asking redundant questions, "
        "and instead ask progressive questions that build on previous research.\n\n"
        "Return the sub-questions strictly as a JSON list of strings. Do not include any markdown "
        "formatting, explainers, or text outside the JSON list.\n"
        "Example output: [\"Question 1\", \"Question 2\", \"Question 3\"]"
    )
    
    user_content = (
        f"User Research History:\n{memory_history}\n\n"
        f"New Research Question: {query}"
    )
    
    try:
        llm = get_llm(max_tokens=2000)
        
        # Check if we are running in Mock mode
        from core.llm import MockChatAnthropic
        if isinstance(llm, MockChatAnthropic):
            # Dynamic mock questions based on domain
            logger.info(f"Mock Planner: Generating {domain} sub-questions...")
            q_clean = query.strip()
            if domain == "Medical":
                return [
                    f"What is the clinical overview of {q_clean}?",
                    f"What peer-reviewed studies exist for {q_clean}?",
                    f"What are current FDA guidelines or policies on {q_clean}?"
                ]
            elif domain == "Legal":
                return [
                    f"What case precedents exist for {q_clean}?",
                    f"What statutory regulations govern {q_clean}?",
                    f"What legal interpretations are discussed for {q_clean}?"
                ]
            elif domain == "Tech":
                return [
                    f"What are system architectures and libraries for {q_clean}?",
                    f"What benchmarks or GitHub repositories exist for {q_clean}?",
                    f"What are key technical specifications of {q_clean}?"
                ]
            elif domain == "Finance":
                return [
                    f"What do SEC filings and earnings report say about {q_clean}?",
                    f"What are the valuation metrics and market size of {q_clean}?",
                    f"What industry trends are showing up for {q_clean}?"
                ]
            else:
                return [
                    f"What is the general overview of {q_clean}?",
                    f"What are key trends and statistics related to {q_clean}?",
                    f"Who are the top players and organizations involved in {q_clean}?"
                ]
                
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
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
        # Fallback list based on domain
        fallback_query = query.strip()
        if domain == "Medical":
            return [
                f"What is the clinical overview of {fallback_query}?",
                f"What recent clinical trial data exists for {fallback_query}?"
            ]
        elif domain == "Legal":
            return [
                f"What statutory regulations govern {fallback_query}?",
                f"What are key legal rulings regarding {fallback_query}?"
            ]
        else:
            return [
                f"What is the general overview of {fallback_query}?",
                f"What are the key statistics and recent developments in {fallback_query}?"
            ]
