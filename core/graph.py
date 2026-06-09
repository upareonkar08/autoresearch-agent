import logging
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END

# Import agents and functions
from agents.planner import generate_sub_questions
from agents.searcher import search_sub_questions
from agents.reader import process_search_results, process_pdf, retrieve, reset_db
from agents.reasoner import reason_over_sources
from agents.reporter import generate_report

logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    original_query: str
    sub_questions: List[str]
    search_results: List[Dict[str, Any]]
    chunks: List[Dict[str, Any]]
    reasoning: Dict[str, Any]
    final_report: str
    feedback: Optional[Dict[str, Any]]
    pdf_bytes: Optional[bytes]
    pdf_filename: Optional[str]
    errors: List[str]

# Node 1: Planner
def planner_node(state: AgentState) -> Dict[str, Any]:
    logger.info("Executing Planner Node")
    errors = list(state.get("errors", []))
    try:
        sub_qs = generate_sub_questions(state["original_query"])
        return {"sub_questions": sub_qs}
    except Exception as e:
        err_msg = f"Planner node failed: {e}"
        logger.error(err_msg)
        errors.append(err_msg)
        return {"sub_questions": [state["original_query"]], "errors": errors}

# Node 2: Searcher
def searcher_node(state: AgentState) -> Dict[str, Any]:
    logger.info("Executing Searcher Node")
    errors = list(state.get("errors", []))
    try:
        results = search_sub_questions(state["sub_questions"])
        return {"search_results": results}
    except Exception as e:
        err_msg = f"Searcher node failed: {e}"
        logger.error(err_msg)
        errors.append(err_msg)
        return {"search_results": [], "errors": errors}

# Node 3: Reader
def reader_node(state: AgentState) -> Dict[str, Any]:
    logger.info("Executing Reader Node")
    errors = list(state.get("errors", []))
    try:
        # Reset database for a fresh run
        reset_db()
        
        # Process Tavily search results if any
        if state.get("search_results"):
            process_search_results(state["search_results"], state["original_query"])
            
        # Process uploaded PDF if provided
        if state.get("pdf_bytes") and state.get("pdf_filename"):
            process_pdf(state["pdf_bytes"], state["pdf_filename"], state["original_query"])
            
        # Retrieve top 10 relevant chunks for reasoning
        # (retrieving 10 chunks provides richer context for analysis than just 5)
        retrieved = retrieve(state["original_query"], n=10)
        return {"chunks": retrieved}
    except Exception as e:
        err_msg = f"Reader node failed: {e}"
        logger.error(err_msg)
        errors.append(err_msg)
        return {"chunks": [], "errors": errors}

# Node 4: Reasoner
def reasoner_node(state: AgentState) -> Dict[str, Any]:
    logger.info("Executing Reasoner Node")
    errors = list(state.get("errors", []))
    try:
        reasoning_data = reason_over_sources(state["original_query"], state["chunks"])
        return {"reasoning": reasoning_data}
    except Exception as e:
        err_msg = f"Reasoner node failed: {e}"
        logger.error(err_msg)
        errors.append(err_msg)
        fallback_reasoning = {
            "key_findings": ["Failed to analyze sources due to reasoning error."],
            "confidence_level": "Low",
            "confidence_reasoning": str(e),
            "references": [],
            "gaps": ["Reasoner failure"],
            "contradictions_resolved": []
        }
        return {"reasoning": fallback_reasoning, "errors": errors}

# Node 5: Reporter
def reporter_node(state: AgentState) -> Dict[str, Any]:
    logger.info("Executing Reporter Node")
    errors = list(state.get("errors", []))
    try:
        report = generate_report(state["original_query"], state["reasoning"])
        return {"final_report": report}
    except Exception as e:
        err_msg = f"Reporter node failed: {e}"
        logger.error(err_msg)
        errors.append(err_msg)
        fallback_report = f"# Research Report\nAn error occurred: {e}"
        return {"final_report": fallback_report, "errors": errors}

# Build workflow graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("planner", planner_node)
workflow.add_node("searcher", searcher_node)
workflow.add_node("reader", reader_node)
workflow.add_node("reasoner", reasoner_node)
workflow.add_node("reporter", reporter_node)

# Set entry point
workflow.set_entry_point("planner")

# Define execution flow
workflow.add_edge("planner", "searcher")
workflow.add_edge("searcher", "reader")
workflow.add_edge("reader", "reasoner")
workflow.add_edge("reasoner", "reporter")
workflow.add_edge("reporter", END)

# Compile graph
research_graph = workflow.compile()
