import logging
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END

# Import agents and functions
from agents.planner import generate_sub_questions
from agents.searcher import search_sub_questions
from agents.reader import process_search_results, process_pdf, process_image, retrieve, reset_db
from agents.reasoner import reason_over_sources
from agents.reporter import generate_report
from agents.evaluator import evaluate_report

# Import memory and collaboration helpers
from core.memory import save_session, get_recent_memories
from core.collaboration import join_topic, post_finding

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
    
    # New Multimodal image upload key
    # List of dicts: [{"bytes": bytes, "filename": str}]
    images: Optional[List[Dict[str, Any]]]
    
    # New Domain-Specialized Mode key
    domain: str # 'Medical', 'Legal', 'Tech', 'Finance', 'General'
    
    # New Agent Memory keys
    user_id: Optional[str]
    
    # New Collaborative Research keys
    username: Optional[str]
    topic_id: Optional[str]
    
    # New Evaluation Module key
    eval_scores: Dict[str, Any]
    
    errors: List[str]

# Node 1: Planner
def planner_node(state: AgentState) -> Dict[str, Any]:
    logger.info("Executing Planner Node")
    errors = list(state.get("errors", []))
    domain = state.get("domain", "General")
    user_id = state.get("user_id", "")
    
    # Retrieve user's past research history to guide planning
    memory_history = ""
    if user_id:
        try:
            memory_history = get_recent_memories(user_id, n=3)
        except Exception as e:
            logger.warning(f"Failed to fetch memories: {e}")
            
    try:
        sub_qs = generate_sub_questions(state["original_query"], domain, memory_history)
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

# Node 3: Reader (Processes PDFs, Web Text, and Chart Images)
def reader_node(state: AgentState) -> Dict[str, Any]:
    logger.info("Executing Reader Node")
    errors = list(state.get("errors", []))
    try:
        # Reset database for a fresh run
        reset_db()
        
        # Process Tavily search results (which automatically extracts YouTube transcripts)
        if state.get("search_results"):
            process_search_results(state["search_results"], state["original_query"])
            
        # Process uploaded PDF if provided
        if state.get("pdf_bytes") and state.get("pdf_filename"):
            process_pdf(state["pdf_bytes"], state["pdf_filename"], state["original_query"])
            
        # Process uploaded images if provided (multimodal support)
        images = state.get("images", [])
        if images:
            for img in images:
                img_bytes = img.get("bytes")
                filename = img.get("filename", "image.png")
                if img_bytes:
                    process_image(img_bytes, filename, state["original_query"])
            
        # Retrieve top 10 relevant chunks for reasoning
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
    domain = state.get("domain", "General")
    try:
        reasoning_data = reason_over_sources(state["original_query"], state["chunks"], domain)
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
    domain = state.get("domain", "General")
    try:
        report = generate_report(state["original_query"], state["reasoning"], domain)
        return {"final_report": report}
    except Exception as e:
        err_msg = f"Reporter node failed: {e}"
        logger.error(err_msg)
        errors.append(err_msg)
        fallback_report = f"# Research Report\nAn error occurred: {e}"
        return {"final_report": fallback_report, "errors": errors}

# Node 6: Evaluator
def evaluator_node(state: AgentState) -> Dict[str, Any]:
    logger.info("Executing Evaluator Node")
    errors = list(state.get("errors", []))
    try:
        scores = evaluate_report(state["original_query"], state["final_report"], state["chunks"])
        
        # Post-process: Save memory and collaboration logs on successful execution
        user_id = state.get("user_id")
        if user_id:
            try:
                # Save session to memory
                findings = state["reasoning"].get("key_findings", [])
                save_session(user_id, state["original_query"], findings, {"domain": state.get("domain", "General")})
            except Exception as mem_err:
                logger.error(f"Failed to save session memory: {mem_err}")
                
        topic_id = state.get("topic_id")
        username = state.get("username", "Guest")
        if topic_id:
            try:
                # Join topic & Publish findings to shared collaborative session
                join_topic(topic_id, username, state["original_query"])
                post_finding(topic_id, username, state["original_query"], state["final_report"], state["reasoning"].get("references", []))
            except Exception as col_err:
                logger.error(f"Failed to post collaborative logs: {col_err}")
                
        return {"eval_scores": scores}
    except Exception as e:
        err_msg = f"Evaluator node failed: {e}"
        logger.error(err_msg)
        errors.append(err_msg)
        fallback_scores = {
            "accuracy": {"score": 5.0, "reason": str(e)},
            "coverage": {"score": 5.0, "reason": "Failed to run evaluator"},
            "citation_quality": {"score": 5.0, "reason": "Failed to run evaluator"},
            "clarity": {"score": 5.0, "reason": "Failed to run evaluator"},
            "average_score": 5.0
        }
        return {"eval_scores": fallback_scores, "errors": errors}

# Build workflow graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("planner", planner_node)
workflow.add_node("searcher", searcher_node)
workflow.add_node("reader", reader_node)
workflow.add_node("reasoner", reasoner_node)
workflow.add_node("reporter", reporter_node)
workflow.add_node("evaluator", evaluator_node)

# Set entry point
workflow.set_entry_point("planner")

# Define execution flow
workflow.add_edge("planner", "searcher")
workflow.add_edge("searcher", "reader")
workflow.add_edge("reader", "reasoner")
workflow.add_edge("reasoner", "reporter")
workflow.add_edge("reporter", "evaluator")
workflow.add_edge("evaluator", END)

# Compile graph
research_graph = workflow.compile()
