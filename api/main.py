import os
import json
import logging
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import LangGraph research graph
from core.graph import research_graph

# Import memory and collaboration utilities
from core.memory import get_user_history
from core.collaboration import get_topic_feed, get_all_topics, join_topic

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directory setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
FEEDBACK_FILE = os.path.join(DATA_DIR, "feedback.json")

app = FastAPI(
    title="AutoResearch Agent API",
    description="Backend API for running the autonomous research graph with memory, domains, and collaboration.",
    version="2.0.0"
)

# CORS configuration to allow connections from Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ResearchRequestJSON(BaseModel):
    query: str = Field(..., description="The main research question.")
    domain: Optional[str] = Field("General", description="Medical, Legal, Tech, Finance, General")
    user_id: Optional[str] = Field("", description="Unique user session ID")
    username: Optional[str] = Field("Guest", description="Name of the researcher")
    topic_id: Optional[str] = Field("", description="Optional collaborative topic ID")

class FeedbackRequest(BaseModel):
    query: str = Field(..., description="The research question submitted.")
    rating: int = Field(..., ge=1, le=5, description="Star rating from 1 to 5.")
    comment: Optional[str] = Field("", description="Optional comment or suggestions.")

@app.get("/health")
async def health_check():
    """
    Returns API health status.
    """
    return {"status": "ok"}

@app.post("/research")
async def run_research(
    query: str = Form(...),
    domain: str = Form("General"),
    user_id: Optional[str] = Form(""),
    username: Optional[str] = Form("Guest"),
    topic_id: Optional[str] = Form(""),
    file: Optional[UploadFile] = File(None),
    images: Optional[List[UploadFile]] = File(None)
):
    """
    Runs the full LangGraph autonomous research pipeline.
    Accepts query, domain, memory keys, collaboration keys, reference PDF, and reference images.
    """
    logger.info(f"Received research request: '{query}' in domain '{domain}'")
    pdf_bytes = None
    pdf_filename = None
    images_payload = []
    
    # Process PDF reference
    if file:
        logger.info(f"Processing attached PDF: {file.filename}")
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        try:
            pdf_bytes = await file.read()
            pdf_filename = file.filename
        except Exception as e:
            logger.error(f"Failed to read uploaded PDF: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to read PDF file: {str(e)}")

    # Process reference images (multimodal)
    if images:
        for img in images:
            if img.filename:
                logger.info(f"Processing attached image: {img.filename}")
                try:
                    img_bytes = await img.read()
                    images_payload.append({
                        "bytes": img_bytes,
                        "filename": img.filename
                    })
                except Exception as e:
                    logger.error(f"Failed to read uploaded image '{img.filename}': {e}")
                    raise HTTPException(status_code=500, detail=f"Failed to read image '{img.filename}': {str(e)}")

    # Initialize graph state
    initial_state = {
        "original_query": query,
        "sub_questions": [],
        "search_results": [],
        "chunks": [],
        "reasoning": {},
        "final_report": "",
        "feedback": None,
        "pdf_bytes": pdf_bytes,
        "pdf_filename": pdf_filename,
        "images": images_payload,
        "domain": domain,
        "user_id": user_id,
        "username": username,
        "topic_id": topic_id,
        "eval_scores": {},
        "errors": []
    }
    
    try:
        # Run graph execution
        result = research_graph.invoke(initial_state)
        
        report = result.get("final_report", "")
        reasoning = result.get("reasoning", {})
        sources = reasoning.get("references", [])
        confidence = reasoning.get("confidence_level", "Medium")
        eval_scores = result.get("eval_scores", {})
        
        if result.get("errors"):
            logger.warning(f"Graph finished with errors: {result['errors']}")
            
        return {
            "report": report,
            "sources": sources,
            "confidence": confidence,
            "eval_scores": eval_scores
        }
        
    except Exception as e:
        logger.error(f"Error running research workflow: {e}")
        raise HTTPException(status_code=500, detail=f"Internal research error: {str(e)}")

@app.post("/research/json")
async def run_research_json(request: ResearchRequestJSON):
    """
    Alternative JSON body endpoint for simple API calls.
    """
    initial_state = {
        "original_query": request.query,
        "sub_questions": [],
        "search_results": [],
        "chunks": [],
        "reasoning": {},
        "final_report": "",
        "feedback": None,
        "pdf_bytes": None,
        "pdf_filename": None,
        "images": [],
        "domain": request.domain,
        "user_id": request.user_id,
        "username": request.username,
        "topic_id": request.topic_id,
        "eval_scores": {},
        "errors": []
    }
    try:
        result = research_graph.invoke(initial_state)
        return {
            "report": result.get("final_report", ""),
            "sources": result.get("reasoning", {}).get("references", []),
            "confidence": result.get("reasoning", {}).get("confidence_level", "Medium"),
            "eval_scores": result.get("eval_scores", {})
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/collaborative/topics")
async def fetch_topics():
    """
    Retrieves all active shared collaboration topics.
    """
    return get_all_topics()

@app.get("/collaborative/topic/{topic_id}")
async def fetch_topic_feed(topic_id: str):
    """
    Retrieves the activity feed and active users for a collaborative topic.
    """
    return get_topic_feed(topic_id)

@app.post("/collaborative/join")
async def join_collab_topic(topic_id: str = Form(...), username: str = Form(...), query: str = Form("")):
    """
    Registers a user joining a collaborative topic.
    """
    try:
        join_topic(topic_id, username, query)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memory/{user_id}")
async def fetch_user_memory(user_id: str):
    """
    Retrieves a user's persistent research history.
    """
    return get_user_history(user_id)

@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """
    Saves user rating and comments to data/feedback.json.
    """
    logger.info(f"Received feedback for query: '{request.query}'")
    new_feedback = {
        "query": request.query,
        "rating": request.rating,
        "comment": request.comment
    }
    
    try:
        feedback_list = []
        if os.path.exists(FEEDBACK_FILE):
            try:
                with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        feedback_list = json.loads(content)
            except Exception:
                pass
                
        feedback_list.append(new_feedback)
        with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
            json.dump(feedback_list, f, indent=4, ensure_ascii=False)
            
        return {"status": "success", "message": "Feedback saved."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
