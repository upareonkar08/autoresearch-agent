import os
import json
import logging
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import LangGraph research graph
from core.graph import research_graph

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
    description="Backend API for running the autonomous research graph.",
    version="1.0.0"
)

# CORS configuration to allow connections from Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for dev simplicity
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ResearchRequestJSON(BaseModel):
    query: str = Field(..., description="The main research question.")

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
    file: Optional[UploadFile] = File(None)
):
    """
    Runs the full LangGraph autonomous research pipeline.
    Accepts research query as form parameter and optional PDF file.
    """
    logger.info(f"Received research request: '{query}'")
    pdf_bytes = None
    pdf_filename = None
    
    if file:
        logger.info(f"Processing attached file: {file.filename}")
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        try:
            pdf_bytes = await file.read()
            pdf_filename = file.filename
        except Exception as e:
            logger.error(f"Failed to read uploaded PDF: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to read file: {str(e)}")

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
        "errors": []
    }
    
    try:
        # Run graph execution (using async invoke if possible, or synchronous invoke)
        # LangGraph invoke() is synchronous by default unless we use ainvoke()
        # We will run the synchronous invoke in an async endpoint, which is standard in FastAPI.
        result = research_graph.invoke(initial_state)
        
        report = result.get("final_report", "")
        reasoning = result.get("reasoning", {})
        sources = reasoning.get("references", [])
        confidence = reasoning.get("confidence_level", "Medium")
        
        if result.get("errors"):
            logger.warning(f"Graph finished with warnings/errors: {result['errors']}")
            
        return {
            "report": report,
            "sources": sources,
            "confidence": confidence
        }
        
    except Exception as e:
        logger.error(f"Error running research workflow: {e}")
        raise HTTPException(status_code=500, detail=f"Internal research error: {str(e)}")

@app.post("/research/json")
async def run_research_json(request: ResearchRequestJSON):
    """
    Alternative endpoint to accept simple JSON requests for queries
    without file uploads.
    """
    return await run_research(query=request.query, file=None)

@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """
    Saves user rating and comments to data/feedback.json.
    """
    logger.info(f"Received feedback for: '{request.query}' with rating {request.rating}")
    
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
                        if not isinstance(feedback_list, list):
                            feedback_list = []
            except Exception as e:
                logger.error(f"Failed to read existing feedback file: {e}")
                
        feedback_list.append(new_feedback)
        
        with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
            json.dump(feedback_list, f, indent=4, ensure_ascii=False)
            
        return {"status": "success", "message": "Feedback saved successfully."}
        
    except Exception as e:
        logger.error(f"Failed to save feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save feedback: {str(e)}")
