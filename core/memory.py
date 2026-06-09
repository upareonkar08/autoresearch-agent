import os
import json
import logging
import time

logger = logging.getLogger(__name__)

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MEMORY_FILE = os.path.join(DATA_DIR, "memory.json")

def load_memory() -> dict:
    """
    Loads memory database from data/memory.json.
    
    Returns:
        dict: The memory database.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(MEMORY_FILE):
        return {}
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                return json.loads(content)
    except Exception as e:
        logger.error(f"Failed to load memory file: {e}")
    return {}

def save_memory(data: dict):
    """
    Saves memory database to data/memory.json.
    
    Args:
        data (dict): The memory database to serialize.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to save memory file: {e}")

def save_session(user_id: str, query: str, findings: list[str], preferences: dict = None):
    """
    Saves a research session's query, key findings, and user preferences for a user.
    
    Args:
        user_id (str): The unique ID of the user.
        query (str): The research query.
        findings (list[str]): Key findings generated from the research.
        preferences (dict, optional): Selected user preferences to store.
    """
    if not user_id:
        return
    
    db = load_memory()
    if user_id not in db:
        db[user_id] = {
            "sessions": [],
            "preferences": {}
        }
        
    # Append session
    db[user_id]["sessions"].append({
        "query": query,
        "findings": findings or [],
        "timestamp": time.time()
    })
    
    # Update preferences
    if preferences:
        db[user_id]["preferences"].update(preferences)
        
    save_memory(db)

def get_user_history(user_id: str) -> dict:
    """
    Retrieves the full history and preferences for a user.
    
    Args:
        user_id (str): The user's unique ID.
        
    Returns:
        dict: User history and preference logs.
    """
    if not user_id:
        return {"sessions": [], "preferences": {}}
    db = load_memory()
    return db.get(user_id, {"sessions": [], "preferences": {}})

def get_recent_memories(user_id: str, n: int = 3) -> str:
    """
    Formulates a helper context string detailing the user's recent queries
    and key findings to inject into the planning node.
    
    Args:
        user_id (str): Unique ID of the user.
        n (int): Number of recent sessions to summarize.
        
    Returns:
        str: A formatted summary of the user's recent research.
    """
    history = get_user_history(user_id)
    sessions = history.get("sessions", [])
    if not sessions:
        return "No prior research history."
        
    # Sort sessions by timestamp if available
    sorted_sessions = sorted(sessions, key=lambda x: x.get("timestamp", 0))
    recent = sorted_sessions[-n:]
    
    summary = ""
    for idx, s in enumerate(reversed(recent)):
        q = s.get("query", "Unknown Query")
        finds = s.get("findings", [])
        finds_str = "; ".join(finds[:2]) if finds else "No findings recorded."
        summary += f"- Past query {idx + 1}: '{q}' (Key Findings: {finds_str})\n"
        
    return summary
