import os
import time
import httpx
import streamlit as st

# API Endpoint
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# Page configuration
st.set_page_config(
    page_title="AutoResearch Agent",
    page_icon="🕵️‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Custom CSS
st.markdown(
    """
    <style>
    /* Main container styling */
    .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
    }
    
    /* Header styling */
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(45deg, #FF4B4B, #FF8F8F, #4B8EFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #888896;
        margin-bottom: 2rem;
    }
    
    /* Glassmorphism containers */
    .content-box {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    
    /* Clickable source cards */
    .source-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 0.8rem 1.2rem;
        margin-bottom: 0.8rem;
        transition: transform 0.2s, background 0.2s;
    }
    .source-card:hover {
        transform: translateY(-2px);
        background: rgba(255, 255, 255, 0.05);
        border-color: rgba(255, 255, 255, 0.2);
    }
    
    /* Badges */
    .confidence-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        margin-left: 10px;
    }
    .badge-high {
        background-color: rgba(40, 167, 69, 0.2);
        color: #28a745;
        border: 1px solid #28a745;
    }
    .badge-medium {
        background-color: rgba(255, 193, 7, 0.2);
        color: #ffc107;
        border: 1px solid #ffc107;
    }
    .badge-low {
        background-color: rgba(220, 53, 69, 0.2);
        color: #dc3545;
        border: 1px solid #dc3545;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar config
with st.sidebar:
    st.image("https://img.icons8.com/nolan/128/artificial-intelligence.png", width=80)
    st.markdown("### AutoResearch Config")
    st.markdown("This autonomous agent breaks down questions, searches the web via Tavily, stores in ChromaDB, and reasons using Anthropic Claude.")
    st.divider()
    st.markdown("**Powered by:**")
    st.markdown("- LangGraph")
    st.markdown("- Claude 3.5 Sonnet")
    st.markdown("- Tavily Search API")
    st.markdown("- ChromaDB")

# Main Header
st.markdown("<div class='main-title'>AutoResearch Agent</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Autonomous cross-source deep research engine powered by LangGraph & Claude</div>", unsafe_allow_html=True)

# Session state initialization
if "report" not in st.session_state:
    st.session_state.report = None
if "sources" not in st.session_state:
    st.session_state.sources = []
if "confidence" not in st.session_state:
    st.session_state.confidence = None
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = False

# Search input box
query = st.text_area(
    "What research question would you like to investigate?",
    placeholder="e.g., What are the latest trends in generative AI in 2025?",
    height=100
)

# Optional PDF file upload
uploaded_file = st.file_uploader(
    "Upload a reference PDF document (Optional)",
    type=["pdf"],
    help="The document will be chunked and stored in the vector database alongside web search results."
)

col1, col2 = st.columns([1, 5])
with col1:
    search_button = st.button("Start Research", type="primary", use_container_width=True)

if search_button and query.strip():
    # Reset states
    st.session_state.report = None
    st.session_state.sources = []
    st.session_state.confidence = None
    st.session_state.last_query = query
    st.session_state.feedback_submitted = False
    
    # Progress display using st.status
    with st.status("Initializing research pipeline...", expanded=True) as status_box:
        
        status_box.update(label="Planning search strategy...", state="running")
        # Step 1: Planning
        st.write("🟢 Planning... Decomposing question into sub-queries.")
        time.sleep(1.5)
        
        # Step 2: Searching
        status_box.update(label="Searching the web...", state="running")
        st.write("🟢 Searching... Executing queries via Tavily Search API.")
        time.sleep(1.5)
        
        # Step 3: Reading
        status_box.update(label="Reading and indexing findings...", state="running")
        st.write("🟢 Reading... Chunking content & storing in ChromaDB vector store.")
        time.sleep(1.5)
        
        # Step 4: Reasoning
        status_box.update(label="Synthesizing and resolving conflicts...", state="running")
        st.write("🟢 Reasoning... Analyzing documents and resolving contradictions using Claude.")
        
        # Call backend API
        api_success = False
        try:
            files = None
            data = {"query": query}
            
            if uploaded_file:
                # Add file content for multipart post
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                
            response = httpx.post(f"{API_URL}/research", data=data, files=files, timeout=180.0)
            
            if response.status_code == 200:
                result = response.json()
                st.session_state.report = result.get("report", "")
                st.session_state.sources = result.get("sources", [])
                st.session_state.confidence = result.get("confidence", "Medium")
                api_success = True
                
                # Step 5: Writing Report
                status_box.update(label="Writing final research report...", state="running")
                st.write("🟢 Writing Report... Generating formatted markdown document.")
                time.sleep(1.0)
                status_box.update(label="Research complete!", state="complete")
            else:
                st.warning(f"Backend API returned status {response.status_code}. Attempting local fallback.")
        except Exception as e:
            st.info("Backend API is offline or unreachable. Falling back to in-process execution...")
            
        # If API failed or was unreachable, run the graph in-process
        error_message = None
        if not api_success:
            try:
                # Dynamically import graph components
                from core.graph import research_graph
                
                pdf_bytes = uploaded_file.getvalue() if uploaded_file else None
                pdf_filename = uploaded_file.name if uploaded_file else None
                
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
                
                # Run the graph locally in the Streamlit process
                result = research_graph.invoke(initial_state)
                
                st.session_state.report = result.get("final_report", "")
                reasoning = result.get("reasoning", {})
                st.session_state.sources = reasoning.get("references", [])
                st.session_state.confidence = reasoning.get("confidence_level", "Medium")
                
                # Step 5: Writing Report
                status_box.update(label="Writing final research report...", state="running")
                st.write("🟢 Writing Report... Generating formatted markdown document.")
                time.sleep(1.0)
                status_box.update(label="Research complete!", state="complete")
                api_success = True
            except Exception as local_err:
                import traceback
                status_box.update(label="Research failed!", state="error", expanded=True)
                error_message = f"**In-process execution failed:** `{local_err}`\n\n```python\n{traceback.format_exc()}\n```"

    # Display error if one occurred
    if error_message:
        st.error(error_message)



# Display results
if st.session_state.report:
    st.divider()
    
    # Title & Confidence badge
    conf = st.session_state.confidence
    badge_class = "badge-high" if conf == "High" else "badge-medium" if conf == "Medium" else "badge-low"
    
    st.markdown(
        f"### Research Report "
        f"<span class='confidence-badge {badge_class}'>Confidence: {conf}</span>", 
        unsafe_allow_html=True
    )
    
    # Report container
    st.markdown(f"<div class='content-box'>{st.session_state.report}</div>", unsafe_allow_html=True)
    
    # Clickable sources list
    if st.session_state.sources:
        st.markdown("### Reference Sources")
        for ref in st.session_state.sources:
            idx = ref.get("index", "?")
            title = ref.get("title", "Unknown Source")
            url = ref.get("url", "")
            
            st.markdown(
                f"<div class='source-card'>"
                f"<strong>[{idx}] {title}</strong><br>"
                f"<a href='{url}' target='_blank' style='color:#4B8EFF;'>{url}</a>"
                f"</div>",
                unsafe_allow_html=True
            )
            
    # Feedback section
    st.divider()
    st.markdown("### Rate this Research Report")
    
    if not st.session_state.feedback_submitted:
        rating = st.feedback("stars")
        comment = st.text_input("Any additional feedback? (Optional)", key="feedback_comment")
        
        # Custom button submit
        if rating is not None:
            # We map 0-based star indices (0-4) to 1-5 rating if needed, or if it returns 1-5 already.
            # Streamlit st.feedback("stars") returns 0, 1, 2, 3, 4. So we add 1.
            score = rating + 1
            if st.button("Submit Feedback"):
                try:
                    feedback_data = {
                        "query": st.session_state.last_query,
                        "rating": score,
                        "comment": comment
                    }
                    fb_response = httpx.post(f"{API_URL}/feedback", json=feedback_data)
                    if fb_response.status_code == 200:
                        st.session_state.feedback_submitted = True
                        st.success("Thank you for your feedback! Saved to database.")
                        st.rerun()
                    else:
                        st.error("Failed to save feedback on server.")
                except Exception as e:
                    st.error(f"Error submitting feedback: {e}")
    else:
        st.success("Feedback submitted. Thank you for helping improve the agent!")
elif search_button and not query.strip():
    st.warning("Please enter a research question before submitting.")
