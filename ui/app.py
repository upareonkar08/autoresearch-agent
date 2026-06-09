import os
import sys
import time
import httpx
import streamlit as st

# Add repository root to python path to prevent ModuleNotFoundError
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import core modules directly for in-process fallback
from core.memory import get_user_history
from core.collaboration import get_topic_feed, get_all_topics

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
    
    /* Evaluation scorecard styling */
    .score-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .score-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #4B8EFF;
        margin-bottom: 0.2rem;
    }
    
    /* Collaborative activity feed */
    .collab-item {
        background: rgba(255, 255, 255, 0.02);
        border-left: 3px solid #ff4b4b;
        border-radius: 0 6px 6px 0;
        padding: 0.6rem 0.8rem;
        margin-bottom: 0.6rem;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar configurations
with st.sidebar:
    st.image("https://img.icons8.com/nolan/128/artificial-intelligence.png", width=60)
    st.markdown("### Profile Settings")
    
    user_id = st.text_input("User ID", value="user_1", help="Restores your memory across sessions.")
    username = st.text_input("Username", value="Researcher_1", help="Identifies your edits in collaboration.")
    
    st.divider()
    st.markdown("### Collaborative Workspaces")
    topic_id = st.text_input("Topic Channel ID (Optional)", placeholder="e.g. generative_ai", help="Connects you to a shared workspace.")
    
    # Live collaborative feed
    if topic_id.strip():
        st.markdown(f"#### Shared Activity Feed: `{topic_id}`")
        try:
            # Try getting via API first
            try:
                collab_data = httpx.get(f"{API_URL}/collaborative/topic/{topic_id}").json()
            except Exception:
                # Fallback to local import if API offline
                collab_data = get_topic_feed(topic_id)
                
            feed = collab_data.get("feed", [])
            users = collab_data.get("users", {})
            
            if users:
                st.caption(f"Active Researchers: {', '.join(users.keys())}")
            
            if not feed:
                st.info("Feed is currently empty. Start searching to publish logs.")
            else:
                for item in reversed(feed[-5:]):  # Display last 5 events
                    user_tag = f"**{item.get('username')}**"
                    msg = item.get('message', '')
                    st.markdown(f"<div class='collab-item'>{msg}</div>", unsafe_allow_html=True)
        except Exception as e:
            st.caption(f"Could not load shared feed: {e}")
            
    st.divider()
    st.markdown("### Memory History")
    if user_id.strip():
        try:
            # Try getting via API first
            try:
                hist = httpx.get(f"{API_URL}/memory/{user_id}").json()
            except Exception:
                # Fallback to local import if API offline
                hist = get_user_history(user_id)
                
            sessions = hist.get("sessions", [])
            if not sessions:
                st.caption("No past search history recorded.")
            else:
                for s in reversed(sessions[-3:]):
                    with st.expander(f"📜 {s.get('query')[:35]}..."):
                        st.caption(f"Date: {time.strftime('%Y-%m-%d %H:%M', time.localtime(s.get('timestamp', 0)))}")
                        findings = s.get("findings", [])
                        for f in findings[:3]:
                            st.markdown(f"- {f}")
        except Exception as e:
            st.caption(f"Could not load memory: {e}")

# Main Header
st.markdown("<div class='main-title'>AutoResearch Agent</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Autonomous cross-source deep research engine with Multimodal, Memory & Collaboration</div>", unsafe_allow_html=True)

# Session state initialization
if "report" not in st.session_state:
    st.session_state.report = None
if "sources" not in st.session_state:
    st.session_state.sources = []
if "confidence" not in st.session_state:
    st.session_state.confidence = None
if "eval_scores" not in st.session_state:
    st.session_state.eval_scores = {}
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = False

# Domain & Query controls
col_dom, col_query = st.columns([1, 4])
with col_dom:
    domain = st.selectbox(
        "Domain Specialist Mode",
        options=["General", "Medical", "Legal", "Tech", "Finance"],
        help="Adjusts search strategies, prompts, and citations based on the selection."
    )
    
with col_query:
    query = st.text_input(
        "What research question would you like to investigate?",
        placeholder="e.g., What are the latest trends in generative AI in 2025?",
        value="What are the latest trends in generative AI in 2025?"
    )

# Multimodal inputs
col_pdf, col_img = st.columns(2)
with col_pdf:
    uploaded_file = st.file_uploader(
        "Upload reference PDF document (Optional)",
        type=["pdf"],
        help="Will be indexed in ChromaDB alongside search results."
    )
with col_img:
    uploaded_images = st.file_uploader(
        "Upload reference images or charts (Optional / Multimodal)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        help="Uploaded charts will be analyzed using Claude Vision and indexed."
    )

search_button = st.button("Start Research", type="primary", use_container_width=True)

if search_button and query.strip():
    # Reset states
    st.session_state.report = None
    st.session_state.sources = []
    st.session_state.confidence = None
    st.session_state.eval_scores = {}
    st.session_state.last_query = query
    st.session_state.feedback_submitted = False
    
    # Progress display using st.status
    with st.status("Initializing research pipeline...", expanded=True) as status_box:
        
        status_box.update(label=f"Planning {domain} search strategy...", state="running")
        st.write("🟢 Planning... Decomposing query into domain-specialized sub-queries.")
        time.sleep(1.0)
        
        status_box.update(label="Searching sources (Web & YouTube)...", state="running")
        st.write("🟢 Searching... Executing queries via Tavily (capturing video transcripts if found).")
        time.sleep(1.0)
        
        status_box.update(label="Reading and indexing findings...", state="running")
        st.write("🟢 Reading... Chunking content, analyzing uploaded images/charts, and indexing in ChromaDB.")
        time.sleep(1.0)
        
        status_box.update(label=f"Synthesizing {domain} findings...", state="running")
        st.write(f"🟢 Reasoning... Cross-analyzing sources utilizing a {domain} persona.")
        time.sleep(1.0)
        
        # Call backend API
        api_success = False
        try:
            files = []
            data = {
                "query": query,
                "domain": domain,
                "user_id": user_id,
                "username": username,
                "topic_id": topic_id
            }
            
            if uploaded_file:
                files.append(("file", (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")))
            if uploaded_images:
                for img in uploaded_images:
                    files.append(("images", (img.name, img.getvalue(), "image/png")))
                    
            # Make request
            response = httpx.post(
                f"{API_URL}/research",
                data=data,
                files=files if files else None,
                timeout=180.0
            )
            
            if response.status_code == 200:
                result = response.json()
                st.session_state.report = result.get("report", "")
                st.session_state.sources = result.get("sources", [])
                st.session_state.confidence = result.get("confidence", "Medium")
                st.session_state.eval_scores = result.get("eval_scores", {})
                api_success = True
                
                status_box.update(label="Writing final report & evaluating...", state="running")
                st.write("🟢 Evaluating... Report compiled and scored successfully.")
                time.sleep(0.5)
                status_box.update(label="Research complete!", state="complete")
            else:
                st.warning(f"Backend API returned status {response.status_code}. Attempting local fallback.")
        except Exception as e:
            st.info("Backend API is offline. Falling back to in-process execution...")
            
        # In-process execution fallback
        error_message = None
        if not api_success:
            try:
                from core.graph import research_graph
                
                pdf_bytes = uploaded_file.getvalue() if uploaded_file else None
                pdf_filename = uploaded_file.name if uploaded_file else None
                
                images_payload = []
                if uploaded_images:
                    for img in uploaded_images:
                        images_payload.append({
                            "bytes": img.getvalue(),
                            "filename": img.name
                        })
                
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
                
                # Execute graph locally
                result = research_graph.invoke(initial_state)
                
                st.session_state.report = result.get("final_report", "")
                reasoning = result.get("reasoning", {})
                st.session_state.sources = reasoning.get("references", [])
                st.session_state.confidence = reasoning.get("confidence_level", "Medium")
                st.session_state.eval_scores = result.get("eval_scores", {})
                
                status_box.update(label="Writing final report & evaluating...", state="running")
                st.write("🟢 Evaluating... Report compiled and scored successfully.")
                time.sleep(0.5)
                status_box.update(label="Research complete!", state="complete")
                api_success = True
            except Exception as local_err:
                import traceback
                status_box.update(label="Research failed!", state="error", expanded=True)
                error_message = f"**In-process execution failed:** `{local_err}`\n\n```python\n{traceback.format_exc()}\n```"
                
    if error_message:
        st.error(error_message)

# Display Results & Evaluation Scorecard
if st.session_state.report:
    st.divider()
    
    # Title & Confidence badge
    conf = st.session_state.confidence
    badge_class = "badge-high" if conf == "High" else "badge-medium" if conf == "Medium" else "badge-low"
    
    st.markdown(
        f"### Research Report ({domain} Specialist Mode) "
        f"<span class='confidence-badge {badge_class}'>Confidence: {conf}</span>", 
        unsafe_allow_html=True
    )
    
    # Display Report content
    st.markdown(f"<div class='content-box'>{st.session_state.report}</div>", unsafe_allow_html=True)
    
    # Clickable references
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
            
    # Evaluation Module Dashboard
    evals = st.session_state.eval_scores
    if evals:
        st.divider()
        st.markdown("### Report Quality Evaluation Scores")
        
        # Format evaluation metrics into columns
        c_acc, c_cov, c_cit, c_cla, c_avg = st.columns(5)
        
        with c_acc:
            acc = evals.get("accuracy", {"score": 0.0, "reason": ""})
            st.markdown(
                f"<div class='score-card'>"
                f"<div>Accuracy</div>"
                f"<div class='score-value'>{acc.get('score', 0.0):.1f}</div>"
                f"</div>",
                unsafe_allow_html=True
            )
            with st.expander("Justification"):
                st.caption(acc.get("reason", ""))
                
        with c_cov:
            cov = evals.get("coverage", {"score": 0.0, "reason": ""})
            st.markdown(
                f"<div class='score-card'>"
                f"<div>Coverage</div>"
                f"<div class='score-value'>{cov.get('score', 0.0):.1f}</div>"
                f"</div>",
                unsafe_allow_html=True
            )
            with st.expander("Justification"):
                st.caption(cov.get("reason", ""))
                
        with c_cit:
            cit = evals.get("citation_quality", {"score": 0.0, "reason": ""})
            st.markdown(
                f"<div class='score-card'>"
                f"<div>Citation Quality</div>"
                f"<div class='score-value'>{cit.get('score', 0.0):.1f}</div>"
                f"</div>",
                unsafe_allow_html=True
            )
            with st.expander("Justification"):
                st.caption(cit.get("reason", ""))
                
        with c_cla:
            cla = evals.get("clarity", {"score": 0.0, "reason": ""})
            st.markdown(
                f"<div class='score-card'>"
                f"<div>Clarity</div>"
                f"<div class='score-value'>{cla.get('score', 0.0):.1f}</div>"
                f"</div>",
                unsafe_allow_html=True
            )
            with st.expander("Justification"):
                st.caption(cla.get("reason", ""))
                
        with c_avg:
            avg_score = evals.get("average_score", 0.0)
            st.markdown(
                f"<div class='score-card' style='border-color:#4B8EFF; background: rgba(75, 142, 255, 0.08);'>"
                f"<div style='color:#4B8EFF; font-weight:700;'>Overall Average</div>"
                f"<div class='score-value' style='color:#4B8EFF;'>{avg_score:.1f}</div>"
                f"</div>",
                unsafe_allow_html=True
            )
            
    # Feedback section
    st.divider()
    st.markdown("### Rate this Research Report")
    
    if not st.session_state.feedback_submitted:
        rating = st.feedback("stars")
        comment = st.text_input("Any additional feedback? (Optional)", key="feedback_comment")
        
        if rating is not None:
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
                    # Fallback write directly to feedback file in case API server offline
                    try:
                        from api.main import DATA_DIR
                        f_file = os.path.join(DATA_DIR, "feedback.json")
                        feedback_list = []
                        if os.path.exists(f_file):
                            with open(f_file, "r", encoding="utf-8") as f_in:
                                content_in = f_in.read().strip()
                                if content_in:
                                    feedback_list = json.loads(content_in)
                        feedback_list.append(feedback_data)
                        with open(f_file, "w", encoding="utf-8") as f_out:
                            json.dump(feedback_list, f_out, indent=4)
                        st.session_state.feedback_submitted = True
                        st.success("Thank you for your feedback! Saved locally.")
                        st.rerun()
                    except Exception as fallback_err:
                        st.error(f"Error submitting feedback: {fallback_err}")
    else:
        st.success("Feedback submitted. Thank you for helping improve the agent!")
elif search_button and not query.strip():
    st.warning("Please enter a research question before submitting.")
