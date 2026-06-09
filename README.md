# AutoResearch Agent

AutoResearch Agent is a full-stack, AI-powered autonomous research system designed to answer complex inquiries. Given a research query, the agent breaks the goal down into multiple search paths, executes deep web queries, scrapes relevant pages, and indexes findings in a local vector database. By reasoning across all sources simultaneously, it identifies anomalies or contradictions, resolves differences, and compiles a comprehensive, cited markdown report.

The system is built on top of LangGraph for robust node-based state-graph orchestration, ensuring error boundaries and deterministic flow transitions. Powered by FastAPI for a high-performance backend, Streamlit for a user-friendly dark-themed GUI, and Claude 3.5 Sonnet for advanced planning, reasoning, and report formulation, the agent operates entirely autonomously from start to finish.

## System Architecture

```
                              +--------------------------+
                              |    User Research Query   |
                              +-------------+------------+
                                            |
                                            v
                                 +----------+----------+
                                 |    Planner Agent    |  ( Claude Sonnet )
                                 +----------+----------+
                                            |
                                            v  ( 3-5 Sub-questions )
                                 +----------+----------+
                                 |   Searcher Agent    |  ( Tavily Search API )
                                 +----------+----------+
                                            |
                                            v  ( Web Snippets )
                                 +----------+----------+
                                 |    Reader Agent     |  ( PyMuPDF + Chunking )
                                 +----------+----------+
                                            |
                                            +------------------------+
                                            |                        |
                                            v                        v
                                     [ ChromaDB (local) ]     [ PDF Upload ]
                                            |
                                            v  ( Top N contexts )
                                 +----------+----------+
                                 |   Reasoner Agent    |  ( Claude Sonnet )
                                 +----------+----------+
                                            |
                                            v  ( Resolved reasoning )
                                 +----------+----------+
                                 |   Reporter Agent    |  ( Claude Sonnet )
                                 +----------+----------+
                                            |
                                            v  ( Markdown Report )
                                  [ Streamlit Frontend ]
```

## Setup Instructions

### 1. Prerequisites
- Python 3.11 or higher
- API Keys for **Anthropic** and **Tavily**

### 2. Installation
Clone the repository (or navigate to the workspace directory) and install the python dependencies:
```bash
# Navigate to project folder
cd autoresearch-agent

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in the `autoresearch-agent` directory based on the `.env.example` template:
```bash
copy .env.example .env
```
Fill in your API keys in the `.env` file:
```ini
ANTHROPIC_API_KEY=your-actual-claude-api-key
TAVILY_API_KEY=your-actual-tavily-api-key
```

## How to Run

To run the full stack application, you need to spin up both the FastAPI backend server and the Streamlit frontend UI.

### Step 1: Start the FastAPI Backend
```bash
# Run from the autoresearch-agent directory
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

### Step 2: Start the Streamlit Frontend
```bash
# Run from the autoresearch-agent directory in a new terminal
streamlit run ui/app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser to start searching!

## Example Queries to Test
- `What are the latest trends in generative AI in 2025?`
- `What is the market size of AI in healthcare in India?`
- `Compare recent electric vehicle market shares between US, EU, and China.`
