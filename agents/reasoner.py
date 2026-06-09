import json
import logging
from core.llm import get_llm

logger = logging.getLogger(__name__)

def reason_over_sources(query: str, retrieved_chunks: list[dict], domain: str = "General") -> dict:
    """
    Accepts the original research question and retrieved documents from ChromaDB,
    uses Claude to reason across all sources simultaneously adjusting the persona and
    analytical style based on the domain, resolves any contradictions, and returns
    a structured reasoning output as a dictionary.
    
    Args:
        query (str): The original user query.
        retrieved_chunks (list[dict]): List of retrieved document chunks from Reader.
        domain (str): The selected domain (Medical, Legal, Tech, Finance, General).
        
    Returns:
        dict: A structured reasoning dictionary.
    """
    if not retrieved_chunks:
        return {
            "key_findings": ["No relevant search results were retrieved to answer this query."],
            "confidence_level": "Low",
            "confidence_reasoning": "No sources were available for analysis.",
            "references": [],
            "gaps": ["Missing all search data."],
            "contradictions_resolved": []
        }
        
    # Format retrieved chunks as source text for the LLM prompt
    sources_text = ""
    for idx, chunk in enumerate(retrieved_chunks):
        meta = chunk.get("metadata", {})
        url = meta.get("source_url", "Unknown Source")
        title = meta.get("title", "Untitled")
        content = chunk.get("content", "")
        sources_text += f"--- SOURCE {idx + 1} ---\nTitle: {title}\nURL: {url}\nContent: {content}\n\n"
        
    # Domain personas and constraints
    domain_personas = {
        "General": (
            "You are an expert analytical researcher. Present clear facts, dates, figures, and direct answers."
        ),
        "Medical": (
            "You are a medical research officer. Focus on clinical trial metrics (sample size, double-blind status), "
            "clinical efficacy, safety margins, peer-reviewed medical publications, and official healthcare guidelines. "
            "Highlight any clinical safety issues or FDA status."
        ),
        "Legal": (
            "You are a senior legal counsel. Focus on statutory law, relevant regulatory bodies, judicial precedents, "
            "legal tests, jurisdictional boundaries, and potential liabilities or compliance issues."
        ),
        "Tech": (
            "You are a principal systems architect. Focus on software/hardware benchmarks, performance throughput, "
            "version compatibilities, APIs, developer standards, architectural patterns, and engineering trade-offs."
        ),
        "Finance": (
            "You are a Chartered Financial Analyst (CFA). Focus on financial statement metrics (revenue growth, margins, "
            "earnings multiples, CAGR), SEC filing quotes, market capitalization, valuation dynamics, and financial risk profiles."
        )
    }
    
    persona = domain_personas.get(domain, domain_personas["General"])
    
    system_prompt = (
        f"{persona}\n\n"
        "Your task is to analyze multiple search result sources to answer a user's research query. "
        "You must reason across all sources simultaneously, identify and resolve any contradictions "
        "or discrepancies between them, and evaluate what information is missing.\n\n"
        "You must return your analysis strictly as a JSON object with the following keys:\n"
        "1. \"key_findings\": List of strings outlining the core findings. Each finding should include a brief bracketed citation like [Source 1] or [Source 2].\n"
        "2. \"confidence_level\": One of \"Low\", \"Medium\", or \"High\".\n"
        "3. \"confidence_reasoning\": A short paragraph explaining why you chose this confidence level based on source reliability, completeness, and consistency.\n"
        "4. \"references\": A list of dictionaries representing the unique sources used. Each dict must have \"index\" (integer starting at 1), \"title\", and \"url\".\n"
        "5. \"gaps\": List of strings explaining what information was not found or is missing from the sources.\n"
        "6. \"contradictions_resolved\": List of strings describing any contradictions found between sources and how you resolved them (or an empty list if none).\n\n"
        "Do not include any explanation or formatting outside the JSON object itself."
    )
    
    user_content = (
        f"Original Research Query: {query}\n\n"
        f"Retrieved Source Chunks:\n{sources_text}"
    )
    
    try:
        llm = get_llm(max_tokens=2000)
        
        # Check if we are running in Mock mode
        from core.llm import MockChatAnthropic
        if isinstance(llm, MockChatAnthropic):
            logger.info(f"Mock Reasoner: Analyzing {domain} sources...")
            unique_sources = {}
            for chunk in retrieved_chunks:
                meta = chunk.get("metadata", {})
                url = meta.get("source_url", "")
                title = meta.get("title", "Untitled Source")
                if url and url not in unique_sources:
                    unique_sources[url] = title
                    
            refs = [{"index": idx + 1, "title": title, "url": url} 
                    for idx, (url, title) in enumerate(unique_sources.items())]
            
            # Custom findings based on domain
            findings = []
            for idx, chunk in enumerate(retrieved_chunks[:4]):
                content = chunk.get("content", "")
                title = chunk.get("metadata", {}).get("title", "Source")
                snippet = content[:120] + "..." if len(content) > 120 else content
                
                if domain == "Medical":
                    findings.append(f"Clinical data from {title}: Patient trial indicates '{snippet}' [Source {idx + 1}]")
                elif domain == "Legal":
                    findings.append(f"Statutory authority in {title}: Ruling stipulates '{snippet}' [Source {idx + 1}]")
                elif domain == "Tech":
                    findings.append(f"Architectural specs in {title}: Benchmark reveals '{snippet}' [Source {idx + 1}]")
                elif domain == "Finance":
                    findings.append(f"Financial analysis in {title}: Earnings show '{snippet}' [Source {idx + 1}]")
                else:
                    findings.append(f"Key data points from {title}: '{snippet}' [Source {idx + 1}]")
                    
            if not findings:
                findings = [f"Aggregated general evidence for {query}."]
                
            return {
                "key_findings": findings,
                "confidence_level": "High" if len(refs) >= 3 else "Medium",
                "confidence_reasoning": f"Derived by synthesizing {len(refs)} sources in accordance with {domain} research guidelines.",
                "references": refs,
                "gaps": [f"Granular timelines beyond 2026 for {query}."],
                "contradictions_resolved": []
            }
            
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
        
        reasoning_data = json.loads(content)
        return reasoning_data
        
    except Exception as e:
        logger.error(f"Error in Reasoner Agent: {e}")
        # Build fallback
        unique_sources = {}
        for chunk in retrieved_chunks:
            meta = chunk.get("metadata", {})
            url = meta.get("source_url", "")
            title = meta.get("title", "Untitled Source")
            if url and url not in unique_sources:
                unique_sources[url] = title
                
        refs = [{"index": idx + 1, "title": title, "url": url} 
                for idx, (url, title) in enumerate(unique_sources.items())]
                
        return {
            "key_findings": [f"Fallback data extracted. Detailed {domain} reasoning failed due to internal error."],
            "confidence_level": "Low",
            "confidence_reasoning": str(e),
            "references": refs,
            "gaps": ["Reasoner node failure"],
            "contradictions_resolved": []
        }
