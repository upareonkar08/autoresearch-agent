import json
import logging
from core.llm import get_llm

logger = logging.getLogger(__name__)

def generate_report(query: str, reasoning_data: dict, domain: str = "General") -> str:
    """
    Accepts the original research question, structured reasoning data, and domain,
    queries Claude to compile them into a beautifully styled markdown report specialized
    for the selected domain, and returns it.
    
    Args:
        query (str): The original user query.
        reasoning_data (dict): The structured dictionary returned by the Reasoner.
        domain (str): The selected domain (Medical, Legal, Tech, Finance, General).
        
    Returns:
        str: The final styled markdown report.
    """
    # Domain-specific style requirements
    domain_styles = {
        "General": (
            "Use standard headings:\n"
            "1. # Executive Summary\n"
            "2. # Key Findings (with bracketed citations like [1] or [2])\n"
            "3. # Detailed Analysis\n"
            "4. # Source List\n"
            "5. # Confidence Score\n"
            "6. # Gaps & Limitations"
        ),
        "Medical": (
            "Format the report in a Medical Review style:\n"
            "1. # Clinical Executive Summary\n"
            "2. # Key Medical Findings (with NLM-style bracketed indices like [1])\n"
            "3. # Clinical Trial & Evidence Analysis (focus on trial sizes, safety data, and FDA status)\n"
            "4. # Indexed Medical Bibliography\n"
            "5. # Clinical Evidence Confidence Score\n"
            "6. # Gaps, Contraindications & Limitations"
        ),
        "Legal": (
            "Format the report in a Legal Opinion Brief style:\n"
            "1. # Executive Summary (Statement of Facts)\n"
            "2. # Key Legal Precedents & Holdings (using citations like [1])\n"
            "3. # Statutory & Regulatory Analysis (governing authorities and liabilities)\n"
            "4. # Legal Citations & Table of Authorities\n"
            "5. # Legal Liability Confidence Score\n"
            "6. # Jurisdictional Gaps & Statutory Caveats"
        ),
        "Tech": (
            "Format the report in a Technical Specification Blueprint style:\n"
            "1. # Architecture & Executive Summary\n"
            "2. # Technical Findings & Benchmarks (with index citations like [1])\n"
            "3. # Systems Design & API Analysis (architectural details, versions, and performance trade-offs)\n"
            "4. # Technical Specifications & References\n"
            "5. # Architectural Confidence Score\n"
            "6. # Technical Debt, Gaps & Edge Cases"
        ),
        "Finance": (
            "Format the report in an Equity Research & Valuation style:\n"
            "1. # Executive Summary (Investment thesis summary)\n"
            "2. # Key Financial & Market Indicators (with index citations like [1])\n"
            "3. # Revenue Analysis & Valuation Dynamics (focus on earnings, CAGRs, multiples, and balance sheet indicators)\n"
            "4. # Financial Data Sources & References\n"
            "5. # Valuation Model Confidence Score\n"
            "6. # Gaps, Financial Risks & Disclaimers"
        )
    }

    style_guide = domain_styles.get(domain, domain_styles["General"])

    system_prompt = (
        "You are an expert technical writer and researcher. Your task is to take a structured reasoning output "
        "about a research query and compile it into a clean, professional, and publication-ready markdown report.\n\n"
        f"Selected Domain: {domain}\n"
        f"Format Guidelines:\n{style_guide}\n\n"
        "Requirements:\n"
        "- The 'Key Findings' section MUST present findings as bullet points ending with citations referring to the source list (e.g., [1] or [2]).\n"
        "- The 'Source List' section MUST list references in a numbered list, including their Title and clickable URL. Format: [Index] [Title](URL)\n"
        "- Keep the tone professional, objective, and clear. Return only the markdown string, starting directly with the first section (do not prefix it with code block quotes or introductory notes)."
    )

    user_content = (
        f"Original Research Query: {query}\n\n"
        f"Structured Reasoning Data (JSON):\n{json.dumps(reasoning_data, indent=2)}"
    )

    try:
        llm = get_llm(max_tokens=3000)
        
        # Check if we are running in Mock mode
        from core.llm import MockChatAnthropic
        if isinstance(llm, MockChatAnthropic):
            logger.info(f"Mock Reporter: Writing report in {domain} style...")
            findings_li = "\n".join([f"- {finding}" for finding in reasoning_data.get("key_findings", [])])
            sources_li = ""
            for ref in reasoning_data.get("references", []):
                idx = ref.get("index", "?")
                title = ref.get("title", "Unknown Source")
                url = ref.get("url", "")
                sources_li += f"{idx}. [{title}]({url})\n"
                
            gaps_li = "\n".join([f"- {gap}" for gap in reasoning_data.get("gaps", [])])
            
            # Map headers dynamically
            if domain == "Medical":
                return (
                    f"# Clinical Executive Summary\n"
                    f"This clinical report evaluates: '{query}'. It compiles trial metrics and efficacy indications from literature.\n\n"
                    f"# Key Medical Findings\n{findings_li}\n\n"
                    f"# Clinical Trial & Evidence Analysis\n"
                    f"The retrieved trials indicate strong initial parameters. Analysis confirms patient safety benchmarks are met.\n\n"
                    f"# Indexed Medical Bibliography\n{sources_li}\n"
                    f"# Clinical Evidence Confidence Score\n**{reasoning_data.get('confidence_level', 'High')}**\n{reasoning_data.get('confidence_reasoning', '')}\n\n"
                    f"# Gaps, Contraindications & Limitations\n{gaps_li}"
                )
            elif domain == "Legal":
                return (
                    f"# Executive Summary (Statement of Facts)\n"
                    f"This legal brief analyzes issues pertaining to: '{query}' based on available statutory summaries.\n\n"
                    f"# Key Legal Precedents & Holdings\n{findings_li}\n\n"
                    f"# Statutory & Regulatory Analysis\n"
                    f"Review of governing statutes suggests compliance standards are applicable as outlined in local case holdings.\n\n"
                    f"# Legal Citations & Table of Authorities\n{sources_li}\n"
                    f"# Legal Liability Confidence Score\n**{reasoning_data.get('confidence_level', 'High')}**\n{reasoning_data.get('confidence_reasoning', '')}\n\n"
                    f"# Jurisdictional Gaps & Statutory Caveats\n{gaps_li}"
                )
            elif domain == "Tech":
                return (
                    f"# Architecture & Executive Summary\n"
                    f"This specification sheet defines the design and performance metrics of: '{query}'.\n\n"
                    f"# Technical Findings & Benchmarks\n{findings_li}\n\n"
                    f"# Systems Design & API Analysis\n"
                    f"System throughput exhibits linear scaling. Benchmarks indicate execution latencies fall within standard levels.\n\n"
                    f"# Technical Specifications & References\n{sources_li}\n"
                    f"# Architectural Confidence Score\n**{reasoning_data.get('confidence_level', 'High')}**\n{reasoning_data.get('confidence_reasoning', '')}\n\n"
                    f"# Technical Debt, Gaps & Edge Cases\n{gaps_li}"
                )
            elif domain == "Finance":
                return (
                    f"# Executive Summary\n"
                    f"This equity report details our investment thesis and valuation multiples regarding: '{query}'.\n\n"
                    f"# Key Financial & Market Indicators\n{findings_li}\n\n"
                    f"# Revenue Analysis & Valuation Dynamics\n"
                    f"CAGR forecasts suggest steady revenue multiples. Valuation ratios align with sector averages.\n\n"
                    f"# Financial Data Sources & References\n{sources_li}\n"
                    f"# Valuation Model Confidence Score\n**{reasoning_data.get('confidence_level', 'High')}**\n{reasoning_data.get('confidence_reasoning', '')}\n\n"
                    f"# Gaps, Financial Risks & Disclaimers\n{gaps_li}"
                )
            else:
                return (
                    f"# Executive Summary\n"
                    f"This report presents findings for: '{query}'.\n\n"
                    f"# Key Findings\n{findings_li}\n\n"
                    f"# Detailed Analysis\n"
                    f"Detailed analysis suggests a strong correlation across sources.\n\n"
                    f"# Source List\n{sources_li}\n"
                    f"# Confidence Score\n**{reasoning_data.get('confidence_level', 'High')}**\n{reasoning_data.get('confidence_reasoning', '')}\n\n"
                    f"# Gaps & Limitations\n{gaps_li}"
                )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        response = llm.invoke(messages)
        report_markdown = response.content.strip()
        
        # Clean up markdown code block wrapping if present
        if report_markdown.startswith("```markdown"):
            report_markdown = report_markdown[11:]
        elif report_markdown.startswith("```"):
            report_markdown = report_markdown[3:]
        if report_markdown.endswith("```"):
            report_markdown = report_markdown[:-3]
        report_markdown = report_markdown.strip()
        
        return report_markdown
        
    except Exception as e:
        logger.error(f"Error in Reporter Agent: {e}")
        # Build fallback report
        findings_li = "\n".join([f"- {finding}" for finding in reasoning_data.get("key_findings", [])])
        sources_li = "\n".join([f"{ref.get('index', '?')}. [{ref.get('title', 'Ref')}]({ref.get('url', '')})" 
                               for ref in reasoning_data.get("references", [])])
        
        return (
            f"# {domain} Research Report\n"
            f"Query: {query}\n\n"
            f"## Key Findings\n{findings_li}\n\n"
            f"## Sources\n{sources_li}"
        )
