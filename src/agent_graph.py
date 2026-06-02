import os
import json
from typing import List, Dict, Any, TypedDict
# from groq import Groq
from openai import OpenAI
from langgraph.graph import StateGraph, END
from src.retriever import HybridRetriever

from src.ingest import download_and_parse_arxiv_corpus

# ============================================================================
# 1. XAI GROK ENGINE CONFIGURATION
# ============================================================================

# Replace client init:
grok_engine = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)



# Passing http_client=None causes the SDK to internally call SyncHttpxClientWrapper()
# with a 'proxies' argument — which was removed in httpx >= 0.28, causing the crash.
# Simply omit it; the SDK will build a clean default client on its own.
# grok_engine = GrokClient(
#     api_key=xai_grok_key,
#     base_url="https://api.x.ai/v1"
# )

def call_grok_model(prompt_text: str) -> str:
    """
    Sends a clean prompt string straight to the xAI Grok inference engine.
    Temperature=0.1 forces deterministic, highly factual answers.
    """
    completion = grok_engine.chat.completions.create(
        model="mistral",
        messages=[
            {
                "role": "user",
                "content": prompt_text
            }
        ],
        temperature=0.1
    )
    return completion.choices[0].message.content

# ============================================================================
# 2. SEEDED DATA CORPUS (FAISS / BM25 TARGET BACKGROUNDS)
# ============================================================================


mock_corpus = download_and_parse_arxiv_corpus(max_results=500)

# Instantiate the hybrid index module
global_retriever = HybridRetriever(mock_corpus)

# ============================================================================
# 3. STATE DEFINITIONS & STRUCTURAL GRAPH NODES
# ============================================================================

class AgentState(TypedDict):
    question: str
    plan: List[str]
    current_step_index: int
    retrieved_context: List[Dict[str, str]]
    loop_count: int
    raw_synthesis: str
    verified_answer: str
    cited_papers: List[str]
    ablation_mode: str  # Controls which system parts are intentionally broken


def planner_node(state: AgentState) -> Dict[str, Any]:
    """Decomposes complex goals into single target objectives."""
    if state.get("ablation_mode") == "no_planner":
        return {"plan": [state["question"]], "current_step_index": 0, "loop_count": 0, "retrieved_context": []}

    prompt = f"""You are an advanced planning agent. Break this question down into a raw JSON list of 1 or 2 search query objectives.
Question: {state['question']}
Return ONLY a valid JSON array of strings. Do not use markdown backticks or blocks."""

    response = call_grok_model(prompt).strip()
    try:
        clean_json = response.replace("```json", "").replace("```", "").strip()
        plan = json.loads(clean_json)
    except Exception:
        plan = [state["question"]]

    return {"plan": plan, "current_step_index": 0, "loop_count": 0, "retrieved_context": []}


def retriever_node(state: AgentState) -> Dict[str, Any]:
    """Generates an optimized lookup key string and queries the index."""
    current_step = state["plan"][state["current_step_index"]]

    prompt = f"Provide a single string of simple search keywords to find research text on: {current_step}. Return only the keywords."
    search_query = call_grok_model(prompt).strip()

    disable_hybrid_flag = (state.get("ablation_mode") == "no_hybrid")
    new_chunks = global_retriever.retrieve(search_query, top_k=2, disable_hybrid=disable_hybrid_flag)

    updated_context = list(state["retrieved_context"])
    for chunk in new_chunks:
        if chunk not in updated_context:
            updated_context.append(chunk)

    return {
        "retrieved_context": updated_context,
        "current_step_index": state["current_step_index"] + 1,
        "loop_count": state["loop_count"] + 1
    }


def should_continue_loop(state: AgentState) -> str:
    """Evaluates information abundance to break out of the processing loops."""
    if state.get("ablation_mode") == "no_reflector":
        return "synthesize"

    if state["current_step_index"] >= len(state["plan"]) or state["loop_count"] >= 3:
        return "synthesize"

    prompt = f"""Question: {state['question']}\nGathered Facts: {json.dumps(state['retrieved_context'])}
Do we have enough complete evidence to answer accurately? Reply with exactly 'YES' or 'NO'."""

    decision = call_grok_model(prompt).strip().upper()
    return "synthesize" if "YES" in decision else "retriever"


def synthesizer_node(state: AgentState) -> Dict[str, Any]:
    """Drafts an integrated text response embedded with bracketed IDs."""
    context_to_use = state["retrieved_context"]
    if state.get("ablation_mode") == "baseline" and len(context_to_use) > 0:
        context_to_use = [context_to_use[0]]

    formatted_context = "\n".join([f"[{c['arxiv_id']}]: {c['text']}" for c in context_to_use])

    prompt = f"""Synthesize a rigorous research answer to the question using ONLY the facts provided.
You MUST embed explicit bracket citations matching their source numbers exactly (e.g., [2403.12345]).
Question: {state['question']}
Facts:
{formatted_context}
Answer:"""

    response = call_grok_model(prompt).strip()
    return {"raw_synthesis": response}


def citation_verifier_node(state: AgentState) -> Dict[str, Any]:
    """Fact-checks citations against original data strings to isolate hallucinations."""
    if state.get("ablation_mode") == "no_verifier":
        cited_papers = [c["arxiv_id"] for c in state["retrieved_context"] if c["arxiv_id"] in state["raw_synthesis"]]
        return {"verified_answer": state["raw_synthesis"], "cited_papers": list(set(cited_papers))}

    synthesis = state["raw_synthesis"]
    context = state["retrieved_context"]

    prompt = f"""You are a strict academic verifier. Read the text and compare it to the true source fragments.
Remove any sentence where the statement is not completely backed by the text in the context file.
Draft Text: {synthesis}
True Context Documents: {json.dumps(context)}
Return only the clean, verified text containing valid statements and correct citations."""

    verified_answer = call_grok_model(prompt).strip()

    cited_papers = []
    for chunk in context:
        paper_id = chunk["arxiv_id"]
        if paper_id in verified_answer and paper_id not in cited_papers:
            cited_papers.append(paper_id)

    return {"verified_answer": verified_answer, "cited_papers": cited_papers}


# ============================================================================
# 4. COMPILING THE STATE SYSTEM GRAPH
# ============================================================================

builder = StateGraph(AgentState)

builder.add_node("planner", planner_node)
builder.add_node("retriever", retriever_node)
builder.add_node("synthesizer", synthesizer_node)
builder.add_node("verifier", citation_verifier_node)

builder.set_entry_point("planner")
builder.add_edge("planner", "retriever")

builder.add_conditional_edges(
    "retriever",
    should_continue_loop,
    {
        "retriever": "retriever",
        "synthesize": "synthesizer"
    }
)
builder.add_edge("synthesizer", "verifier")
builder.add_edge("verifier", END)

agent_app = builder.compile()