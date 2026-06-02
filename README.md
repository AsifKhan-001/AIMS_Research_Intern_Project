#  Agentic Deep Research over LLM-Agent Papers
### AIMS DTU Research Intern 2026 — Technical Submission

> **Asif Khan** | June 2026  
> GitHub: [AsifKhan-001/AIMS_Research_Intern_Project](https://github.com/AsifKhan-001/AIMS_Research_Intern_Project.git)

---

## What This Project Does

This project builds an **end-to-end Agentic Deep Research system** over a corpus of 500+ recent LLM-agent papers from arXiv. Instead of answering research questions in a single LLM call, the system runs in a loop — it breaks the question into sub-queries, retrieves relevant paper chunks, reflects on whether it has enough evidence, and only then writes a final answer with proper arXiv citations.

The goal of the assignment is not just to build the system, but to **measure how much each component actually contributes** by running ablation experiments — turning off one component at a time and comparing results.

All 6 required configurations are complete (30/30 questions each), scored using a local LLM judge via `evaluator.py`.

---

## Quick Results

| Configuration | Accuracy (LLM Judge) | Avg Citations/Q | Avg Answer Length | Status |
|---------------|---------------------|-----------------|-------------------|--------|
| **full_agent** | **92.2%** | **2.6** | **1156 chars** | ✅ Complete |
| baseline | 88.5% | 1.2 | 931 chars | ✅ Complete |
| no_planner | 91.3% | 1.6 | 984 chars | ✅ Complete |
| no_reflector | 91.5% | 1.7 | 833 chars | ✅ Complete |
| no_hybrid | 92.8% | 2.5 | 924 chars | ✅ Complete |
| no_verifier | 92.0% | 2.8 | 1175 chars | ✅ Complete |

> Accuracy scored by Mistral 7B (local Ollama) as LLM-as-judge via `src/evaluator.py`

---

## Project Structure

```
Agentic_AI_Deep_Research/
│
├── eval/
│   └── questions.jsonl          # 30 evaluation questions (factoid, comparative, survey)
│
├── predictions/
│   ├── full_agent.jsonl         # Full agentic system — 30/30
│   ├── baseline.jsonl           # Single-shot RAG baseline — 30/30
│   ├── no_planner.jsonl         # Ablation: no query decomposition — 30/30
│   ├── no_reflector.jsonl       # Ablation: no reflection loop — 30/30
│   ├── no_hybrid.jsonl          # Ablation: dense retrieval only — 30/30
│   └── no_verifier.jsonl        # Ablation: no citation verification — 30/30
│
├── src/
│   ├── agent_graph.py           # LangGraph state machine — all 5 agent nodes
│   ├── ingest.py                # arXiv API scraper + chunker
│   ├── retriever.py             # HybridRetriever (BM25 + FAISS + RRF)
│   └── evaluator.py             # LLM-as-judge scoring script
│
├── main.py                      # Runs all 6 ablation configurations
├── requirements.txt             # All dependencies
└── README.md
```

---

## System Architecture

The agent is built as a **LangGraph state machine** with 5 nodes, each independently ablatable via an `ablation_mode` flag:

```
Question
   │
   ▼
┌─────────┐     ablation: no_planner → skip decomposition
│ Planner │  →  breaks question into 1-2 sub-queries
└────┬────┘
     │
     ▼
┌──────────┐    ablation: no_hybrid → FAISS only (no BM25)
│ Retriever│  →  BM25 + FAISS → RRF fusion → top-2 chunks
└────┬─────┘
     │
     ▼
┌───────────┐   ablation: no_reflector → always go to synthesise
│ Reflector │ →  "Is evidence enough?" YES → synthesise, NO → retrieve again (max 3 rounds)
└────┬──────┘
     │
     ▼
┌─────────────┐  ablation: baseline → use only 1 chunk (single-shot)
│ Synthesiser │ →  writes answer with inline [arXiv_id] citations
└──────┬──────┘
       │
       ▼
┌──────────────┐  ablation: no_verifier → skip check, keep raw answer
│ Cit. Verifier│ →  checks each claim against cited passage, removes hallucinated citations
└──────┬───────┘
       │
       ▼
    Answer + cited_papers[]
```

---

## Tech Stack

| Component | Choice | Why |
|-----------|--------|-----|
| LLM Backend | Mistral 7B via Ollama (local) | Free tier, no API key, runs on MacBook M2 |
| Agent Framework | LangGraph | Clean state machine, easy to ablate nodes |
| Dense Retrieval | FAISS + all-MiniLM-L6-v2 | Lightweight, fast, runs locally |
| Sparse Retrieval | BM25 (rank_bm25) | Exact keyword matching for arXiv IDs |
| Score Fusion | Reciprocal Rank Fusion (RRF, k=60) | Scale-invariant, no tuning needed |
| Corpus Source | arXiv API (free, no key) | 500 papers, cs.CL/cs.AI/cs.LG, Jan 2024–Apr 2026 |
| Evaluation | Mistral-as-judge (evaluator.py) | Same free-tier model, self-contained |

---

## LLM Model Details

This project uses **Mistral 7B** running locally via [Ollama](https://ollama.com/) — no cloud API, no credit card, completely free.

```
Model:      mistral:latest
Parameters: 7 billion
Size:       4.4 GB
Backend:    Ollama (OpenAI-compatible local API at http://localhost:11434/v1)
Used for:   Planning, retrieval query generation, synthesis, citation verification, evaluation scoring
```

The OpenAI Python SDK is used to talk to Ollama since it exposes an OpenAI-compatible endpoint:
```python
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
```

---

## How to Clone and Run

### Step 1 — Clone the repo

```bash
git clone https://github.com/AsifKhan-001/AIMS_Research_Intern_Project.git
cd AIMS_Research_Intern_Project
```

### Step 2 — Set up Python environment

```bash
python3 -m venv myenv
source myenv/bin/activate        # Mac/Linux
# myenv\Scripts\activate         # Windows
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Install Ollama and pull Mistral

```bash
# Install Ollama from https://ollama.com/download
# Then pull the model:
ollama pull mistral
```

### Step 5 — Start Ollama (in a separate terminal)

```bash
ollama serve
```

### Step 6 — Run all 6 ablation configurations

```bash
# This runs full_agent, baseline, no_planner, no_reflector, no_hybrid, no_verifier
# and saves all predictions to predictions/*.jsonl
python main.py
```

> ⚠️ This will take time — each configuration processes 30 questions with a 6-second pacing delay. Total runtime ~90 minutes for all 6 configs on a MacBook M2 Air.

### Step 7 — Score the predictions

```bash
python3 src/evaluator.py
```

---

## Requirements

Install all dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Full `requirements.txt` (exact versions used in this project):

```
annotated-types==0.7.0
anyio==4.13.0
certifi==2026.5.20
cffi==2.0.0
charset-normalizer==3.4.7
cryptography==48.0.0
distro==1.9.0
et_xmlfile==2.0.0
faiss-cpu==1.12.0
filelock==3.29.0
fsspec==2026.4.0
google-ai-generativelanguage==0.4.0
google-api-core==2.30.3
google-auth==2.53.0
google-generativeai==0.4.1
googleapis-common-protos==1.75.0
groq==1.4.0
grpcio==1.80.0
grpcio-status==1.62.3
h11==0.16.0
hf-xet==1.5.0
httpcore==1.0.9
httpx==0.27.2
huggingface_hub==0.36.2
idna==3.17
Jinja2==3.1.6
joblib==1.5.3
jsonpatch==1.33
jsonpointer==3.1.1
langchain-core==0.1.42
langgraph==0.0.36
langsmith==0.1.147
MarkupSafe==3.0.3
mpmath==1.3.0
networkx==3.6.1
numpy==1.26.4
openai==1.14.1
openpyxl==3.1.2
orjson==3.11.9
packaging==23.2
pandas==2.2.2
pillow==12.2.0
proto-plus==1.28.0
protobuf==4.25.9
pyasn1==0.6.3
pyasn1_modules==0.4.2
pycparser==3.0
pydantic==2.13.4
pydantic_core==2.46.4
pypdf==4.2.0
python-dateutil==2.9.0.post0
pytz==2026.2
PyYAML==6.0.3
rank-bm25==0.2.2
regex==2026.5.9
requests==2.31.0
requests-toolbelt==1.0.0
safetensors==0.7.0
scikit-learn==1.8.0
scipy==1.17.1
sentence-transformers==2.6.1
setuptools==81.0.0
six==1.17.0
sniffio==1.3.1
sympy==1.14.0
tenacity==8.5.0
threadpoolctl==3.6.0
tokenizers==0.22.2
torch==2.12.0
tqdm==4.67.3
transformers==4.57.6
typing-inspection==0.4.2
typing_extensions==4.15.0
tzdata==2026.2
urllib3==2.7.0
```

The core packages that matter for this project:

| Package | Version | Purpose |
|---------|---------|---------|
| `openai` | 1.14.1 | OpenAI-compatible client to talk to local Ollama |
| `langgraph` | 0.0.36 | Agent state machine framework |
| `sentence-transformers` | 2.6.1 | all-MiniLM-L6-v2 embeddings for FAISS |
| `faiss-cpu` | 1.12.0 | Vector similarity search index |
| `rank-bm25` | 0.2.2 | BM25 keyword search engine |
| `torch` | 2.12.0 | Required by sentence-transformers |
| `transformers` | 4.57.6 | Required by sentence-transformers |
| `numpy` | 1.26.4 | Array operations for retrieval scoring |
| `groq` | 1.4.0 | Groq SDK (kept in env, not used in final version) |

> **Note:** `faiss-gpu` can be used instead of `faiss-cpu` if you have a CUDA GPU. On Apple Silicon (M2), `faiss-cpu` works fine.

---

## Prediction File Format

Each `.jsonl` file in `predictions/` follows the submission format exactly:

```json
{
  "id": "q01",
  "answer": "The memory-architecture variant in the Mem0 paper [2605.30465] that augments the vector store with an entity-relation graph...",
  "cited_papers": ["2605.30465", "2606.01435"]
}
```

| Field | Description |
|-------|-------------|
| `id` | Matches the question ID from `eval/questions.jsonl` |
| `answer` | Natural language answer with inline `[arXiv_id]` citations |
| `cited_papers` | Flat list of arXiv IDs used as evidence (no version suffix, no URL) |

All 6 prediction files contain exactly 30 lines — one per question, no duplicates.

---

## Ablation Design

Each component is disabled by passing an `ablation_mode` string. No code changes needed — just a different flag:

| Flag | What Gets Disabled | What Stays Active |
|------|--------------------|-------------------|
| `"none"` | Nothing (full agent) | All 5 components |
| `"baseline"` | Planner + Reflector + Verifier (1 chunk only) | Retriever + Synthesiser |
| `"no_planner"` | Query decomposition | Retriever + Reflector + Synthesiser + Verifier |
| `"no_reflector"` | Sufficiency check loop | Planner + Retriever + Synthesiser + Verifier |
| `"no_hybrid"` | BM25 (dense-only retrieval) | Planner + FAISS + Reflector + Synthesiser + Verifier |
| `"no_verifier"` | Citation fact-checking | Planner + Retriever + Reflector + Synthesiser |

---

## Key Findings

- **Full agent vs baseline:** +3.7% accuracy, +1.4 more citations per question, +225 chars longer answers — the agentic loop clearly helps.
- **Most impactful component:** The **planner** — removing it dropped citations from 2.6 to 1.6. Without query decomposition, the retriever misses relevant papers on multi-paper questions.
- **Surprising result:** `no_hybrid` scored the highest accuracy (92.8%). Dense-only retrieval produces focused answers that score better with the LLM judge, even though hybrid retrieval finds more papers overall.
- **Citation verifier tradeoff:** Without it, citation count goes up (2.8 vs 2.6) but those extra citations are unverified — the verifier trades quantity for trustworthiness.
- **Honest limitation:** The judge model (Mistral) is the same as the generator. This is not truly independent scoring — it is a known limitation of the evaluation setup.

---

## Corpus Details

| Property | Value |
|----------|-------|
| Source | arXiv API (free, no key required) |
| Categories | cs.CL, cs.AI, cs.LG |
| Date Range | January 2024 – April 2026 |
| Papers Downloaded | 500 |
| Total Chunks | 2,743 |
| Chunk Size | 400 characters |
| Chunk Overlap | 100 characters |
| Content | Title + Abstract per paper |

---

## Evaluation Questions Breakdown

| Type | Count | Description |
|------|-------|-------------|
| Factoid | 10 | Single-fact lookup from one paper |
| Comparative | 10 | Compare 2+ papers on a specific dimension |
| Survey | 10 | Summarise findings across 4+ papers |
| **Total** | **30** | Provided in `eval/questions.jsonl` |

---

## Notes for Reproducibility

- The arXiv API returns **live results** at run time, sorted by newest first. If you run the system on a different date, the corpus will include newer papers. This may slightly change results.
- All randomness is controlled via `temperature=0.1` in all Mistral calls.
- The 6-second pacing delay in `main.py` prevents Ollama from being overwhelmed on local hardware. You can reduce this on faster machines.
- `out_f.flush()` is called after every question so partial results are saved even if the run is interrupted.

---

## References

- Shinn et al. (2023). *Reflexion: Language agents with verbal reinforcement learning.* NeurIPS 2023.
- Yao et al. (2023). *ReAct: Synergizing reasoning and acting in language models.* ICLR 2023.
- Asai et al. (2023). *Self-RAG: Learning to retrieve, generate, and critique.* arXiv:2310.11511.
- Khot et al. (2023). *Decomposed prompting.* ICLR 2023.
- Es et al. (2023). *RAGAS: Automated evaluation of retrieval augmented generation.* arXiv:2309.15217.

---

*Built for AIMS DTU Research Intern 2026 — Agentic Deep Research assignment.*  
*All components run on free tiers. No cloud API key or credit card required anywhere in the stack.*
