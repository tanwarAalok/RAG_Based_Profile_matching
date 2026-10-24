# RAG-Based Profile Matching

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white) ![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?logo=langchain&logoColor=white) ![Chroma](https://img.shields.io/badge/Vector_DB-Chroma-FF6446) ![OpenRouter](https://img.shields.io/badge/LLM-OpenRouter-6566F1)

Resume screening with retrieval-augmented generation. PDF resumes are parsed, enriched with LLM-extracted metadata, chunked and embedded into a Chroma vector store. A job description is then matched against the store, and an LLM scores each shortlisted candidate with matched skills, supporting excerpts and its reasoning, returned as structured JSON.

## How it works

```
resumes/*.pdf ──▶ pypdf ──▶ LLM metadata extraction ──▶ chunking ──▶ embeddings ──▶ Chroma
                            (name, skills, years, education)                        │
                                                                                    ▼
job description ──▶ semantic search (+ experience filter) ──▶ dedupe by candidate ──▶ LLM scoring ──▶ JSON
```

1. **Ingest (`resume_rag.py`)**
   - Reads the PDFs in `resumes/` with `pypdf`.
   - Extracts `name`, `skills`, `experience_years` and `education` with an LLM and a Pydantic output parser.
   - Splits the text with `RecursiveCharacterTextSplitter` (1,000-character chunks, 200 overlap, section-aware separators).
   - Embeds the chunks with `text-embedding-3-small` and stores them in a persistent Chroma collection, with the metadata on every chunk.
2. **Match (`job_matcher.py`)**
   - Runs a similarity search for the job description, optionally filtered by minimum years of experience.
   - Groups chunks by candidate, so each person is evaluated once with all their relevant excerpts.
   - Asks the LLM, acting as an ATS evaluator, for a 0–100 match score, matched skills, relevant excerpts and reasoning.
   - Sorts by score and returns the top matches as JSON.

## Example output

```json
{
  "job_description": "Looking for a Senior Data Scientist...",
  "top_matches": [
    {
      "candidate_name": "...",
      "resume_path": "./resumes/...pdf",
      "match_score": 86,
      "matched_skills": ["Python", "Machine Learning", "LangChain"],
      "relevant_excerpts": ["..."],
      "reasoning": "..."
    }
  ]
}
```

## Getting started

Requires Python 3.10+ and an [OpenRouter](https://openrouter.ai/) API key (used for both the chat model and the embeddings).

```bash
git clone https://github.com/tanwarAalok/RAG_Based_Profile_matching.git
cd RAG_Based_Profile_matching
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
echo "OPENROUTER_API_KEY=your-openrouter-key" > .env
```

Build the vector store from the sample resumes, then run a match:

```bash
python resume_rag.py     # parses, embeds and stores resumes in ./chroma_db
python job_matcher.py    # matches the sample job description and prints JSON
```

Edit the `jd` string at the bottom of `job_matcher.py`, or import `match_job(job_description, top_k, min_experience)` from your own code.

## Project structure

```
├── resumes/          # 30 sample PDF resumes
├── resume_rag.py     # ingestion: parse, extract metadata, chunk, embed, store
├── job_matcher.py    # retrieval and LLM scoring, returns JSON
└── requirements.txt
```

## Tech stack

LangChain (LCEL chains, prompt templates, Pydantic output parsers) · Chroma · OpenAI `text-embedding-3-small` and Claude 3 Haiku through OpenRouter · pypdf
