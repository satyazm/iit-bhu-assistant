# IIT BHU Assistant

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about IIT
BHU — academics, hostels, placements, or anything else — grounded in your own
indexed PDF documents, with sources cited on every answer.

Built as a placement project to demonstrate practical, production-style RAG
techniques: hybrid retrieval, reciprocal rank fusion, cross-encoder reranking,
and a real (not toy) Streamlit interface.

---

## Table of contents

- [What it does](#what-it-does)
- [Architecture](#architecture)
- [Retrieval pipeline in depth](#retrieval-pipeline-in-depth)
- [Project structure](#project-structure)
- [Setup](#setup)
- [Adding documents](#adding-documents)
- [Running the app](#running-the-app)
- [Configuration](#configuration)
- [How each file works](#how-each-file-works)
- [Talking points for interviews](#talking-points-for-interviews)
- [Troubleshooting](#troubleshooting)
- [Ideas to extend](#ideas-to-extend)

---

## What it does

1. You give it PDFs — academic circulars, hostel rules, placement brochures,
   your own notes, anything.
2. It chunks and embeds them into a local vector database.
3. You ask a question in a chat interface.
4. It retrieves the most relevant chunks using a hybrid dense + keyword search,
   reranks them for precision, and asks Gemini to answer **using only that
   retrieved context** — citing the source file and page for every claim.
5. If the answer isn't in your documents, it says so instead of making
   something up.

This is the core RAG pattern used in production systems (internal knowledge
bases, support bots, legal/medical document QA) scaled down to something you
can build, explain, and demo end to end.

---

## Architecture

```
                     ┌─────────────────────┐
   PDF files  ─────► │   src/ingest.py      │
  (or upload  │      │  read → chunk →      │
   via app)   │      │  embed → upsert       │
              │      └──────────┬───────────┘
              │                 │
              │                 ▼
              │      ┌─────────────────────┐
              │      │  Chroma vector DB     │
              │      │  (data/chroma_db/)    │
              │      └──────────┬───────────┘
              │                 │
              ▼                 ▼
     ┌─────────────────────────────────────┐
     │            src/app.py                │
     │         (Streamlit chat UI)          │
     │                                      │
     │  question                            │
     │    │                                 │
     │    ▼                                 │
     │  embed query (Gemini)                │
     │    │                                 │
     │    ▼                                 │
     │  src/retrieval.py                    │
     │  ┌─────────────┐   ┌───────────────┐ │
     │  │ dense search │   │  BM25 search  │ │
     │  │  (Chroma)    │   │ (rank_bm25)   │ │
     │  └──────┬───────┘   └───────┬───────┘ │
     │         └─────────┬─────────┘         │
     │                    ▼                   │
     │         Reciprocal Rank Fusion         │
     │                    │                   │
     │                    ▼                   │
     │       Cross-encoder reranking          │
     │        (sentence-transformers)         │
     │                    │                   │
     │                    ▼                   │
     │     top-k chunks → prompt → Gemini     │
     │                    │                   │
     │                    ▼                   │
     │        answer + cited sources          │
     └─────────────────────────────────────┘
```

---

## Retrieval pipeline in depth

Retrieval quality is the single biggest lever on answer quality in any RAG
system — this project implements four stages, each of which is toggleable
live from the sidebar so you can demo the difference each one makes.

### 1. Dense vector search
The question is embedded with Gemini's `gemini-embedding-001` model and
compared against every chunk's embedding in Chroma using cosine similarity.
Good at semantic matches ("cutoff" ≈ "minimum requirement") but weaker on
exact strings — numbers, codes, acronyms, company names.

### 2. BM25 (sparse keyword search)
A classic TF-IDF-family ranking algorithm (`rank_bm25`) run over the same
chunk corpus, tokenized and scored purely on term overlap and term rarity.
Strong exactly where dense search is weak: exact keyword and numeric matches.
Built fresh from whatever's currently in the Chroma collection, filtered to
the categories selected in the sidebar, and cached until the corpus changes.

### 3. Reciprocal Rank Fusion (RRF)
Dense similarity scores and BM25 scores live on different, incomparable
scales, so they can't just be averaged. RRF sidesteps this by fusing based on
**rank position** instead of raw score:

```
score(doc) = Σ  1 / (k + rank_in_list)
```

for every ranked list the document appears in (`k = 60`, a standard default).
A document ranked highly by either method scores well; one ranked highly by
*both* scores best.

### 4. Cross-encoder reranking
Dense and BM25 search both had to score against the *entire* corpus, which
means they use cheap, independent scoring functions (cosine similarity /
term overlap). A cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) instead
feeds the **(question, candidate-chunk) pair together** through a transformer,
letting it directly model how well they answer each other — much more
accurate, but too slow to run over a whole corpus. So it's used as the final
pass: only the ~15 candidates that survived fusion get reranked, and the true
top-k are passed to the LLM.

### Why this order
Cheap, approximate methods (vector + BM25) narrow a large corpus down to a
manageable candidate set; an expensive, accurate method (cross-encoder)
re-sorts just that small set. This "retrieve cheap, rerank precise" pattern
is standard in production search and recommendation systems.

### 5. Conversation memory (query condensation)
A follow-up like "what about the backlog policy?" has no useful embedding on
its own — the reference to "the placement process" from the previous turn
lives only in chat history, not in the question text. Before retrieval, the
last few turns are passed to Gemini with an instruction to rewrite the
follow-up into a fully standalone question (`condense_question` in
`src/app.py`); that standalone version is what actually gets embedded,
BM25-searched, and reranked. The UI shows the rewrite ("🧠 Interpreted as:
...") whenever it differs from what you typed, and recent turns are also
included in the final answer prompt so the model's tone stays consistent
across a conversation. Toggleable via "Conversation memory" in the sidebar.

---

## Project structure

```
RAG Project/
├── data/
│   ├── documents/
│   │   ├── academic/    # curriculum, syllabus, academic calendar, exam rules
│   │   ├── hostel/      # hostel rules, mess info, campus circulars
│   │   ├── placement/   # placement brochure, eligibility criteria, past data
│   │   └── custom/      # anything else
│   └── chroma_db/       # generated vector store (gitignored)
├── src/
│   ├── config.py         # paths, model names, chunk settings — single source of truth
│   ├── ingest.py         # PDF → text → chunks → embeddings → Chroma (CLI + reusable functions)
│   ├── retrieval.py      # BM25 index, RRF fusion, cross-encoder reranking
│   └── app.py            # Streamlit chat UI: upload, retrieval controls, chat, sources
├── requirements.txt
├── .env.example          # template for your API key
└── .env                  # your actual key (gitignored, never committed)
```

---

## Setup

```bash
cd "RAG Project"
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# open .env and set:
# GEMINI_API_KEY=your_actual_key
# get a free key at https://aistudio.google.com/apikey
```

`sentence-transformers` (used for reranking) pulls in `torch`, so the first
install can take a few minutes.

---

## Adding documents

**Option A — upload from the running app (easiest).** Open the sidebar's
"📤 Upload a PDF" section, pick a category, choose one or more PDFs, and click
"Index new file(s)". The file is saved to `data/documents/<category>/` and
indexed immediately — no restart needed.

**Option B — drop files manually, then index via CLI:**

```bash
# copy PDFs into the right folder, e.g.:
cp my_placement_brochure.pdf "data/documents/placement/"

# then index everything under data/documents/
python src/ingest.py
```

Re-run `ingest.py` any time you add files outside the app (it's idempotent —
`upsert` means re-running it on unchanged files is a no-op for those chunks).

---

## Running the app

```bash
source .venv/bin/activate
streamlit run src/app.py
```

Opens at `http://localhost:8501` (or the next free port). The UI includes:

- **Chat** with markdown-formatted answers and an expandable, per-answer
  "sources" panel (file name, page number, category).
- **Example question chips** shown before your first message.
- **Category stat pills** at the top showing how many chunks are indexed per
  category.
- **Sidebar controls:**
  - *Answer model* — pick any Gemini model your API key supports (fetched
    live via `genai.list_models()`), independent of the embedding model.
  - *Category filter* — restrict retrieval to specific document categories.
  - *Chunks to retrieve* — how many chunks (`top_k`) feed the final answer.
  - *Retrieval pipeline toggles* — turn hybrid search, reranking, and
    conversation memory on/off to compare answer quality and see which
    pipeline stages ran, shown as a caption under each answer.
  - *Upload a PDF* — add new documents without leaving the app.
  - *Clear chat* button.

---

## Configuration

All tunable constants live in `src/config.py`:

| Constant | Purpose | Default |
|---|---|---|
| `EMBEDDING_MODEL` | Model used to embed both documents and queries — must stay consistent, changing it invalidates the existing index | `gemini-embedding-001` |
| `GENERATION_MODEL` | Default answer model (overridable per-query in the UI) | `gemini-2.5-flash` |
| `CHUNK_SIZE` | Characters per chunk | `1000` |
| `CHUNK_OVERLAP` | Overlap between consecutive chunks, preserves context across chunk boundaries | `150` |
| `TOP_K` | Default number of chunks retrieved per query | `5` |
| `CATEGORIES` | Document categories / folder names | `academic, hostel, placement, custom` |

> **Note:** if you change `EMBEDDING_MODEL`, you must delete `data/chroma_db/`
> and re-run `ingest.py` — old embeddings and new embeddings live in different
> vector spaces and aren't comparable.

---

## How each file works

**`src/config.py`** — loads `.env`, resolves paths relative to the project
root, and centralizes every model name and tunable constant so nothing is
hardcoded elsewhere.

**`src/ingest.py`** — the indexing pipeline:
- `read_pdf_text(pdf_path)` — extracts text per page via `pypdf` (accepts a
  file path or an in-memory file-like object, so it works for both
  disk-based ingestion and in-app uploads).
- `chunk_text(text)` — fixed-size sliding-window chunking with overlap.
- `embed_batch(texts)` — batched calls to Gemini's embedding API.
- `collect_chunks(...)` / `index_pdf(...)` — reusable single-PDF indexing
  logic, imported directly by `app.py` for the upload feature.
- `build_collection()` — walks `data/documents/<category>/`, indexes every
  PDF found; run standalone via `python src/ingest.py`.

**`src/retrieval.py`** — the advanced retrieval stages:
- `BM25Index` — builds an in-memory BM25 index from whatever's currently in
  the Chroma collection (optionally category-filtered).
- `dense_search(...)` — thin wrapper around Chroma's vector query.
- `reciprocal_rank_fusion(...)` — generic RRF over any number of ranked
  lists.
- `hybrid_search(...)` — orchestrates dense + BM25 + RRF into one candidate
  list.
- `rerank(...)` — cross-encoder reranking of the fused candidates,
  lazy-loading the model on first use (`get_reranker()`).

**`src/app.py`** — the Streamlit UI: wires together config, ingestion (for
uploads), and retrieval into the chat experience described above.

---

## Talking points for interviews

- **End-to-end RAG pipeline** you built and can explain line by line, not a
  black-box library call.
- **Hybrid retrieval**: why dense search alone misses exact-match queries
  (numbers, codes, names) and how BM25 complements it.
- **Reciprocal Rank Fusion**: why you can't just average a cosine-similarity
  score and a BM25 score, and how RRF avoids that by fusing on rank instead
  of raw score.
- **Two-stage retrieve-then-rerank architecture**: the standard
  cheap-recall / expensive-precision pattern used in real search systems,
  and why a cross-encoder can't replace the first stage (too slow at corpus
  scale) but is much more accurate than embedding similarity at small scale.
- **Grounding and hallucination control**: the prompt explicitly instructs
  the model to answer only from retrieved context and cite sources, and the
  UI surfaces exactly which chunks were used.
- **Live document ingestion**: uploaded PDFs are chunked, embedded, and
  queryable within the same session — no separate build step.
- **Model flexibility**: the answer-generation model is swappable per query
  from whatever your API key supports, while the embedding model is
  deliberately fixed (and you can explain *why* — vector space consistency).

---

## Troubleshooting

**`GEMINI_API_KEY is not set`** — copy `.env.example` to `.env` and fill in a
real key from https://aistudio.google.com/apikey.

**`404 ... is not found for API version v1beta`** — the model name in
`config.py` isn't available for your key/region. Run this to see what your
key actually supports:
```python
import google.generativeai as genai
genai.configure(api_key="...")
for m in genai.list_models():
    print(m.name, m.supported_generation_methods)
```

**`np.float_ was removed in the NumPy 2.0 release`** — an older `chromadb`
version is incompatible with `sentence-transformers`' numpy 2.x dependency.
Fixed by pinning `chromadb>=1.0.0` in `requirements.txt`; if you hit this
again, `pip install --upgrade chromadb`.

**No answers / "I couldn't find anything relevant"** — check the sidebar's
chunk counter; if it's 0, no PDFs are indexed yet. Upload one via the sidebar
or run `python src/ingest.py`.

**Slow first query** — the cross-encoder reranker model downloads on first
use (~90 MB) and loads into memory; subsequent queries are fast.

---

## Ideas to extend

- Swap fixed-size chunking for a recursive or semantic splitter.
- Add a "quiz me" mode that generates interview questions from indexed notes.
- Support DOCX/HTML sources alongside PDF.
- Show BM25 / vector / rerank scores side by side in the UI as a
  retrieval-comparison teaching tool.
- Add conversation memory so follow-up questions ("what about branch-wise
  eligibility?") resolve against chat history.
- Deploy the Streamlit app publicly (Streamlit Community Cloud) and link it
  from your resume.
