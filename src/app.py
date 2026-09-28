import io

import chromadb
import google.generativeai as genai
import streamlit as st

from config import (
    CATEGORIES,
    CHROMA_DIR,
    DOCUMENTS_DIR,
    EMBEDDING_MODEL,
    GEMINI_API_KEY,
    GENERATION_MODEL,
    TOP_K,
)
from ingest import index_pdf
from retrieval import RERANKER_MODEL, BM25Index, hybrid_search, rerank

RERANKER_LABEL = RERANKER_MODEL.split("/")[-1]

CATEGORY_META = {
    "academic": {"icon": "📚", "label": "Academic"},
    "hostel": {"icon": "🏠", "label": "Hostel & Campus"},
    "placement": {"icon": "💼", "label": "Placement"},
    "custom": {"icon": "📄", "label": "Other"},
}

EXAMPLE_QUESTIONS = [
    "What is the CPI cutoff for placements?",
    "What was the highest package last year?",
    "What are the steps in the placement process?",
]

MAX_HISTORY_TURNS = 4  # user+assistant pairs kept for conversational context


def get_recent_history(messages, max_turns=MAX_HISTORY_TURNS):
    return messages[-(max_turns * 2):]


def format_history(history):
    return "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in history)


def condense_question(history, question, model_name):
    """Rewrites a follow-up question into a standalone one using recent chat history,
    so retrieval doesn't have to rely on pronouns/references it can't resolve."""
    if not history:
        return question

    prompt = f"""Given the conversation history and a follow-up question, rewrite the \
follow-up as a standalone question that includes any necessary context from the \
history (e.g. resolve "it", "that", "what about X" into explicit terms). If the \
follow-up is already standalone, return it unchanged. Only output the rewritten \
question, nothing else.

Conversation history:
{format_history(history)}

Follow-up question: {question}

Standalone question:"""

    model = genai.GenerativeModel(f"models/{model_name}")
    response = model.generate_content(prompt)
    return response.text.strip()

st.set_page_config(page_title="IIT BHU Assistant", page_icon="🎓", layout="centered")

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; max-width: 820px; }
    .app-header { display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.1rem; }
    .app-header .emoji { font-size: 2.1rem; }
    .app-header h1 { margin: 0; font-size: 1.7rem; }
    .app-subtitle { color: var(--text-color-secondary, #8a8f98); margin-bottom: 1.1rem; font-size: 0.95rem; }
    .stat-row { display: flex; gap: 0.6rem; margin-bottom: 1.4rem; flex-wrap: wrap; }
    .stat-pill {
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 999px;
        padding: 0.3rem 0.85rem;
        font-size: 0.82rem;
        display: flex; align-items: center; gap: 0.4rem;
    }
    .source-card {
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 10px;
        padding: 0.6rem 0.9rem;
        margin-bottom: 0.5rem;
        font-size: 0.87rem;
    }
    .source-card .src-title { font-weight: 600; }
    .source-card .src-meta { color: var(--text-color-secondary, #8a8f98); font-size: 0.78rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="app-header"><span class="emoji">🎓</span><h1>IIT BHU Assistant</h1></div>'
    '<div class="app-subtitle">Ask about academics, hostels, and placements — every answer is grounded '
    'in your indexed IIT BHU documents, with sources cited.</div>',
    unsafe_allow_html=True,
)

if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY is not set. Copy .env.example to .env and fill it in.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)


@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection("iit_bhu_docs")


collection = get_collection()


@st.cache_resource
def get_bm25_index(_collection, chunk_count, categories):
    """Cached on chunk_count + categories so re-running ingest.py or changing the
    category filter invalidates and rebuilds it."""
    where_filter = (
        {"category": {"$in": list(categories)}} if categories and len(categories) < len(CATEGORIES) else None
    )
    return BM25Index.from_collection(_collection, where_filter)


@st.cache_resource
def list_generation_models():
    """Models that support generateContent, excluding niche/non-text ones."""
    exclude_keywords = ("tts", "image", "embedding", "lyria", "computer-use", "robotics")
    models = []
    for m in genai.list_models():
        name = m.name.replace("models/", "")
        if "generateContent" not in m.supported_generation_methods:
            continue
        if any(kw in name for kw in exclude_keywords):
            continue
        models.append(name)
    return sorted(models)


def category_counts():
    total = collection.count()
    counts = {cat: 0 for cat in CATEGORIES}
    if total:
        all_meta = collection.get(include=["metadatas"])["metadatas"]
        for meta in all_meta:
            cat = meta.get("category")
            if cat in counts:
                counts[cat] += 1
    return counts, total


counts, total_chunks = category_counts()

stat_pills = "".join(
    f'<div class="stat-pill">{CATEGORY_META[cat]["icon"]} {CATEGORY_META[cat]["label"]}: '
    f'<b>{counts[cat]}</b></div>'
    for cat in CATEGORIES
)
st.markdown(f'<div class="stat-row">{stat_pills}</div>', unsafe_allow_html=True)

if total_chunks == 0:
    st.warning(
        "No documents indexed yet. Add PDFs to `data/documents/<category>/` and run "
        "`python src/ingest.py`, then refresh this page."
    )

with st.sidebar:
    st.header("⚙️ Settings")

    available_models = list_generation_models()
    default_model = GENERATION_MODEL.replace("models/", "")
    if default_model not in available_models:
        available_models = [default_model] + available_models
    selected_model = st.selectbox(
        "Answer model",
        available_models,
        index=available_models.index(default_model),
        help="Which Gemini model generates the answer. Retrieval/embeddings stay fixed "
        "so they keep matching your indexed documents.",
    )

    selected_categories = st.multiselect(
        "Search only these categories",
        CATEGORIES,
        default=CATEGORIES,
        format_func=lambda c: f"{CATEGORY_META[c]['icon']} {CATEGORY_META[c]['label']}",
    )
    top_k = st.slider("Chunks to retrieve", min_value=2, max_value=10, value=TOP_K)

    st.markdown("---")
    st.subheader("🔍 Retrieval pipeline")
    use_bm25 = st.checkbox(
        "Hybrid search (BM25 + vector)",
        value=True,
        help="Fuses dense vector search with BM25 keyword search using Reciprocal Rank Fusion, "
        "instead of relying on vector similarity alone.",
    )
    use_rerank = st.checkbox(
        "Cross-encoder reranking",
        value=True,
        help="Re-scores the retrieved candidates with a cross-encoder "
        f"({RERANKER_LABEL}) for a final precision pass before answering.",
    )
    use_memory = st.checkbox(
        "Conversation memory",
        value=True,
        help="Rewrites follow-up questions (e.g. 'what about branch-wise?') into "
        "standalone questions using recent chat history before retrieval, and gives "
        "the model recent turns as context.",
    )

    st.markdown("---")
    st.caption(f"**{total_chunks}** chunks indexed across {sum(1 for c in counts.values() if c > 0)} categories.")

    st.markdown("---")
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.subheader("📤 Upload a PDF")
    upload_category = st.selectbox(
        "File category",
        CATEGORIES,
        index=CATEGORIES.index("custom"),
        format_func=lambda c: f"{CATEGORY_META[c]['icon']} {CATEGORY_META[c]['label']}",
        key="upload_category",
    )
    uploaded_files = st.file_uploader(
        "Add any PDF to the knowledge base",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if "indexed_uploads" not in st.session_state:
        st.session_state.indexed_uploads = set()

    if uploaded_files:
        new_files = [
            f for f in uploaded_files
            if (f.name, f.size) not in st.session_state.indexed_uploads
        ]
        if new_files and st.button(f"➕ Index {len(new_files)} new file(s)"):
            progress = st.progress(0.0, text="Starting...")
            for i, uploaded_file in enumerate(new_files):
                progress.progress(
                    i / len(new_files), text=f"Indexing {uploaded_file.name}..."
                )
                pdf_bytes = uploaded_file.getvalue()

                dest_dir = DOCUMENTS_DIR / upload_category
                dest_dir.mkdir(parents=True, exist_ok=True)
                (dest_dir / uploaded_file.name).write_bytes(pdf_bytes)

                index_pdf(
                    collection,
                    io.BytesIO(pdf_bytes),
                    upload_category,
                    uploaded_file.name,
                )
                st.session_state.indexed_uploads.add((uploaded_file.name, uploaded_file.size))
            progress.progress(1.0, text="Done!")
            st.success(f"Indexed {len(new_files)} file(s). Refreshing...")
            st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.markdown("**Try asking:**")
    cols = st.columns(len(EXAMPLE_QUESTIONS))
    for col, example in zip(cols, EXAMPLE_QUESTIONS):
        if col.button(example, use_container_width=True):
            st.session_state.pending_question = example
            st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander(f"📎 {len(message['sources'])} source(s)"):
                for meta in message["sources"]:
                    icon = CATEGORY_META.get(meta["category"], {}).get("icon", "📄")
                    st.markdown(
                        f'<div class="source-card">'
                        f'<div class="src-title">{icon} {meta["source"]}</div>'
                        f'<div class="src-meta">Page {meta["page"]} · {meta["category"]}</div>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )

question = st.chat_input("Ask something about IIT BHU...")
if not question and st.session_state.get("pending_question"):
    question = st.session_state.pop("pending_question")

if question:
    prior_messages = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        history = get_recent_history(prior_messages) if use_memory else []

        search_question = question
        if history:
            with st.spinner("Understanding follow-up..."):
                search_question = condense_question(history, question, selected_model)
            if search_question.strip().lower() != question.strip().lower():
                st.caption(f"🧠 Interpreted as: _{search_question}_")

        with st.spinner("Searching documents..."):
            query_embedding = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=search_question,
                task_type="retrieval_query",
            )["embedding"]

            where_filter = (
                {"category": {"$in": selected_categories}}
                if selected_categories and len(selected_categories) < len(CATEGORIES)
                else None
            )

            bm25_index = get_bm25_index(collection, total_chunks, tuple(sorted(selected_categories)))

            candidates = hybrid_search(
                collection,
                bm25_index,
                search_question,
                query_embedding,
                top_k,
                where_filter=where_filter,
                use_bm25=use_bm25,
            )

        if use_rerank and candidates:
            with st.spinner(f"Reranking with {RERANKER_LABEL}..."):
                candidates = rerank(search_question, candidates, top_k)
        else:
            candidates = candidates[:top_k]

        documents = [hit["document"] for hit in candidates]
        metadatas = [hit["metadata"] for hit in candidates]

        if not documents:
            answer = (
                "I couldn't find anything relevant in the indexed documents. "
                "Try adding more PDFs under `data/documents/` and re-running `ingest.py`."
            )
            st.markdown(answer)
            sources = []
        else:
            context_blocks = [
                f"[Source: {meta['source']}, page {meta['page']}]\n{doc}"
                for doc, meta in zip(documents, metadatas)
            ]
            context = "\n\n---\n\n".join(context_blocks)

            history_block = ""
            if history:
                history_block = (
                    "\n\nRecent conversation (for context only — answer using the "
                    "Context section below, not facts recalled only from here):\n"
                    f"{format_history(history)}\n"
                )

            prompt = f"""You are an assistant answering questions about IIT BHU using only the context below.
If the answer isn't in the context, say you don't have that information.
Cite the source file and page number for any fact you use.
{history_block}
Context:
{context}

Question: {question}

Answer:"""

            with st.spinner("Thinking..."):
                model = genai.GenerativeModel(f"models/{selected_model}")
                response = model.generate_content(prompt)
                answer = response.text

            st.markdown(answer)
            pipeline_stages = []
            if history:
                pipeline_stages.append("Condensed question")
            pipeline_stages.append("Vector search")
            if use_bm25:
                pipeline_stages.append("BM25 + RRF fusion")
            if use_rerank:
                pipeline_stages.append(f"Reranked ({RERANKER_LABEL})")
            st.caption("Pipeline: " + " → ".join(pipeline_stages))
            sources = metadatas
            with st.expander(f"📎 {len(sources)} source(s)"):
                for meta in sources:
                    icon = CATEGORY_META.get(meta["category"], {}).get("icon", "📄")
                    st.markdown(
                        f'<div class="source-card">'
                        f'<div class="src-title">{icon} {meta["source"]}</div>'
                        f'<div class="src-meta">Page {meta["page"]} · {meta["category"]}</div>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
