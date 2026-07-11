"""Semantic Search — RAG-powered with keyword comparison and latency breakdown."""

import streamlit as st
import time
from utils.loaders import (
    load_reviews,
    load_embedding_model,
    load_faiss_index,
    load_hf_client
)
from utils.search import semantic_search_with_timing, keyword_search
from utils.rag import generate_rag_answer
from utils.config import SENTIMENT_COLORS, SENTIMENT_EMOJI

st.set_page_config(page_title="Semantic Search", page_icon="🔍", layout="wide")

st.title("🔍 Semantic Search")
st.markdown(
    "Ask questions in natural language. FAISS retrieves the most relevant reviews, "
    "then an LLM synthesizes an answer with a business recommendation."
)

st.divider()

# ─── Load resources ─────────────────────────────────────────────────────
df = load_reviews()
embedding_model = load_embedding_model()
index = load_faiss_index()

# ─── Query input ────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([3, 1, 1])

with col1:
    query = st.text_input(
        "What are you looking for?",
        placeholder="e.g. cold food and slow service"
    )

with col2:
    sentiment_filter = st.selectbox(
        "Filter by sentiment",
        ["All", "positive", "negative", "neutral"]
    )

with col3:
    topic_filter = st.selectbox(
        "Filter by topic",
        ["All"] + sorted(df['topic_label'].dropna().unique().tolist())
    )

col_a, col_b = st.columns([2, 1])
with col_a:
    top_k = st.slider("Number of results", min_value=3, max_value=10, value=5)
with col_b:
    show_comparison = st.checkbox("Compare with keyword search", value=False)

if query:
    sf = sentiment_filter if sentiment_filter != "All" else None
    tf = topic_filter if topic_filter != "All" else None

    # ─── Run semantic search with timing ─────────────────────────────────
    with st.spinner("Retrieving..."):
        results, timings = semantic_search_with_timing(
            query, embedding_model, index, df,
            top_k=top_k, sentiment_filter=sf, topic_filter=tf
        )

    if not results:
        st.warning("No results found. Try adjusting filters.")
        st.stop()

    # ─── RAG Summary ─────────────────────────────────────────────────────
    st.markdown("### 🧠 AI Summary")
    with st.spinner("Generating synthesis..."):
        try:
            client = load_hf_client()
            rag_answer, model_used, llm_latency = generate_rag_answer(
                query, results, client
            )
            timings['llm_ms'] = llm_latency

            st.success(rag_answer)
            st.caption(f"Generated using: `{model_used}`")
        except Exception as e:
            st.error(f"Summary unavailable: {e}")
            import traceback
            with st.expander("Full traceback"):
                st.code(traceback.format_exc())
            timings['llm_ms'] = 0

    # ─── Latency breakdown ───────────────────────────────────────────────
    st.markdown("### ⚡ Latency Breakdown")

    l1, l2, l3, l4 = st.columns(4)
    l1.metric("Embedding", f"{timings['embedding_ms']:.0f} ms")
    l2.metric("FAISS Search", f"{timings['faiss_ms']:.1f} ms")
    l3.metric("LLM Generation",
              f"{timings['llm_ms']:.0f} ms" if timings['llm_ms'] else "N/A")
    total = timings['embedding_ms'] + timings['faiss_ms'] + timings['llm_ms']
    l4.metric("Total", f"{total:.0f} ms")

    st.caption(
        "💡 Retrieval alone (embedding + FAISS) stays under 100ms. "
        "LLM adds latency but delivers structured insight."
    )

    st.divider()

    # ─── Semantic vs Keyword Comparison ──────────────────────────────────
    if show_comparison:
        st.markdown("### 🔀 Semantic vs Keyword Search")
        st.markdown(
            "Keyword search only matches literal words. Semantic search "
            "understands meaning — often finding relevant reviews that share "
            "no keywords with the query."
        )

        comp_col1, comp_col2 = st.columns(2)

        with comp_col1:
            st.markdown("**🧠 Semantic Search (FAISS + embeddings)**")
            for r in results[:5]:
                with st.expander(
                    f"{SENTIMENT_EMOJI[r['sentiment']]} "
                    f"{'⭐' * int(r['stars'])} | "
                    f"Score: {r['score']:.2f}"
                ):
                    st.write(r['text'][:400] + "...")

        with comp_col2:
            st.markdown("**🔤 Keyword Search (literal word matching)**")
            kw_results = keyword_search(query, df, top_k=5)
            if kw_results:
                for r in kw_results:
                    with st.expander(
                        f"{SENTIMENT_EMOJI[r['sentiment']]} "
                        f"{'⭐' * int(r['stars'])} | "
                        f"Match: {r['score']:.2f}"
                    ):
                        st.write(r['text'][:400] + "...")
            else:
                st.info(
                    "No reviews contain any of these exact words. "
                    "Semantic search still found relevant results."
                )

        st.divider()

    # ─── Source Reviews ──────────────────────────────────────────────────
    st.markdown(f"### 📋 Source Reviews ({len(results)} retrieved)")

    for r in results:
        with st.expander(
            f"{SENTIMENT_EMOJI[r['sentiment']]} "
            f"{'⭐' * int(r['stars'])} | "
            f"{r['topic']} | "
            f"Cosine similarity: {r['score']:.3f}"
        ):
            st.write(r['text'])
            st.caption(f"Date: {r['date'].strftime('%B %Y')}")