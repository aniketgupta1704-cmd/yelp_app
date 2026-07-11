"""Yelp Review Intelligence — Landing page."""

import streamlit as st
from utils.config import DATA_STATS

st.set_page_config(
    page_title="Yelp Review Intelligence",
    page_icon="⭐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Header ─────────────────────────────────────────────────────────────
st.title("⭐ Yelp Review Intelligence")
st.markdown(
    "A production-style **NLP + RAG** system for extracting insights "
    "from customer reviews at scale."
)

st.markdown("`NLP` `RAG` `FAISS` `DistilBERT` `BERTopic`")

st.divider()

# ─── What it does ───────────────────────────────────────────────────────
st.subheader("What this app does")

col1, col2, col3 = st.columns(3)

with col1:
    with st.container(border=True):
        st.markdown("### 🎯 Sentiment Classification")
        st.markdown(
            "Fine-tuned DistilBERT achieves **0.783 F1**, "
            "a 37% improvement over zero-shot BART baseline."
        )

with col2:
    with st.container(border=True):
        st.markdown("### 🔍 Semantic Search + RAG")
        st.markdown(
            "Sub-15ms FAISS retrieval plus LLM synthesis. Ask questions "
            "in natural language, get structured answers."
        )

with col3:
    with st.container(border=True):
        st.markdown("### 📊 Unsupervised Topics")
        st.markdown(
            "BERTopic discovered **6 meaningful topics** without any "
            "labeling — from car washes to salon services."
        )

st.divider()

# ─── Data snapshot ──────────────────────────────────────────────────────
st.subheader("Data snapshot")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Raw Yelp Reviews", f"{DATA_STATS['total_yelp_reviews']:,}")
c2.metric("Balanced Sample", f"{DATA_STATS['balanced_sample']:,}")
c3.metric("Embedded & Indexed", f"{DATA_STATS['embedded_sample']:,}")
c4.metric("Topics Discovered", DATA_STATS['topics_discovered'])

st.caption(
    f"Reviews span {DATA_STATS['date_range']} across restaurants, salons, "
    "gyms, pet services, car washes, and more."
)

st.divider()

# ─── Performance highlights ─────────────────────────────────────────────
st.subheader("Performance highlights")

p1, p2, p3, p4 = st.columns(4)
p1.metric("Sentiment F1", "0.783", "+0.210 vs zero-shot")
p2.metric("RAG P@5", "0.88")
p3.metric("RAG MRR", "0.95")
p4.metric("Search Latency", "~10 ms")

st.divider()

# ─── Navigation ─────────────────────────────────────────────────────────
st.subheader("Explore the app")

nav_items = [
    ("📊 Overview Dashboard", "Sentiment and topic distribution across all 20k reviews"),
    ("🔧 How It Works", "Architecture, models used, and evaluation metrics"),
    ("🔍 Semantic Search", "Natural language search with keyword vs semantic comparison"),
    ("🗺️ Embedding Explorer", "2D visualization of all 20k reviews via UMAP"),
    ("📚 Topic Deep Dive", "Details on each discovered topic with sample reviews"),
    ("✍️ Analyze Your Review", "Paste any text and see sentiment + similar reviews"),
]

col_a, col_b = st.columns(2)
for i, (title, desc) in enumerate(nav_items):
    target = col_a if i % 2 == 0 else col_b
    with target:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(desc)

st.divider()
st.caption(
    "Built with Sentence Transformers · DistilBERT · BERTopic · FAISS · "
    "HuggingFace Inference · Streamlit"
)