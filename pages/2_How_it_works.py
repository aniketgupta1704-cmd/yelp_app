"""How It Works — architecture, models, and evaluation metrics."""

import streamlit as st
import plotly.graph_objects as go
from utils.config import MODEL_INFO, EVAL_METRICS, DATA_STATS

st.set_page_config(page_title="How It Works", page_icon="🔧", layout="wide")

st.title("🔧 How It Works")
st.markdown(
    "The architecture, models, and evaluation behind this app. "
    "Everything you see in other pages is powered by the pipeline below."
)

st.divider()

st.subheader("🏗️ System Architecture")
st.caption(
    "End-to-end NLP pipeline powering sentiment analysis, topic discovery, "
    "semantic retrieval, and Retrieval-Augmented Generation (RAG)."
)

import plotly.graph_objects as go

fig = go.Figure()

bg = "#0E1117"
line = "#6EA8FE"

fig.update_layout(
    width=1350,
    height=980,
    paper_bgcolor=bg,
    plot_bgcolor=bg,
    margin=dict(l=10, r=10, t=20, b=10),
    xaxis=dict(visible=False, range=[0,100]),
    yaxis=dict(visible=False, range=[0,100]),
)

# -------------------------------------------------
# Helper
# -------------------------------------------------

def node(x, y, title, body, color="#1B1F2A"):

    fig.add_shape(
        type="rect",
        x0=x-8.8,
        x1=x+8.8,
        y0=y-4.3,
        y1=y+4.3,
        fillcolor=color,
        line=dict(color="#525252", width=1.5),
        layer="below"
    )

    fig.add_annotation(
        x=x,
        y=y+1.35,
        text=f"<b>{title}</b>",
        showarrow=False,
        font=dict(size=15, color="white")
    )

    fig.add_annotation(
        x=x,
        y=y-1.45,
        text=body,
        showarrow=False,
        font=dict(size=11, color="#B8B8B8")
    )


def arrow(x1,y1,x2,y2):

    fig.add_annotation(
        x=x2,
        y=y2,
        ax=x1,
        ay=y1,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowwidth=1.6,
        arrowhead=2,
        arrowsize=1.1,
        arrowcolor=line,
    )

# -------------------------------------------------
# Main
# -------------------------------------------------

node(
    50,94,
    "📥 Raw Yelp Reviews",
    "200k reviews • Yelp Open Dataset"
)

arrow(50,89.5,50,84)

node(
    50,79,
    "🧹 Data Preprocessing",
    "Clean • Balance • Sample 20k"
)

# split

arrow(50,75,25,66)
arrow(50,75,70,66)

# -------------------------------------------------
# Left
# -------------------------------------------------

fig.add_annotation(
    x=25,
    y=70,
    text="<b>Sentiment Classifier</b>",
    showarrow=False,
    font=dict(size=18,color="#FFB347")
)

node(
    25,62,
    "🎯 Fine-tuned DistilBERT",
    "10k labeled reviews • Weighted F1 = 0.783",
    "#202733"
)

arrow(25,58,25,48)

node(
    25,44,
    "✅ Sentiment Prediction",
    "Positive • Neutral • Negative",
    "#193324"
)

# -------------------------------------------------
# Right
# -------------------------------------------------

fig.add_annotation(
    x=70,
    y=70,
    text="<b>Embedding + Retrieval</b>",
    showarrow=False,
    font=dict(size=18,color="#90CAF9")
)

node(
    70,62,
    "🧠 SentenceTransformer",
    "all-MiniLM-L6-v2",
    "#202733"
)

arrow(70,58,70,49)

node(
    70,45,
    "🔢 384-d Embeddings",
    "20k semantic vectors",
    "#1D3527"
)

arrow(70,40.8,56,29)
arrow(70,40.8,84,29)

node(
    56,24,
    "📚 BERTopic",
    "UMAP • HDBSCAN\nc-TF-IDF • 6 Topics",
    "#32243A"
)

node(
    84,24,
    "🗂️ FAISS Index",
    "IndexFlatIP\n20k vectors",
    "#2C2740"
)

arrow(84,20,84,10)

node(
    84,6,
    "🔍 Semantic Search + RAG",
    "Cosine Search • Top-K • HF LLM Summary",
    "#1D3527"
)

# -------------------------------------------------

st.plotly_chart(
    fig,
    use_container_width=True,
    config={
        "displaylogo": False,
        "modeBarButtonsToRemove": [
            "lasso2d",
            "select2d",
            "autoScale2d"
        ]
    }
)

st.divider()
# ─── Models Used ────────────────────────────────────────────────────────
st.subheader("Models Used")
st.markdown("Every component and why it was chosen.")

model_cards = [
    ("Sentence Transformer", MODEL_INFO['embedding']['name'],
     f"{MODEL_INFO['embedding']['dimensions']}-dim embeddings",
     "Fast, strong quality, standard for semantic search"),
    ("Sentiment Classifier", MODEL_INFO['sentiment']['name'],
     f"{MODEL_INFO['sentiment']['params']} parameters",
     "Fine-tuned on domain data, beats zero-shot BART by +0.21 F1"),
    ("Topic Model", MODEL_INFO['topic']['name'],
     MODEL_INFO['topic']['components'],
     "Unsupervised topic discovery without any labels"),
    ("Vector Index", MODEL_INFO['search']['name'],
     f"{MODEL_INFO['search']['vectors']} vectors indexed",
     "Sub-15ms retrieval latency for real-time search"),
    ("RAG Generator", MODEL_INFO['rag']['name'],
     f"Fallback: {MODEL_INFO['rag']['fallback']}",
     "Tiered fallback ensures the app never breaks"),
]

for i in range(0, len(model_cards), 2):
    cols = st.columns(2)
    for j, col in enumerate(cols):
        if i + j < len(model_cards):
            name, model, spec, rationale = model_cards[i + j]
            with col:
                with st.container(border=True):
                    st.markdown(f"**{name}**")
                    st.markdown(f"`{model}`")
                    st.caption(f"📐 {spec}")
                    st.caption(f"💡 {rationale}")

st.divider()

# ─── Sentiment Evaluation ───────────────────────────────────────────────
st.subheader("Sentiment Classifier Evaluation")
st.markdown("Zero-shot vs fine-tuned comparison on held-out test set.")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("**Head-to-head F1 comparison**")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=['Zero-shot BART', 'Fine-tuned DistilBERT'],
        y=[EVAL_METRICS['sentiment']['zero_shot_bart_f1'],
           EVAL_METRICS['sentiment']['finetuned_distilbert_f1']],
        marker_color=['#888780', '#1D9E75'],
        text=[f"{EVAL_METRICS['sentiment']['zero_shot_bart_f1']:.3f}",
              f"{EVAL_METRICS['sentiment']['finetuned_distilbert_f1']:.3f}"],
        textposition='outside'
    ))
    fig.update_layout(
        yaxis_title="Weighted F1",
        yaxis_range=[0, 1],
        showlegend=False,
        height=300,
        margin=dict(t=20, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("**Interpretation**")
    st.markdown(
        f"- Zero-shot achieves **{EVAL_METRICS['sentiment']['zero_shot_bart_f1']:.3f}** "
        "without any training data\n"
        f"- Fine-tuning on 10k Yelp reviews improves F1 by "
        f"**+{EVAL_METRICS['sentiment']['improvement']:.3f}**\n"
        "- Biggest gains on **neutral** class (hardest to classify)\n"
        "- Confirms fine-tuning on domain data is worth the compute"
    )

st.divider()

# ─── RAG Evaluation ─────────────────────────────────────────────────────
st.subheader("RAG Retrieval Evaluation")
st.markdown(
    f"Evaluated across **{EVAL_METRICS['rag']['queries_tested']} test queries** "
    "spanning service, food, cleanliness, location, and complaint handling."
)

r1, r2, r3, r4 = st.columns(4)
r1.metric("Precision@1", f"{EVAL_METRICS['rag']['precision_at_1']:.3f}")
r2.metric("Precision@5", f"{EVAL_METRICS['rag']['precision_at_5']:.3f}")
r3.metric("Mean Reciprocal Rank", f"{EVAL_METRICS['rag']['mean_reciprocal_rank']:.3f}")
r4.metric("Avg Latency", f"{EVAL_METRICS['rag']['avg_latency_ms']} ms")

st.markdown("**What these metrics mean**")
st.markdown(
    "- **Precision@K** — of the top K results returned, how many were relevant\n"
    "- **MRR** — how high does the first relevant result appear "
    "(1.0 = always first)\n"
    "- Industry benchmark for production search systems is P@5 ≥ 0.7 and MRR ≥ 0.8"
)

st.divider()

# ─── Known Limitations ──────────────────────────────────────────────────
st.subheader("Known Limitations")
st.markdown("Honest failure modes I documented during evaluation.")

with st.container(border=True):
    st.markdown("**Abstract queries perform worse than specific ones**")
    st.markdown(
        "Queries like *\"friendly atmosphere\"* score lower than *\"long wait times\"* "
        "because embedding models pick up keyword overlap on 'friendly' but miss "
        "semantically related concepts like 'cozy' or 'welcoming feel'."
    )
    st.caption("💡 Fix: LLM-based query expansion before search — planned improvement")

with st.container(border=True):
    st.markdown("**6 topics is thin for 20k reviews**")
    st.markdown(
        "BERTopic clustered aggressively on this sample. A larger corpus "
        "(50k+) or tighter HDBSCAN parameters would surface more granular topics "
        "like *pizza vs sushi* rather than one *General Dining* bucket."
    )
    st.caption("💡 Fix: increase min_topic_size threshold or use larger corpus")

with st.container(border=True):
    st.markdown("**Neutral class remains hard**")
    st.markdown(
        "3-star reviews are genuinely ambiguous — mixed positive/negative signals. "
        "Even the fine-tuned model has the lowest F1 on this class."
    )
    st.caption("💡 Fix: binary positive/negative framing may serve business better")

st.divider()

# ─── Tech Stack Footer ──────────────────────────────────────────────────
st.subheader("Tech Stack")

tech = [
    "Python", "PyTorch", "Sentence Transformers", "Hugging Face Transformers",
    "BERTopic", "FAISS", "scikit-learn", "Streamlit", "Plotly",
    "HuggingFace Inference API"
]

st.markdown(" ".join([f"`{t}`" for t in tech]))