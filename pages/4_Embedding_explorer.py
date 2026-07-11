"""Embedding Explorer — 2D t-SNE visualization of the review corpus."""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.loaders import (
    load_reviews,
    load_embeddings,
    load_embedding_model,
    load_faiss_index,
    compute_2d_projection
)
from utils.search import semantic_search
from utils.config import SENTIMENT_COLORS, SENTIMENT_EMOJI

st.set_page_config(page_title="Embedding Explorer", page_icon="🗺️", layout="wide")

st.title("🗺️ Embedding Explorer")
st.markdown(
    "Review embeddings projected to 2D using **t-SNE**. "
    "Reviews with similar meaning cluster together in this space — a visual "
    "diagnostic that the embeddings capture real semantic structure."
)

st.divider()

# ─── Load data ──────────────────────────────────────────────────────────
df = load_reviews()

with st.spinner("Computing 2D projection (cached after first run, ~30-60 sec)..."):
    projection, sample_indices = compute_2d_projection()

# Only work with sampled reviews (matches projection)
df_plot = df.iloc[sample_indices].copy().reset_index(drop=True)
df_plot['tsne_x'] = projection[:, 0]
df_plot['tsne_y'] = projection[:, 1]

# ─── Controls ───────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    color_by = st.selectbox(
        "Color points by",
        options=["sentiment", "topic_label", "stars"],
        format_func=lambda x: {
            "sentiment": "Sentiment",
            "topic_label": "Topic",
            "stars": "Star Rating"
        }[x]
    )

with col2:
    filter_topic = st.selectbox(
        "Filter to topic (optional)",
        options=["All"] + sorted(df['topic_label'].dropna().unique().tolist())
    )

with col3:
    st.metric("Points shown", f"{len(df_plot):,}")

# Apply filter
df_display = df_plot.copy()
if filter_topic != "All":
    df_display = df_display[df_display['topic_label'] == filter_topic]

# ─── Main scatter plot ──────────────────────────────────────────────────
st.markdown(
    f"**Displaying {len(df_display):,} of 20,000 reviews** (random sample for performance)"
)

df_display = df_display.copy()
df_display['hover_text'] = df_display['text'].str[:150] + "..."

if color_by == "sentiment":
    fig = px.scatter(
        df_display,
        x='tsne_x', y='tsne_y',
        color='sentiment',
        color_discrete_map=SENTIMENT_COLORS,
        hover_data={
            'tsne_x': False, 'tsne_y': False,
            'stars': True, 'topic_label': True,
            'sentiment': True, 'hover_text': True
        },
        labels={'hover_text': 'Review'},
        opacity=0.6
    )
elif color_by == "topic_label":
    fig = px.scatter(
        df_display,
        x='tsne_x', y='tsne_y',
        color='topic_label',
        color_discrete_sequence=px.colors.qualitative.Set2,
        hover_data={
            'tsne_x': False, 'tsne_y': False,
            'stars': True, 'topic_label': True,
            'sentiment': True, 'hover_text': True
        },
        labels={'hover_text': 'Review'},
        opacity=0.6
    )
else:  # stars
    fig = px.scatter(
        df_display,
        x='tsne_x', y='tsne_y',
        color='stars',
        color_continuous_scale='RdYlGn',
        hover_data={
            'tsne_x': False, 'tsne_y': False,
            'stars': True, 'topic_label': True,
            'sentiment': True, 'hover_text': True
        },
        labels={'hover_text': 'Review'},
        opacity=0.6
    )

fig.update_traces(marker=dict(size=5))
fig.update_layout(
    height=600,
    margin=dict(t=20, b=20, l=20, r=20),
    xaxis_title="t-SNE dimension 1",
    yaxis_title="t-SNE dimension 2",
    legend_title=color_by.replace("_label", "").title()
)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    "💡 Hover over any point to see the review. Distinct clusters emerge naturally "
    "based on what customers are talking about — this is your corpus organized by meaning."
)

st.divider()

# ─── Explore nearest neighbors of a query ───────────────────────────────
st.subheader("Explore neighbors of any query")
st.markdown(
    "Type a query below. FAISS finds the 5 nearest reviews via semantic search, "
    "and they get highlighted on the t-SNE map."
)

query = st.text_input(
    "Query",
    placeholder="e.g. cold pizza and slow delivery",
    key="tsne_query"
)

if query:
    embedding_model = load_embedding_model()
    index = load_faiss_index()
    results = semantic_search(query, embedding_model, index, df, top_k=5)

    # Find retrieved reviews in the projected sample
    retrieved_x, retrieved_y, retrieved_text = [], [], []
    for r in results:
        matches = df_plot[df_plot['text'] == r['text']]
        if len(matches) > 0:
            row = matches.iloc[0]
            retrieved_x.append(row['tsne_x'])
            retrieved_y.append(row['tsne_y'])
            retrieved_text.append(r['text'][:150] + "...")

    fig2 = go.Figure()

    # All points in gray
    fig2.add_trace(go.Scatter(
        x=df_plot['tsne_x'],
        y=df_plot['tsne_y'],
        mode='markers',
        marker=dict(size=3, color='rgba(150,150,150,0.2)'),
        name='All reviews',
        hoverinfo='skip'
    ))

    # Retrieved points in red (only those found in the sample)
    if retrieved_x:
        fig2.add_trace(go.Scatter(
            x=retrieved_x,
            y=retrieved_y,
            mode='markers',
            marker=dict(size=14, color='#E24B4A',
                       line=dict(width=2, color='white')),
            name=f'Top {len(retrieved_x)} matches (in sample)',
            text=retrieved_text,
            hovertemplate='<b>%{text}</b><extra></extra>'
        ))

    fig2.update_layout(
        height=500,
        margin=dict(t=20, b=20, l=20, r=20),
        xaxis_title="t-SNE dimension 1",
        yaxis_title="t-SNE dimension 2",
        showlegend=True
    )

    st.plotly_chart(fig2, use_container_width=True)

    if not retrieved_x:
        st.info(
            "Retrieved reviews were not in the 5k visualization sample, "
            "but the full search results are below."
        )

    st.markdown("**Top 5 matches**")
    for r in results:
        with st.expander(
            f"{SENTIMENT_EMOJI[r['sentiment']]} "
            f"{'⭐' * int(r['stars'])} | "
            f"{r['topic']} | "
            f"Similarity: {r['score']:.2f}"
        ):
            st.write(r['text'])

st.divider()

# ─── What t-SNE is doing ────────────────────────────────────────────────
st.subheader("What is t-SNE doing here?")

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        "Every review is a 384-dimensional vector produced by the sentence "
        "transformer. **t-SNE compresses these 384 dimensions down to 2** while "
        "preserving local structure — reviews about similar topics stay close "
        "together, reviews about different topics move apart."
    )

with col2:
    st.markdown(
        "This isn't a modeling result. It's a **visual diagnostic** — it lets "
        "you see whether the embeddings are actually capturing meaningful "
        "structure. If they were random, you would see a uniform blob. Instead, "
        "distinct clusters emerge that align with topics and sentiment."
    )