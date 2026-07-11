"""Topic Deep Dive — details on each discovered topic."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.loaders import load_reviews
from utils.config import SENTIMENT_COLORS, SENTIMENT_EMOJI

st.set_page_config(page_title="Topic Deep Dive", page_icon="📚", layout="wide")

st.title("📚 Topic Deep Dive")
st.markdown(
    "Explore each of the 6 topics BERTopic discovered from the review corpus. "
    "See sentiment breakdown, and representative reviews per topic."
)

st.divider()

df = load_reviews()
labeled = df[df['topic_label'].notna()].copy()

# ─── Topic selector ─────────────────────────────────────────────────────
topics = sorted(labeled['topic_label'].unique())

col1, col2 = st.columns([2, 3])

with col1:
    selected_topic = st.selectbox("Select a topic to explore", topics)

with col2:
    topic_df = labeled[labeled['topic_label'] == selected_topic]
    st.markdown(f"### {selected_topic}")
    st.caption(f"{len(topic_df):,} reviews · "
               f"{(topic_df['sentiment']=='positive').mean():.0%} positive · "
               f"{(topic_df['sentiment']=='negative').mean():.0%} negative")

st.divider()

# ─── Topic metrics ──────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Reviews", f"{len(topic_df):,}")
c2.metric("Avg Rating", f"{topic_df['stars'].mean():.2f} ⭐")
c3.metric("Positive %", f"{(topic_df['sentiment']=='positive').mean():.0%}")
c4.metric("Negative %", f"{(topic_df['sentiment']=='negative').mean():.0%}")

st.divider()

# ─── Sentiment distribution + Trend over time ───────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Sentiment Breakdown")
    sentiment_counts = topic_df['sentiment'].value_counts()
    fig = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        color=sentiment_counts.index,
        color_discrete_map=SENTIMENT_COLORS,
        hole=0.4
    )
    fig.update_layout(margin=dict(t=20, b=20), height=350)
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("Volume Over Time")
    yearly = topic_df.groupby('year').size().reset_index(name='count')
    fig = px.bar(
        yearly,
        x='year',
        y='count',
        color_discrete_sequence=['#185FA5']
    )
    fig.update_layout(
        margin=dict(t=20, b=20),
        height=350,
        xaxis_title="Year",
        yaxis_title="Number of reviews"
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ─── Representative reviews ─────────────────────────────────────────────
st.subheader("Representative Reviews")
st.markdown(
    "Sampled reviews from this topic. Toggle between highest and lowest rated "
    "to see what customers love and hate about this category."
)

view = st.radio(
    "Show:",
    ["Highest rated", "Lowest rated", "Random sample"],
    horizontal=True
)

if view == "Highest rated":
    sample = topic_df.nlargest(5, 'stars')
elif view == "Lowest rated":
    sample = topic_df.nsmallest(5, 'stars')
else:
    sample = topic_df.sample(min(5, len(topic_df)), random_state=42)

for _, r in sample.iterrows():
    with st.expander(
        f"{SENTIMENT_EMOJI[r['sentiment']]} "
        f"{'⭐' * int(r['stars'])} | "
        f"{r['date'].strftime('%B %Y')}"
    ):
        st.write(r['text'])

st.divider()

# ─── Comparison across all topics ───────────────────────────────────────
st.subheader("How this topic compares to others")

st.markdown("Negative review rate by topic (lower is better for the business):")

topic_neg_rate = labeled.groupby('topic_label').apply(
    lambda x: (x['sentiment'] == 'negative').mean(),
    include_groups=False
).sort_values(ascending=True).reset_index()
topic_neg_rate.columns = ['topic', 'neg_rate']

# Highlight the selected topic
colors = [
    '#E24B4A' if t == selected_topic else '#888780'
    for t in topic_neg_rate['topic']
]

fig = go.Figure()
fig.add_trace(go.Bar(
    x=topic_neg_rate['neg_rate'],
    y=topic_neg_rate['topic'],
    orientation='h',
    marker_color=colors,
    text=[f"{v:.0%}" for v in topic_neg_rate['neg_rate']],
    textposition='outside'
))
fig.update_layout(
    height=300,
    margin=dict(t=20, b=20),
    xaxis_title="Negative review rate",
    xaxis_tickformat='.0%',
    showlegend=False
)
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "💡 The selected topic is highlighted in red. Topics with higher negative "
    "rates are opportunities for business improvement."
)