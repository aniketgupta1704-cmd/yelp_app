"""Overview Dashboard — sentiment and topic distributions."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from utils.loaders import load_reviews
from utils.config import SENTIMENT_COLORS

st.set_page_config(page_title="Overview Dashboard", page_icon="📊", layout="wide")

df = load_reviews()

st.title("Review Overview Dashboard")
st.markdown("Insights from 20,000 Yelp reviews across multiple business categories.")

# ─── KPI cards ──────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Reviews", f"{len(df):,}")
col2.metric("Positive", f"{(df['sentiment']=='positive').mean():.1%}")
col3.metric("Negative", f"{(df['sentiment']=='negative').mean():.1%}")
col4.metric("Avg Rating", f"{df['stars'].mean():.2f} stars")

# Business insight KPIs
st.markdown("### Business Insights")
labeled = df[df['topic_label'].notna()]

# Most complained topic (highest % negative)
topic_neg = labeled.groupby('topic_label').apply(
    lambda x: (x['sentiment'] == 'negative').mean()
).sort_values(ascending=False)

# Most positive topic
topic_pos = labeled.groupby('topic_label').apply(
    lambda x: (x['sentiment'] == 'positive').mean()
).sort_values(ascending=False)

col1, col2, col3 = st.columns(3)
col1.metric("Most Complained Topic", topic_neg.index[0],
           f"{topic_neg.iloc[0]:.0%} negative")
col2.metric("Most Praised Topic", topic_pos.index[0],
           f"{topic_pos.iloc[0]:.0%} positive")
col3.metric("Largest Topic",
           labeled['topic_label'].value_counts().index[0],
           f"{labeled['topic_label'].value_counts().iloc[0]:,} reviews")

st.markdown("---")

# ─── Sentiment + Topic distribution ─────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Sentiment Distribution")
    sentiment_counts = df['sentiment'].value_counts()
    fig = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        color=sentiment_counts.index,
        color_discrete_map=SENTIMENT_COLORS,
        hole=0.4
    )
    fig.update_layout(margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("Reviews by Topic")
    topic_counts = labeled['topic_label'].value_counts()
    fig = px.bar(
        x=topic_counts.values,
        y=topic_counts.index,
        orientation='h',
        color=topic_counts.index,
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_layout(showlegend=False, margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

# ─── Sentiment by Topic ─────────────────────────────────────────────────
st.subheader("Sentiment by Topic")
topic_sentiment = labeled.groupby(['topic_label', 'sentiment']).size().unstack(fill_value=0)
topic_sentiment_pct = topic_sentiment.div(topic_sentiment.sum(axis=1), axis=0).reset_index()

fig = go.Figure()
for sentiment, color in SENTIMENT_COLORS.items():
    if sentiment in topic_sentiment_pct.columns:
        fig.add_trace(go.Bar(
            name=sentiment,
            x=topic_sentiment_pct['topic_label'],
            y=topic_sentiment_pct[sentiment],
            marker_color=color
        ))
fig.update_layout(
    barmode='stack',
    xaxis_title='Topic',
    yaxis_title='Proportion',
    legend_title='Sentiment',
    margin=dict(t=20, b=20)
)
st.plotly_chart(fig, use_container_width=True)

# ─── Sentiment Trend Over Time ──────────────────────────────────────────
st.subheader("Sentiment Trend Over Time")
yearly = df.groupby(['year', 'sentiment']).size().unstack(fill_value=0)
yearly_pct = yearly.div(yearly.sum(axis=1), axis=0).reset_index()

fig = go.Figure()
for sentiment, color in SENTIMENT_COLORS.items():
    if sentiment in yearly_pct.columns:
        fig.add_trace(go.Scatter(
            x=yearly_pct['year'],
            y=yearly_pct[sentiment],
            name=sentiment,
            line=dict(color=color, width=2),
            mode='lines+markers'
        ))
fig.update_layout(
    xaxis_title='Year',
    yaxis_title='Proportion of reviews',
    legend_title='Sentiment',
    margin=dict(t=20, b=20)
)
st.plotly_chart(fig, use_container_width=True)