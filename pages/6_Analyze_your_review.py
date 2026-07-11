"""Analyze a single review — sentiment prediction + similar reviews."""

import streamlit as st
import torch
import plotly.express as px
from utils.loaders import (
    load_embedding_model,
    load_sentiment_model,
    load_reviews,
    load_faiss_index
)
from utils.search import semantic_search
from utils.config import SENTIMENT_COLORS, SENTIMENT_EMOJI

st.set_page_config(page_title="Analyze Your Review", page_icon="✍️", layout="wide")

# Load everything
embedding_model = load_embedding_model()
tokenizer, sentiment_model = load_sentiment_model()
df = load_reviews()
index = load_faiss_index()


def predict_sentiment(text, tokenizer, model):
    id2label = {0: 'negative', 1: 'neutral', 2: 'positive'}
    inputs = tokenizer(
        text[:512], return_tensors='pt', truncation=True,
        padding=True, max_length=128
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]
        pred = torch.argmax(probs).item()
    return id2label[pred], probs.numpy()


st.title("Analyze a Review")
st.markdown("Paste any review text and get instant sentiment analysis + similar reviews.")

user_review = st.text_area(
    "Paste a review here",
    placeholder="e.g. The food was amazing but the service was really slow...",
    height=150
)

if st.button("Analyze", type="primary") and user_review:
    with st.spinner("Analyzing..."):
        sentiment, probs = predict_sentiment(user_review, tokenizer, sentiment_model)

    col1, col2 = st.columns(2)

    with col1:
        color = SENTIMENT_COLORS[sentiment]
        st.markdown("### Predicted Sentiment")
        st.markdown(
            f"<h2 style='color:{color}'>{sentiment.upper()}</h2>",
            unsafe_allow_html=True
        )

    with col2:
        st.markdown("### Confidence Scores")
        labels = ['negative', 'neutral', 'positive']
        fig = px.bar(
            x=labels,
            y=[float(probs[0]), float(probs[1]), float(probs[2])],
            color=labels,
            color_discrete_map=SENTIMENT_COLORS,
            labels={'x': 'Sentiment', 'y': 'Confidence'}
        )
        fig.update_layout(
            showlegend=False, margin=dict(t=20, b=20), yaxis_range=[0, 1]
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Similar Reviews in Dataset")
    similar = semantic_search(user_review, embedding_model, index, df, top_k=3)
    for r in similar:
        with st.expander(
            f"{SENTIMENT_EMOJI[r['sentiment']]} "
            f"{'⭐' * int(r['stars'])} | "
            f"Relevance: {r['score']:.2f}"
        ):
            st.write(r['text'])