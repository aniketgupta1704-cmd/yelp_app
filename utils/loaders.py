"""Model and data loaders — all cached with Streamlit."""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import streamlit as st
import pandas as pd
import numpy as np
import faiss
import torch
torch.set_num_threads(1)

from sentence_transformers import SentenceTransformer
from transformers import (
    DistilBertForSequenceClassification,
    AutoTokenizer
)
from huggingface_hub import InferenceClient


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')


@st.cache_resource
def load_sentiment_model():
    tokenizer = AutoTokenizer.from_pretrained('./data/distilbert_sentiment')
    model = DistilBertForSequenceClassification.from_pretrained(
        './data/distilbert_sentiment'
    )
    model.eval()
    model.to("cpu")
    return tokenizer, model


@st.cache_data
def load_reviews():
    df = pd.read_csv('./data/app_reviews.csv')
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    return df


@st.cache_resource
def load_faiss_index():
    return faiss.read_index('./data/faiss_index.bin')


@st.cache_data
def load_embeddings():
    """Load raw embeddings for UMAP visualization."""
    return np.load('./data/embeddings.npy')


@st.cache_resource
def load_hf_client():
    return InferenceClient(token=st.secrets["HF_TOKEN"])

@st.cache_resource
def load_local_llm():
    """Fallback local LLM — small, CPU-friendly."""
    from transformers import pipeline
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        device=-1
    )

@st.cache_data
def compute_2d_projection():
    """Compute 2D t-SNE projection of all embeddings. Cached — runs once."""
    from sklearn.manifold import TSNE
    embeddings = load_embeddings()
    
    # Sample for speed — t-SNE is slower than UMAP
    n_samples = min(5000, len(embeddings))
    sample_indices = np.random.RandomState(42).choice(
        len(embeddings), n_samples, replace=False
    )
    sample_embeddings = embeddings[sample_indices]
    
    reducer = TSNE(
        n_components=2,
        perplexity=30,
        random_state=42,
        init='pca',
        learning_rate='auto'
    )
    projection = reducer.fit_transform(sample_embeddings)
    return projection, sample_indices