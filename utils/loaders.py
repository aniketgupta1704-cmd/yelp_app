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

from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import (
    DistilBertForSequenceClassification,
    AutoTokenizer
)
from huggingface_hub import InferenceClient, hf_hub_download, snapshot_download

# HF Dataset repo containing app artifacts
HF_DATASET_REPO = "aniket32/yelp-review-intelligence-artifacts"

# Local paths (populated from HF on first run)
DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)


def _ensure_file(filename: str) -> str:
    """Download a single file from HF Dataset if not present locally."""
    local_path = DATA_DIR / filename
    if not local_path.exists():
        hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename=filename,
            repo_type="dataset",
            local_dir=str(DATA_DIR),
        )
    return str(local_path)


def _ensure_model_folder() -> str:
    """Download the entire distilbert_sentiment folder if not present."""
    local_folder = DATA_DIR / "distilbert_sentiment"
    if not local_folder.exists() or not any(local_folder.iterdir()):
        snapshot_download(
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            allow_patterns="distilbert_sentiment/*",
            local_dir=str(DATA_DIR),
        )
    return str(local_folder)

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')


@st.cache_resource
def load_sentiment_model():
    model_path = _ensure_model_folder()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    model.eval()
    model.to("cpu")
    return tokenizer, model


@st.cache_data
def load_reviews():
    csv_path = _ensure_file("app_reviews.csv")
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    return df


@st.cache_resource
def load_faiss_index():
    index_path = _ensure_file("faiss_index.bin")
    return faiss.read_index(index_path)


@st.cache_data
def load_embeddings():
    """Load raw embeddings for t-SNE visualization."""
    emb_path = _ensure_file("embeddings.npy")
    return np.load(emb_path)


@st.cache_resource
def load_hf_client():
    return InferenceClient(token=st.secrets["HF_TOKEN"])


@st.cache_resource
def load_local_llm():
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