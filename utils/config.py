"""Shared configuration for the app."""

# Color scheme used across all charts
SENTIMENT_COLORS = {
    'positive': '#1D9E75',
    'neutral': '#888780',
    'negative': '#E24B4A'
}

SENTIMENT_EMOJI = {
    'positive': '🟢',
    'neutral': '🟡',
    'negative': '🔴'
}

# Model information for display on "How It Works" page
MODEL_INFO = {
    'embedding': {
        'name': 'all-MiniLM-L6-v2',
        'provider': 'Sentence Transformers',
        'dimensions': 384,
        'use': 'Semantic embeddings for search'
    },
    'sentiment': {
        'name': 'DistilBERT (fine-tuned)',
        'provider': 'Hugging Face',
        'params': '66M',
        'use': 'Sentiment classification'
    },
    'topic': {
        'name': 'BERTopic',
        'provider': 'MaartenGr',
        'components': 'UMAP + HDBSCAN + c-TF-IDF',
        'use': 'Unsupervised topic discovery'
    },
    'search': {
        'name': 'FAISS IndexFlatIP',
        'provider': 'Meta AI',
        'vectors': '20,000',
        'use': 'Fast similarity search'
    },
    'rag': {
        'name': 'Qwen2.5-7B-Instruct (with fallback)',
        'provider': 'HuggingFace Inference',
        'fallback': 'flan-t5-base (local)',
        'use': 'RAG summarization'
    }
}

# Evaluation metrics from Day 2 and RAG evaluation
EVAL_METRICS = {
    'sentiment': {
        'zero_shot_bart_f1': 0.573,
        'finetuned_distilbert_f1': 0.783,
        'improvement': 0.210
    },
    'rag': {
        'precision_at_1': 0.900,
        'precision_at_3': 0.900,
        'precision_at_5': 0.880,
        'mean_reciprocal_rank': 0.950,
        'queries_tested': 10,
        'avg_latency_ms': 10
    }
}

# Data pipeline stats
DATA_STATS = {
    'total_yelp_reviews': 200_000,
    'balanced_sample': 50_000,
    'embedded_sample': 20_000,
    'date_range': '2005-2019',
    'topics_discovered': 6
}