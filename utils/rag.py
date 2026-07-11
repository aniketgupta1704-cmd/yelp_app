"""RAG generation with dynamic model discovery via HF Hub API."""

import time
import re
import requests
import streamlit as st
from openai import OpenAI
from utils.loaders import load_local_llm


@st.cache_data(ttl=3600)  # Refresh model list every hour
def fetch_available_models(_token):
    """Query HF Hub for currently trending inference-available text-generation models."""
    url = (
        "https://huggingface.co/api/models"
        "?inference_provider=all"
        "&pipeline_tag=text-generation"
        "&sort=likes"
        "&direction=-1"
        "&limit=30"
    )

    try:
        response = requests.get(
            url,
            headers={"Authorization": f"Bearer {_token}"},
            timeout=10
        )
        response.raise_for_status()
        models = response.json()

        # Extract just model IDs
        return [m.get('id', '') for m in models if m.get('id')]
    except Exception as e:
        print(f"Model discovery failed: {e}")
        return []


def generate_rag_answer(query, retrieved_reviews, client=None):
    context = ""
    for i, r in enumerate(retrieved_reviews):
        context += f"Review {i+1} (Stars: {r['stars']}, Sentiment: {r['sentiment']}):\n"
        context += f"{r['text'][:400]}\n\n"

    prompt = f"""You are a business analyst reviewing customer feedback.

Query from a business owner: "{query}"

Customer reviews:
{context}

Write a business intelligence report in this exact format:

**Overall Theme:** [one sentence about the pattern]

**Key Observation:** [one sentence with a concrete detail]

**Business Recommendation:** [one sentence with a specific action]

Do not explain what you are doing. Ellaborate on Key Observations section by mentioning specific Review Ids. Do not mention your process. Write only the report. Under 100 words total."""

    token = st.secrets["HF_TOKEN"]
    openai_client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=token,
    )

    # Discover trending inference-available models
    available_models = fetch_available_models(token)

    if not available_models:
        return ("Model discovery failed. Check HF connection.", "None", 0)

    # Filter out models that likely won't work as chat (base models without instruct)
    # and models that are too niche/small
    chat_models = [
        m for m in available_models
        if 'Instruct' in m or 'chat' in m.lower() or 'gpt-oss' in m or 'DeepSeek' in m
        or 'GLM' in m or 'Kimi' in m or 'QwQ' in m
    ]

    # If filtering removed everything, use the full list
    if not chat_models:
        chat_models = available_models

    t0 = time.time()
    errors_seen = []

    for model_id in chat_models[:10]:  # Try up to 10 models
        try:
            response = openai_client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,  # Increased for reasoning models
            temperature=0.3,
        )
            content = response.choices[0].message.content

            # Strip </think> tag if present
            content = content.replace("</think>", "").replace("<think>", "").strip()

                        # Extract only the 3 sections we want using regex
            sections = {}
            patterns = {
                "Overall Theme": r"(?:\*\*)?Overall Theme(?:\*\*)?:\s*(.+?)(?=(?:\*\*)?Key Observation(?:\*\*)?:|$)",
                "Key Observation": r"(?:\*\*)?Key Observation(?:\*\*)?:\s*(.+?)(?=(?:\*\*)?Business Recommendation(?:\*\*)?:|$)",
                "Business Recommendation": r"(?:\*\*)?Business Recommendation(?:\*\*)?:\s*(.+?)(?=(?:\*\*)?Overall Theme(?:\*\*)?:|\Z)"
            }

            for section_name, pattern in patterns.items():
                match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
                if match:
                    section_text = match.group(1).strip()
                    # Remove leading/trailing asterisks and whitespace
                    section_text = section_text.lstrip('*').strip()
                    section_text = section_text.rstrip('*').strip()
                    sections[section_name] = section_text

            # Rebuild clean output
            if len(sections) == 3:
                content = (
                    f"**Overall Theme:** {sections['Overall Theme']}\n\n"
                    f"**Key Observation:** {sections['Key Observation']}\n\n"
                    f"**Business Recommendation:** {sections['Business Recommendation']}"
                )
            else:
                # Fallback: if regex fails, use raw content but clean it up
                content = content.strip()

            content = content.strip()
            # Some models return None if they only used reasoning tokens
            if not content:
                errors_seen.append(f"{model_id}: empty response")
                continue

            latency_ms = (time.time() - t0) * 1000
            return (content, f"HF: {model_id}", latency_ms)

        except Exception as e:
            errors_seen.append(f"{model_id}: {str(e)[:150]}")
            continue

    print("=== HF FAILURES ===")
    for err in errors_seen:
        print(err)
    print("===================")

    # Local fallback
    try:
        local_llm = load_local_llm()
        short_prompt = (
            f"Summarize what customers say about '{query}' in 2 sentences and "
            f"give one business recommendation:\n\n{context[:1000]}"
        )
        result = local_llm(short_prompt, max_length=150, do_sample=False)
        latency_ms = (time.time() - t0) * 1000
        return result[0]['generated_text'], "Local: flan-t5-base", latency_ms
    except Exception as e:
        return f"All models unavailable. Error: {str(e)}", "None", 0