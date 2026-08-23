"""
InsightForge AI — LLM & Gemini Model Resolver
=============================================
Centralized Gemini AI client with automatic model discovery and multi-model fallback.
Handles API version variances and 404 model-not-found errors across different API keys.
"""

from typing import List, Optional
import os

FALLBACK_MODELS = [
    "gemini-1.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-exp",
    "gemini-1.5-flash-latest",
    "gemini-1.5-pro",
    "gemini-1.5-pro-latest",
    "gemini-pro"
]


def list_supported_models(gemini_key: str) -> List[str]:
    """Retrieves all models supported for content generation on the provided API key."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=gemini_key)
        available = []
        for m in genai.list_models():
            if "generateContent" in m.supported_generation_methods:
                clean_name = m.name.replace("models/", "")
                available.append(clean_name)
        return available
    except Exception:
        return []


def call_gemini(prompt: str, gemini_key: str) -> str:
    """
    Calls Gemini API with automatic model resolution and multi-model fallback.
    """
    if not gemini_key:
        raise ValueError("No Gemini API key provided.")

    import google.generativeai as genai
    genai.configure(api_key=gemini_key)

    # 1. Discover models supported by this specific key
    supported = list_supported_models(gemini_key)
    
    # Prioritize standard fast models
    candidate_order = []
    for pref in FALLBACK_MODELS:
        if pref in supported:
            candidate_order.append(pref)
    for other in supported:
        if other not in candidate_order:
            candidate_order.append(other)
    
    # If listing failed, try default candidate list
    if not candidate_order:
        candidate_order = FALLBACK_MODELS

    last_error = None
    for model_name in candidate_order:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            if response and response.text:
                return response.text.strip()
        except Exception as e:
            last_error = e
            continue

    if last_error:
        raise last_error

    return ""


def get_gemini_model(gemini_key: str):
    """
    Returns a configured GenerativeModel instance with fallback protection.
    """
    import google.generativeai as genai
    genai.configure(api_key=gemini_key)

    supported = list_supported_models(gemini_key)
    for candidate in FALLBACK_MODELS:
        if candidate in supported:
            return genai.GenerativeModel(candidate)

    if supported:
        return genai.GenerativeModel(supported[0])

    return genai.GenerativeModel("gemini-1.5-flash")
