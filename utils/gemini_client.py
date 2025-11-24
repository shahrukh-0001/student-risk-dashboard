import os
import logging
import time
import random
from typing import Optional, Dict, Any

from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Setup simple logging to track issues
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Attempt to import streamlit for secrets handling
try:
    import streamlit as st
except ImportError:
    st = None

# ---- CONFIGURATION ----
# OPTIMIZED MODEL LIST:
# 1. gemini-1.5-flash-8b: Lightweight, highest rate limits, best for free tier dashboards.
# 2. gemini-1.5-flash: Standard fast model.
# 3. gemini-1.5-pro: Powerful but very strict rate limits (avoid unless necessary).
MODELS_TO_TRY = [
    "gemini-1.5-flash-8b", 
    "gemini-1.5-flash",
    "gemini-1.5-flash-latest",
    "gemini-1.5-pro"
]

def _get_api_key() -> Optional[str]:
    """
    Retrieves API key from Environment Variables or Streamlit Secrets.
    Returns None if not found.
    """
    key = os.getenv("GEMINI_API_KEY")
    if (not key or not key.strip()) and st is not None:
        try:
            key = st.secrets.get("GEMINI_API_KEY", None)
        except Exception:
            pass 
            
    if key:
        return key.strip()
    return None


def _initialize_genai():
    """Configures the Generative AI client safely."""
    api_key = _get_api_key()
    if not api_key:
        logger.warning("Gemini API Key not found. AI features will be disabled.")
        return False

    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        logger.error(f"Failed to configure Gemini API: {e}")
        return False


def _get_fallback_model_name() -> Optional[str]:
    """
    Auto-discovery: Queries the API for ANY available model that supports generation.
    Filters out experimental models to avoid strict quota limits.
    """
    try:
        logger.info("Attempting to auto-discover available models...")
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                # Avoid experimental models if possible as they often have 0 quota
                if "exp" not in m.name.lower():
                    logger.info(f"Auto-discovered valid model: {m.name}")
                    return m.name
    except Exception as e:
        logger.error(f"Auto-discovery failed: {e}")
    return None

def _generate_with_retry(model, prompt, config, safety_settings, max_retries=3):
    """
    Helper function to handle 429 Rate Limit errors with 'Bucket' backoff.
    Since Free Tier often has ~2 RPM (Requests Per Minute), short waits don't help.
    We need longer initial pauses.
    """
    for attempt in range(max_retries):
        try:
            return model.generate_content(
                prompt, 
                generation_config=config, 
                safety_settings=safety_settings
            )
        except Exception as e:
            error_str = str(e).lower()
            # Check for rate limit (429) or quota errors
            if "429" in error_str or "quota" in error_str:
                if attempt < max_retries - 1:
                    # SMART BACKOFF STRATEGY:
                    # Attempt 0: Wait 10s (Clear short spikes)
                    # Attempt 1: Wait 20s (Clear 2 RPM limit)
                    # Attempt 2: Wait 40s (Last ditch effort)
                    wait_time = (10 * (2 ** attempt)) + random.uniform(1, 3)
                    
                    logger.warning(f"Rate limit (429) on {model.model_name}. Pausing for {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
            
            # If it's not a rate limit, or we ran out of retries, raise the error
            raise e


def _safe_call_model(prompt: str) -> str:
    """
    Executes the API call with error handling, model fallback, and lenient safety settings.
    """
    if not _initialize_genai():
        return "⚠️ **AI Unavailable:** `GEMINI_API_KEY` not found. Please set it in your `.env` file or Streamlit secrets."

    last_error = None

    # ---- SAFETY SETTINGS ----
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }
    
    config = genai.GenerationConfig(
        temperature=0.7,
        max_output_tokens=800,
    )

    # 1. Try Hardcoded Preferred Models
    candidate_models = MODELS_TO_TRY.copy()
    
    # 2. Add Auto-discovered model (only if needed and strictly necessary)
    fallback = _get_fallback_model_name()
    if fallback and fallback not in candidate_models:
        candidate_models.append(fallback)

    # Iterate through models list to find one that works
    for model_name in candidate_models:
        try:
            model = genai.GenerativeModel(model_name)

            # Attempt generation with retry logic
            response = _generate_with_retry(model, prompt, config, safety_settings, max_retries=3)
            
            if not response.parts:
                try:
                    return response.text
                except ValueError:
                    logger.warning(f"Gemini ({model_name}) response was blocked.")
                    return "⚠️ **AI Analysis Unavailable:** The model refused to answer (Safety Block)."

            return response.text

        except Exception as e:
            # Check if this was a rate limit that persisted through retries
            if "429" in str(e) or "quota" in str(e).lower():
                logger.warning(f"Model '{model_name}' exhausted retries due to rate limits.")
                last_error = "Rate limit exceeded. Please wait a moment."
                
                # IMPORTANT: If we hit a rate limit, waiting a bit before trying the *next* model
                # helps prevent triggering a project-wide ban.
                time.sleep(2) 
            else:
                logger.warning(f"Failed to use model '{model_name}': {e}")
                last_error = e
            
            continue 

    logger.error(f"All Gemini models failed. Last error: {last_error}")
    
    # Return a friendly error message for the UI
    if "rate limit" in str(last_error).lower():
        return "⚠️ **AI Busy:** Daily/Minute quota exceeded. Please try again in 1-2 minutes."
        
    return f"⚠️ **AI Error:** Could not reach any Gemini model. ({str(last_error)})"


def generate_dataset_insights(summary_dict: Dict[str, Any], extra_instructions: Optional[str] = None) -> str:
    """
    Generates high-level insights for the entire dataset.
    """
    base_prompt = f"""
    You are an expert Academic Data Analyst.
    
    ANALYSIS CONTEXT:
    I have a dataset of student performance with the following summary metrics:
    {summary_dict}
    
    YOUR TASK:
    1. Analyze the overall health of the batch based on the pass/fail counts and averages.
    2. Identify the primary risk factors (e.g., is attendance low? are internal scores dragging down the total?).
    3. Provide 3 specific, actionable recommendations for the faculty to improve results.
    
    FORMATTING:
    - Use Markdown.
    - Use **Bold** for key terms.
    - Use bullet points for readability.
    - Keep it concise (under 200 words).
    """

    if extra_instructions and extra_instructions.strip():
        base_prompt += f"\n\nUSER NOTE: {extra_instructions}"

    return _safe_call_model(base_prompt)


def generate_student_advice(student_row: Dict[str, Any], extra_instructions: Optional[str] = None) -> str:
    """
    Generates personalized advice for a specific student.
    """
    clean_data = {k: v for k, v in student_row.items() if not k.endswith('_Norm')}

    base_prompt = f"""
    You are a supportive Academic Mentor.
    
    STUDENT DATA:
    {clean_data}
    
    YOUR TASK:
    1. Assess the student's current status (Safe, Borderline, or At-Risk).
    2. Identify their specific weak points (e.g., "Your Test 1 score was low" or "Attendance is critical").
    3. Provide 3 short, motivating, and practical steps they can take immediately to pass.
    
    FORMATTING:
    - Address the student directly (use "You").
    - Use Markdown.
    - Be encouraging but realistic.
    """

    if extra_instructions and extra_instructions.strip():
        base_prompt += f"\n\nUSER NOTE: {extra_instructions}"

    return _safe_call_model(base_prompt)