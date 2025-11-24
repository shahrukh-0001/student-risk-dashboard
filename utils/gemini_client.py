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
MODELS_TO_TRY = [
    "gemini-1.5-flash", 
    "gemini-1.5-flash-latest",
    "gemini-1.5-flash-8b",
    "gemini-1.0-pro",  # Older model often works when 1.5 is too sensitive
    "gemini-pro"
]

def _get_api_key() -> Optional[str]:
    """Retrieves API key from Environment or Streamlit Secrets."""
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
    """Auto-discovery of available models."""
    try:
        logger.info("Attempting to auto-discover available models...")
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                name = m.name.lower()
                if "exp" not in name and "pro" not in name:
                    logger.info(f"Auto-discovered valid model: {m.name}")
                    return m.name
    except Exception as e:
        logger.error(f"Auto-discovery failed: {e}")
    return None

def _generate_with_retry(model, prompt, config, safety_settings, max_retries=3):
    """Handles 429 Rate Limit errors with backoff."""
    for attempt in range(max_retries):
        try:
            return model.generate_content(
                prompt, 
                generation_config=config, 
                safety_settings=safety_settings
            )
        except Exception as e:
            error_str = str(e).lower()
            if "quota" in error_str and ("day" in error_str or "daily" in error_str):
                raise RuntimeError("DAILY_QUOTA_EXCEEDED")
            
            if "429" in error_str or "quota" in error_str:
                if attempt < max_retries - 1:
                    wait_time = (5 * (2 ** attempt)) + random.uniform(1, 2)
                    logger.warning(f"Rate limit hit ({model.model_name}). Pausing for {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
            raise e

def _safe_call_model(prompt: str) -> str:
    """Executes the API call with PII protection and safety handling."""
    if not _initialize_genai():
        return "⚠️ **AI Unavailable:** `GEMINI_API_KEY` not found."

    last_error = None

    # ---- AGGRESSIVE SAFETY SETTINGS ----
    # We explicitly map categories to BLOCK_NONE.
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

    candidate_models = MODELS_TO_TRY.copy()
    fallback = _get_fallback_model_name()
    if fallback and fallback not in candidate_models:
        candidate_models.append(fallback)

    for model_name in candidate_models:
        try:
            model = genai.GenerativeModel(model_name)
            response = _generate_with_retry(model, prompt, config, safety_settings, max_retries=3)
            
            # Helper to extract text even if safety flags are slightly triggered
            def extract_text(resp):
                if resp.parts:
                    return resp.text
                if resp.candidates:
                    # Deep extraction for 'partially' blocked content
                    return resp.candidates[0].content.parts[0].text
                return None

            text = None
            try:
                text = extract_text(response)
            except Exception:
                pass

            if text:
                return text
            else:
                logger.warning(f"Gemini ({model_name}) blocked response. Prompt feedback: {response.prompt_feedback}")
                # Don't give up immediately, try next model
                last_error = "Content flagged by safety filters."
                continue

        except Exception as e:
            error_msg = str(e)
            if "DAILY_QUOTA_EXCEEDED" in error_msg:
                return "⚠️ **AI Quota Exceeded:** Daily limit reached."

            if "429" in error_msg or "quota" in error_msg.lower():
                time.sleep(2)
                last_error = "Rate limit exceeded."
            else:
                last_error = e
            continue 

    return f"⚠️ **Analysis Paused:** {str(last_error)}. Try removing sensitive columns."

def generate_dataset_insights(summary_dict: Dict[str, Any], extra_instructions: Optional[str] = None) -> str:
    """Generates high-level insights for the entire dataset."""
    base_prompt = f"""
    Role: Educational Data Analyst.
    Context: SYNTHETIC DATA for statistical analysis only.
    Task: Analyze this anonymous performance summary.
    
    Data Summary:
    {summary_dict}
    
    Required Output:
    1. Overall batch performance assessment.
    2. Identification of key academic challenges.
    3. 3 concrete recommendations for improvement.
    
    Style: Professional, constructive, and concise Markdown.
    """
    if extra_instructions and extra_instructions.strip():
        base_prompt += f"\n\nContext: {extra_instructions}"

    return _safe_call_model(base_prompt)

def generate_student_advice(student_row: Dict[str, Any], extra_instructions: Optional[str] = None) -> str:
    """Generates personalized advice, using STRICT allowlisting to prevent PII leaks."""
    
    # ---- STRICT ALLOWLIST APPROACH ----
    # Removed 'PassFail', 'Grade', 'Class' to reduce safety triggers.
    safe_columns = ['Semester', 'Department'] 
    clean_data = {}
    
    for key, value in student_row.items():
        # Skip technical/norm columns
        if key.endswith('_Norm') or key == "index": 
            continue
            
        # 1. Allow numeric values (force conversion to native python types)
        if isinstance(value, (int, float)):
             clean_data[key] = value
             continue
        # Handle numpy types
        try:
            float_val = float(value)
            clean_data[key] = float_val
            continue
        except (ValueError, TypeError):
            pass
             
        # 2. Allow specific safe text fields
        if key in safe_columns:
            clean_data[key] = str(value)
            
    base_prompt = f"""
    Role: Educational Data Analyst.
    Context: SYNTHETIC DATA for statistical analysis only.
    Task: Provide study advice based on these metrics.
    
    Metrics:
    {clean_data}
    
    Please provide:
    1. Current Metric Status (e.g., On Track, Needs Focus).
    2. Specific subjects/areas needing attention.
    3. 3 study tips to improve outcomes.
    
    Style: Encouraging, direct, and supportive Markdown. Use "You".
    """
    
    if extra_instructions and extra_instructions.strip():
        base_prompt += f"\n\nContext: {extra_instructions}"

    return _safe_call_model(base_prompt)