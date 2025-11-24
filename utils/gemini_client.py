import os
import logging
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
# 1. Preferred models (Tried first)
MODELS_TO_TRY = [
    "gemini-1.5-flash",
    "gemini-1.5-flash-latest",
    "gemini-1.5-pro",
    "gemini-1.0-pro", 
    "gemini-pro"
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
    Used if all hardcoded model names fail.
    """
    try:
        logger.info("Attempting to auto-discover available models...")
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                logger.info(f"Auto-discovered valid model: {m.name}")
                return m.name
    except Exception as e:
        logger.error(f"Auto-discovery failed: {e}")
    return None


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
    
    # Define generation config
    config = genai.GenerationConfig(
        temperature=0.7,
        max_output_tokens=800,
    )

    # 1. Try Hardcoded Preferred Models
    candidate_models = MODELS_TO_TRY.copy()
    
    # 2. Add Auto-discovered model to the end of the list as a last resort
    fallback = _get_fallback_model_name()
    if fallback and fallback not in candidate_models:
        candidate_models.append(fallback)

    # Iterate through models list to find one that works
    for model_name in candidate_models:
        try:
            # Create model instance
            model = genai.GenerativeModel(model_name)

            # Attempt generation
            response = model.generate_content(
                prompt, 
                generation_config=config,
                safety_settings=safety_settings
            )
            
            # Check if response was blocked or empty
            if not response.parts:
                try:
                    return response.text
                except ValueError:
                    logger.warning(f"Gemini ({model_name}) response was blocked despite safety settings.")
                    return "⚠️ **AI Analysis Unavailable:** The model refused to answer (Safety Block)."

            return response.text

        except Exception as e:
            # 404 means model not found, try next. Other errors might be auth related.
            logger.warning(f"Failed to use model '{model_name}': {e}")
            last_error = e
            continue 

    logger.error(f"All Gemini models failed. Last error: {last_error}")
    return f"⚠️ **AI Error:** Could not reach any Gemini model. (Last error: {str(last_error)})"


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
    # Clean up the dictionary for the prompt (remove norm columns if they exist to save tokens/noise)
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