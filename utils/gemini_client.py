# utils/gemini_client.py
import os
from typing import Optional, Dict, Any

from dotenv import load_dotenv
import google.generativeai as genai

# Optional: Streamlit for secrets
try:
    import streamlit as st
except Exception:
    st = None

load_dotenv()

# -----------------------------------------------
#  READ API KEY SAFELY
# -----------------------------------------------
def _get_api_key() -> Optional[str]:
    key = os.getenv("GEMINI_API_KEY")
    if (not key or not key.strip()) and st is not None:
        try:
            key = st.secrets.get("GEMINI_API_KEY", None)
        except Exception:
            key = None

    if key:
        return key.strip()
    return None


API_KEY = _get_api_key()
AI_ENABLED = API_KEY not in (None, "")

if AI_ENABLED:
    try:
        genai.configure(api_key=API_KEY)
        _model = genai.GenerativeModel("gemini-pro")
    except Exception:
        _model = None
        AI_ENABLED = False
else:
    _model = None


# -----------------------------------------------
#  SAFE MODEL CALL — NO UI ERRORS EVER
# -----------------------------------------------
def _safe_call_model(prompt: str) -> str:
    """
    Returns "" (blank) if any error occurs.
    No UI error messages. No crashing.
    Local runs AI normally, deploy silently falls back.
    """
    if not AI_ENABLED or _model is None:
        return ""

    try:
        resp = _model.generate_content(prompt)
        return getattr(resp, "text", "").strip()
    except Exception:
        return ""   # silently ignore errors


# -----------------------------------------------
#  DATASET INSIGHTS
# -----------------------------------------------
def generate_dataset_insights(summary_dict: Dict[str, Any], extra_instructions: Optional[str] = None) -> str:
    base_prompt = f"""
    You are a professional academic analyst.

    Here is a dataset summary:
    {summary_dict}

    Provide:
    - Overall performance analysis
    - Key reasons for fail-risk
    - 3 recommendations to improve batch performance
    """

    if extra_instructions and extra_instructions.strip():
        base_prompt += f"\nAdditional context: {extra_instructions}"

    return _safe_call_model(base_prompt)


# -----------------------------------------------
#  STUDENT ADVICE
# -----------------------------------------------
def generate_student_advice(student_row: Dict[str, Any], extra_instructions: Optional[str] = None) -> str:
    # Remove Norm columns + ID-like fields
    clean_data = {
        k: v for k, v in student_row.items()
        if not k.endswith("_Norm") and k not in ["StudentID", "index"]
    }

    base_prompt = f"""
    You are an academic mentor.

    Student performance data:
    {clean_data}

    Provide:
    - Risk summary (On Track / Needs Attention)
    - Weak areas
    - 3 practical study recommendations
    """

    if extra_instructions and extra_instructions.strip():
        base_prompt += f"\nAdditional context: {extra_instructions}"

    return _safe_call_model(base_prompt)
