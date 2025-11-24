# utils/gemini_client.py
import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# Optional: Streamlit ke bina bhi kaam chal sake isliye safe import
try:
    import streamlit as st
except Exception:
    st = None

import google.generativeai as genai


def _get_api_key() -> Optional[str]:
    """
    GEMINI_API_KEY ko pehle environment se, fir Streamlit secrets se read karta hai.
    """
    key = os.getenv("GEMINI_API_KEY")

    # Agar env me nahi hai aur Streamlit secrets available hain
    if (not key or not key.strip()) and st is not None:
        try:
            key = st.secrets.get("GEMINI_API_KEY", None)
        except Exception:
            key = None

    if key is not None:
        key = key.strip()

    return key


API_KEY = _get_api_key()
AI_ENABLED = API_KEY not in (None, "")

if AI_ENABLED:
    try:
        genai.configure(api_key=API_KEY)
        _model = genai.GenerativeModel("gemini-pro-latest")
    except Exception as e:
        print("Error configuring Gemini:", e)
        _model = None
        AI_ENABLED = False
else:
    _model = None


def _safe_call_model(prompt: str) -> str:
    """
    GenerateContent ko safe tarike se call karta hai, error aane par
    readable message return karta hai.
    """
    if not AI_ENABLED or _model is None:
        return (
            "⚠️ AI insights currently disabled. "
            "Please configure GEMINI_API_KEY in environment or Streamlit secrets."
        )

    try:
        resp = _model.generate_content(prompt)
        return getattr(resp, "text", str(resp))
    except Exception as e:
        print("Gemini API error:", e)
        return (
            "⚠️ Unable to fetch AI insights from Gemini right now.\n\n"
            f"Technical error: `{e}`"
        )


def generate_dataset_insights(summary_dict: dict, extra_instructions: Optional[str] = None) -> str:
    base_prompt = f"""
    You are an AI academic analyst.

    Here is a summary of student performance data:
    {summary_dict}

    Please:
    - Explain the overall performance and risk in simple terms.
    - Highlight key risk factors.
    - Give 3–5 actionable recommendations.

    Keep the answer short, structured, and easy to understand.
    """

    if extra_instructions and extra_instructions.strip():
        base_prompt += f"\n\nAdditional instructions:\n{extra_instructions}"

    return _safe_call_model(base_prompt)


def generate_student_advice(student_row_dict: dict, extra_instructions: Optional[str] = None) -> str:
    base_prompt = f"""
    You are a friendly academic mentor.

    Here is a student's data:
    {student_row_dict}

    Please give helpful advice:
    - Explain if the student is doing well or at risk.
    - Identify weak areas.
    - Give 3–4 actionable study suggestions.
    - Keep tone motivational.
    """

    if extra_instructions and extra_instructions.strip():
        base_prompt += f"\n\nAdditional instructions:\n{extra_instructions}"

    return _safe_call_model(base_prompt)
