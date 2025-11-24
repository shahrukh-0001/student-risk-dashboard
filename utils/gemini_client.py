# utils/gemini_client.py
import os
from typing import Optional

from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# Try to import streamlit (for secrets)
try:
    import streamlit as st
except Exception:
    st = None


def _get_api_key() -> Optional[str]:
    key = os.getenv("GEMINI_API_KEY")
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
        # IMPORTANT: ye model hi use karo
        _model = genai.GenerativeModel("gemini-pro-latest")
        print("Gemini model initialised successfully.")
    except Exception as e:
        print("Error configuring Gemini:", e)
        _model = None
        AI_ENABLED = False
else:
    _model = None


def _safe_call_model(prompt: str) -> str:
    """
    Yahi se actual Gemini call hota hai.
    Error aaye to app crash nahi karega.
    """
    if not AI_ENABLED or _model is None:
        return (
            "⚠️ AI insights are currently disabled. "
            "Please configure GEMINI_API_KEY in Streamlit secrets or environment."
        )

    try:
        resp = _model.generate_content(prompt)
        return getattr(resp, "text", str(resp))
    except Exception as e:
        # Yahan sabhi Gemini errors catch ho jayenge
        print("Gemini API error:", e)
        return (
            "⚠️ Unable to fetch AI insights from Gemini right now.\n\n"
            f"Technical error (hidden in UI): `{e}`"
        )


def generate_dataset_insights(summary_dict: dict, extra_instructions: Optional[str] = None) -> str:
    base_prompt = f"""
    You are an AI academic analyst.

    Here is a summary of student performance data:
    {summary_dict}

    Please:
    - Explain the overall performance and risk in simple terms.
    - Highlight key risk factors (attendance, internal tests, final exam).
    - Give 3–5 actionable recommendations for faculty to improve pass rates.

    Keep the answer short, structured, and easy to understand.
    """

    if extra_instructions and extra_instructions.strip():
        base_prompt += f"\n\nAdditional instructions from the user:\n{extra_instructions}"

    return _safe_call_model(base_prompt)


def generate_student_advice(student_row_dict: dict, extra_instructions: Optional[str] = None) -> str:
    base_prompt = f"""
    You are a friendly academic mentor.

    Here is a student's data (including predicted fail probability):
    {student_row_dict}

    Please:
    - Explain in simple language if the student seems safe or at risk.
    - Point out the weakest areas (like tests, attendance, final exam).
    - Give 3–4 very practical and motivating suggestions.

    Keep tone positive and supportive.
    """

    if extra_instructions and extra_instructions.strip():
        base_prompt += f"\n\nAdditional instructions from the user:\n{extra_instructions}"

    return _safe_call_model(base_prompt)
