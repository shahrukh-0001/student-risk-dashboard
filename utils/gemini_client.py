# utils/gemini_client.py
import os
from typing import Optional

from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
AI_ENABLED = API_KEY is not None and API_KEY.strip() != ""

if AI_ENABLED:
    genai.configure(api_key=API_KEY)
    _model = genai.GenerativeModel("gemini-2.5-flash")
else:
    _model = None


def generate_dataset_insights(summary_dict: dict, extra_instructions: Optional[str] = None) -> str:
    """
    Dataset level insights: teacher / admin ke liye.
    """
    if not AI_ENABLED or _model is None:
        return (
            "⚠️ Gemini API key configured nahi hai (.env me GEMINI_API_KEY set karo), "
            "isliye AI insights disabled hain."
        )

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

    resp = _model.generate_content(base_prompt)
    return resp.text


def generate_student_advice(student_row_dict: dict, extra_instructions: Optional[str] = None) -> str:
    """
    Single student ke liye personalised advice.
    """
    if not AI_ENABLED or _model is None:
        return (
            "⚠️ GEMINI_API_KEY set nahi hai, isliye student-wise AI advice disabled hai."
        )

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

    resp = _model.generate_content(base_prompt)
    return resp.text
