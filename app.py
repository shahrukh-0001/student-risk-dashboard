import os
import pickle
from io import StringIO
from typing import Tuple, List, Optional, Dict, Any

import pandas as pd
import streamlit as st

# Ensure these are strictly imported from your utils
from utils.gemini_client import (
    generate_dataset_insights,
    generate_student_advice,
)

# ---- CONFIGURATION ----
MODEL_PATH = "model.pkl"
EXAMPLE_CSV = (
    "StudentID,Name,Semester,AttendancePercent,AssignmentScore,Test1,Test2,Test3,FinalExam,Total,PassFail,"
    "AttendancePercent_Norm,AssignmentScore_Norm,Test1_Norm,Test2_Norm,Test3_Norm,FinalExam_Norm\n"
    "1,Rahul,1,85,80,78,75,82,88,403,Pass,0.85,0.80,0.78,0.75,0.82,0.88"
)

st.set_page_config(
    page_title="Student Performance Dashboard",
    page_icon="🎓",
    layout="wide",
)


# ---- LOGIC & HELPERS ----

@st.cache_resource
def load_model(model_path: str) -> Tuple[Any, List[str]]:
    """Loads the ML model and feature columns from a pickle file."""
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found. Please run 'python model_training.py' first.")
        st.stop()
    
    with open(model_path, "rb") as f:
        obj = pickle.load(f)
    
    return obj["model"], obj["feature_cols"]


@st.cache_data
def run_predictions(df: pd.DataFrame, _model, feature_cols: List[str]) -> pd.DataFrame:
    """
    Runs predictions on the dataframe.
    Cached so it doesn't re-run on every UI interaction.
    Note: _model is underscored to prevent hashing issues with sklearn objects in Streamlit cache.
    """
    # Validate columns
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        st.error(f"⚠️ The data is missing required columns for the model: {missing}")
        st.stop()

    X = df[feature_cols]
    
    try:
        fail_probs = _model.predict_proba(X)[:, 1]
    except Exception as e:
        st.error(f"Prediction Error: {str(e)}")
        st.stop()

    df_out = df.copy()
    df_out["FailProbability"] = fail_probs

    # Generate PassFail label if missing, based on probability
    if "PassFail" not in df_out.columns:
        df_out["PassFail"] = df_out["FailProbability"].apply(
            lambda p: "Fail" if p > 0.5 else "Pass"
        )
    return df_out


def get_data_from_sidebar() -> Optional[pd.DataFrame]:
    """Handles the sidebar UI for data input and returns a raw DataFrame."""
    st.sidebar.header("📂 Data Input")
    mode = st.sidebar.radio("Input method", ["Upload file", "Paste text (CSV)"])

    if mode == "Upload file":
        uploaded_file = st.sidebar.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"])
        if uploaded_file:
            try:
                if uploaded_file.name.endswith(".csv"):
                    return pd.read_csv(uploaded_file)
                else:
                    return pd.read_excel(uploaded_file)
            except Exception as e:
                st.sidebar.error(f"Error reading file: {e}")
    else:
        st.sidebar.info("Paste data with a header row.")
        pasted_text = st.sidebar.text_area("Paste CSV text", value=EXAMPLE_CSV, height=200)
        if pasted_text.strip():
            try:
                return pd.read_csv(StringIO(pasted_text))
            except Exception as e:
                st.sidebar.error(f"CSV Parsing Error: {e}")
    
    return None


# ---- UI COMPONENT FUNCTIONS ----

def render_overview(df: pd.DataFrame):
    """Renders the Overview tab content."""
    st.markdown("### 📊 Overall Summary")

    # Metrics
    total_students = len(df)
    pass_count = (df["PassFail"] == "Pass").sum()
    fail_count = (df["PassFail"] == "Fail").sum()
    avg_prob = df["FailProbability"].mean()
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Students", total_students)
    c2.metric("Passing", pass_count, delta=f"{(pass_count/total_students)*100:.1f}%")
    c3.metric("Failing", fail_count, delta_color="inverse")
    c4.metric("Avg Fail Risk", f"{avg_prob:.2%}")

    # Charts
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("#### Outcome Distribution")
        st.bar_chart(df["PassFail"].value_counts(), color="#FF4B4B")

    with col_right:
        st.markdown("#### Top Performers")
        sort_col = "Total" if "Total" in df.columns else "FinalExam"
        if sort_col in df.columns:
            top_df = df.sort_values(sort_col, ascending=False).head(10)
            if "Name" in top_df.columns:
                st.bar_chart(top_df.set_index("Name")[sort_col])
            else:
                st.dataframe(top_df[[sort_col, "PassFail"]])


def render_risk_analysis(df: pd.DataFrame):
    """Renders the At-Risk tab content."""
    st.markdown("### ⚠️ At-Risk Analysis")

    col_controls, col_table = st.columns([1, 3])

    with col_controls:
        st.markdown("**Filters**")
        risk_threshold = st.slider("Fail Probability Threshold", 0.0, 1.0, 0.6, 0.05)
        
        # Dynamic Filters based on available columns
        sem_filter = "All"
        if "Semester" in df.columns:
            sem_options = ["All"] + sorted(df["Semester"].dropna().astype(str).unique().tolist())
            sem_filter = st.selectbox("Semester", sem_options)

        dept_filter = "All"
        if "Department" in df.columns:
            dept_options = ["All"] + sorted(df["Department"].dropna().unique().tolist())
            dept_filter = st.selectbox("Department", dept_options)

    # Apply Filtering
    mask = df["FailProbability"] >= risk_threshold
    if sem_filter != "All":
        mask &= (df["Semester"].astype(str) == sem_filter)
    if dept_filter != "All":
        mask &= (df["Department"] == dept_filter)

    filtered_df = df[mask].copy()

    with col_table:
        st.info(f"Found **{len(filtered_df)}** students matching criteria.")
        display_cols = [c for c in ["StudentID", "Name", "Semester", "Total", "FailProbability"] if c in df.columns]
        st.dataframe(filtered_df[display_cols], use_container_width=True)
        
        if not filtered_df.empty:
            st.download_button(
                "⬇️ Download Risk List",
                data=filtered_df.to_csv(index=False).encode("utf-8"),
                file_name="at_risk_students.csv",
                mime="text/csv"
            )


def render_ai_section(df: pd.DataFrame):
    """Renders the AI Insights and Student Advice sections."""
    st.markdown("---")
    
    # 1. Dataset Insights
    st.subheader("🤖 AI Dataset Insights")
    with st.expander("Generate High-Level Analysis", expanded=True):
        dataset_extra = st.text_input("Specific Question for AI (Optional)", placeholder="E.g., Suggest interventions for semester 1 students.")
        
        if st.button("Analyze Dataset"):
            # Prepare summary dict
            summary = {
                "total_students": len(df),
                "pass_count": int((df["PassFail"] == "Pass").sum()),
                "fail_count": int((df["PassFail"] == "Fail").sum()),
                "avg_fail_prob": float(df["FailProbability"].mean()),
            }
            if "AttendancePercent" in df.columns:
                summary["avg_attendance"] = float(df["AttendancePercent"].mean())
            
            with st.spinner("Consulting Gemini..."):
                insights = generate_dataset_insights(summary, dataset_extra)
                if insights:
                    st.markdown(insights)

    # 2. Student Advice
    st.subheader("👤 Individual Student Advice")
    if "StudentID" not in df.columns:
        st.warning("Dataset requires a 'StudentID' column for individual advice.")
        return

    ids = df["StudentID"].unique().tolist()
    c1, c2 = st.columns([1, 3])
    
    with c1:
        selected_id = st.selectbox("Select Student ID", ids)
        student_extra = st.text_area("Custom Instruction", placeholder="E.g., Explain in Hindi", height=100)
        generate_btn = st.button("Get Student Advice")

    with c2:
        student_row = df[df["StudentID"] == selected_id].iloc[0]
        st.caption("Student Data Preview:")
        st.dataframe(student_row.to_frame().T, hide_index=True)

        if generate_btn:
            with st.spinner(f"Generating advice for Student {selected_id}..."):
                advice = generate_student_advice(student_row.to_dict(), student_extra)
                st.success("AI Analysis Generated:")
                st.markdown(advice)


# ---- MAIN ----

def main():
    st.title("🎓 Student Performance & Risk Dashboard")
    
    # 1. Load Resources
    try:
        model, feature_cols = load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"System Error: {e}")
        return

    # 2. Get Data
    raw_df = get_data_from_sidebar()
    
    if raw_df is None:
        st.info("👈 Please upload a file or paste CSV data to begin.")
        st.markdown("#### Raw Data Preview (Empty)")
        return

    # 3. Process Data (Add Predictions)
    df = run_predictions(raw_df, model, feature_cols)
    
    with st.expander("View Raw Data with Predictions"):
        st.dataframe(df.head())

    # 4. Dashboard Tabs
    tab_overview, tab_risk = st.tabs(["📊 Overview", "⚠️ At-Risk Students"])
    
    with tab_overview:
        render_overview(df)
    
    with tab_risk:
        render_risk_analysis(df)

    # 5. AI Section
    render_ai_section(df)


if __name__ == "__main__":
    main()