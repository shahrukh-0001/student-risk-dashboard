# app.py
import os
import pickle
from io import StringIO

import pandas as pd
import streamlit as st

from utils.gemini_client import (
    generate_dataset_insights,
    generate_student_advice,
)

st.set_page_config(
    page_title="Student Performance & Risk Dashboard",
    layout="wide",
)


@st.cache_resource
def load_model(model_path: str = "model.pkl"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file '{model_path}' not found. "
            f"Please run 'python model_training.py' first."
        )
    with open(model_path, "rb") as f:
        obj = pickle.load(f)
    return obj["model"], obj["feature_cols"]


def add_fail_probability(df: pd.DataFrame, model, feature_cols: list) -> pd.DataFrame:
    # Check required columns
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        st.error(f"Uploaded/pasted data is missing required columns: {missing}")
        return df

    X = df[feature_cols]
    fail_probs = model.predict_proba(X)[:, 1]  # Probability of Fail (class 1)

    df = df.copy()
    df["FailProbability"] = fail_probs

    # Agar PassFail nahi hai to threshold se create
    if "PassFail" not in df.columns:
        df["PassFail"] = df["FailProbability"].apply(
            lambda p: "Fail" if p > 0.5 else "Pass"
        )
    return df


def main():
    st.title("🎓 Student Performance & Risk Analysis Dashboard")

    # ---- SIDEBAR: DATA INPUT MODE ----
    st.sidebar.header("Data Input Mode")

    mode = st.sidebar.radio(
        "Select data input method",
        ["Upload file", "Paste text (CSV format)"],
    )

    uploaded_file = None
    pasted_text = ""

    if mode == "Upload file":
        uploaded_file = st.sidebar.file_uploader(
            "Upload student data (CSV / Excel). "
            "Format should be like 'student_datasheet.csv'.",
            type=["csv", "xlsx"],
        )
    else:
        st.sidebar.markdown("Paste data with a **header row** (comma separated).")
        example = (
            "StudentID,Name,Semester,AttendancePercent,AssignmentScore,Test1,Test2,Test3,FinalExam,Total,PassFail,"
            "AttendancePercent_Norm,AssignmentScore_Norm,Test1_Norm,Test2_Norm,Test3_Norm,FinalExam_Norm\n"
            "1,Rahul,1,85,80,78,75,82,88,403,Pass,0.85,0.80,0.78,0.75,0.82,0.88"
        )
        pasted_text = st.sidebar.text_area(
            "Paste CSV text here",
            value=example,
            height=200,
        )

    # ---- MODEL LOAD ----
    try:
        model, feature_cols = load_model("model.pkl")
    except Exception as e:
        st.error(str(e))
        st.stop()

    # ---- READ DATAFRAME ----
    if mode == "Upload file":
        if uploaded_file is None:
            st.info(
                "Please upload a student data file to start, "
                "or switch to 'Paste text (CSV format)'."
            )
            return

        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    else:
        if not pasted_text.strip():
            st.info("Please paste CSV formatted data in the text box.")
            return
        try:
            df = pd.read_csv(StringIO(pasted_text))
        except Exception as e:
            st.error(f"Error while reading pasted text as CSV: {e}")
            return

    st.subheader("📄 Raw Data Preview")
    st.dataframe(df.head())

    # ---- ADD FAIL PROBABILITY ----
    df = add_fail_probability(df, model, feature_cols)
    if "FailProbability" not in df.columns:
        st.stop()

    # ---- PRE-COMPUTE SUMMARY METRICS (use in tabs + AI) ----
    total_students = len(df)
    pass_count = (df["PassFail"] == "Pass").sum()
    fail_count = (df["PassFail"] == "Fail").sum()
    avg_fail_prob = df["FailProbability"].mean()

    avg_attendance = (
        df["AttendancePercent"].mean()
        if "AttendancePercent" in df.columns
        else None
    )
    avg_total = df["Total"].mean() if "Total" in df.columns else None

    # ---- TABS: OVERVIEW + AT-RISK ----
    st.markdown("## 📊 Dashboard")
    tab_overview, tab_risk = st.tabs(["Overview", "At-Risk Students"])

    # ====== OVERVIEW TAB ======
    with tab_overview:
        st.markdown("### 📊 Overall Summary")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Students", total_students)
        c2.metric("Pass Count", pass_count)
        c3.metric("Fail Count", fail_count)
        c4.metric("Avg Fail Probability", f"{avg_fail_prob:.2f}")

        c5, c6 = st.columns(2)
        if avg_attendance is not None:
            c5.metric("Average Attendance (%)", f"{avg_attendance:.1f}")
        if avg_total is not None:
            c6.metric("Average Total Marks", f"{avg_total:.1f}")

        # ---- TOP PERFORMERS ----
        st.markdown("### 🏆 Top Performers")

        basis_options = [
            col for col in ["Total", "Average", "FinalExam"] if col in df.columns
        ]
        if not basis_options:
            st.warning(
                "No 'Total', 'Average' or 'FinalExam' columns found for ranking."
            )
        else:
            basis = st.selectbox("Sort by", basis_options, index=0)
            top_n = st.slider("Show Top N", 3, 20, 10)

            top_df = df.sort_values(basis, ascending=False).head(top_n)

            show_cols = [
                col
                for col in [
                    "StudentID",
                    "Name",
                    basis,
                    "FailProbability",
                    "PassFail",
                ]
                if col in top_df.columns
            ]
            st.dataframe(top_df[show_cols])

            if "Name" in top_df.columns:
                st.bar_chart(
                    data=top_df.set_index("Name")[basis],
                )

        # ---- PASS / FAIL DISTRIBUTION ----
        st.markdown("### 📈 Outcome Distribution")
        st.bar_chart(df["PassFail"].value_counts())

        # ---- SEMESTER-WISE ANALYSIS ----
        if "Semester" in df.columns and "Total" in df.columns:
            st.markdown("### 🗓️ Semester-wise Average Total Marks")
            sem_group = df.groupby("Semester")["Total"].mean().reset_index()
            sem_group = sem_group.sort_values("Semester")
            st.line_chart(sem_group.set_index("Semester")["Total"])

        # ---- DEPARTMENT-WISE ANALYSIS ----
        if "Department" in df.columns and "Total" in df.columns:
            st.markdown("### 🏫 Department-wise Average Total Marks")
            dept_group = df.groupby("Department")["Total"].mean().reset_index()
            st.bar_chart(dept_group.set_index("Department")["Total"])

        # ---- DOWNLOAD FULL DATASET CSV ----
        st.markdown("### ⬇️ Download Full Predictions CSV")
        full_csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Full Dataset with Predictions",
            data=full_csv,
            file_name="student_predictions.csv",
            mime="text/csv",
        )

    # ====== AT-RISK TAB ======
    with tab_risk:
        st.markdown("### ⚠️ At-Risk Students")

        # Threshold slider
        risk_threshold = st.slider(
            "Minimum fail probability to consider as at-risk",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.05,
        )

        filtered = df[df["FailProbability"] >= risk_threshold].copy()

        # Optional filters
        col_f1, col_f2, col_f3 = st.columns(3)

        if "Semester" in df.columns:
            semesters = ["All"] + sorted(
                df["Semester"].dropna().unique().tolist()
            )
            selected_sem = col_f1.selectbox("Filter by Semester", semesters)
            if selected_sem != "All":
                filtered = filtered[filtered["Semester"] == selected_sem]

        if "Department" in df.columns:
            depts = ["All"] + sorted(
                df["Department"].dropna().unique().tolist()
            )
            selected_dept = col_f2.selectbox("Filter by Department", depts)
            if selected_dept != "All":
                filtered = filtered[filtered["Department"] == selected_dept]

        if "AttendancePercent" in df.columns:
            min_att = col_f3.slider(
                "Minimum attendance (%)",
                min_value=0,
                max_value=100,
                value=0,
                step=5,
            )
            filtered = filtered[filtered["AttendancePercent"] >= min_att]

        st.write(f"Total at-risk students: {len(filtered)}")

        show_cols_risk = [
            col
            for col in [
                "StudentID",
                "Name",
                "Semester",
                "Department",
                "AttendancePercent",
                "Total",
                "FailProbability",
                "PassFail",
            ]
            if col in filtered.columns
        ]

        st.dataframe(filtered[show_cols_risk])

        # CSV download for at-risk list
        if not filtered.empty:
            risk_csv = filtered.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download At-Risk Students CSV",
                data=risk_csv,
                file_name="at_risk_students.csv",
                mime="text/csv",
            )
        else:
            st.info("No students found for the selected filters.")

    # ---- DATASET-LEVEL AI INSIGHTS ----
    st.markdown("## 🤖 AI Insights for This Dataset")

    dataset_extra = st.text_area(
        "Optional extra instructions for AI (leave blank for default behaviour)",
        "",
        height=80,
    )

    if st.button("Generate AI Insights"):
            summary_dict = {
        "total_students": int(total_students),
        "pass_count": int(pass_count),
        "fail_count": int(fail_count),
        "avg_attendance": float(avg_attendance) if avg_attendance is not None else None,
        "avg_total": float(avg_total) if avg_total is not None else None,
        "avg_fail_probability": float(avg_fail_prob),
    }

    try:
        with st.spinner("Asking AI for insights..."):
            insights = generate_dataset_insights(summary_dict, dataset_extra)
        st.markdown(insights)
    except Exception as e:
        st.error("⚠️ AI insights are temporarily unavailable.")
        st.text(str(e))


    # ---- STUDENT-WISE AI ADVICE ----
    st.markdown("## 👤 Student-wise AI Advice")

    student_extra = st.text_area(
        "Optional extra instructions for student advice (e.g. 'Explain in Hindi also')",
        "",
        height=80,
    )

    id_col = "StudentID" if "StudentID" in df.columns else None
    if id_col:
        ids = df[id_col].unique().tolist()
        selected_id = st.selectbox("Select Student ID", ids)

        student_row = df[df[id_col] == selected_id].iloc[0]
        st.write("Selected Student Data:")
        st.dataframe(student_row.to_frame().T)

        if st.button("Generate Advice for this Student"):
            with st.spinner("Generating advice..."):
                advice = generate_student_advice(
                    student_row.to_dict(), student_extra
                )
            st.markdown(advice)
    else:
        st.info("StudentID column not found. Individual advice disabled.")


if __name__ == "__main__":
    main()
