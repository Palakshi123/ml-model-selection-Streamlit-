import streamlit as st
import pandas as pd

from src.profiling import profile_dataset
from src.recommend import recommend_models
from src.baseline import build_and_eval_baseline

st.set_page_config(page_title="ML Model Chooser", layout="wide")

st.title("ML Model Chooser (Dataset → Model Recommendations)")
st.caption("Upload a CSV, pick a target, and get model suggestions + a runnable baseline.")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded is None:
    st.info("Upload a CSV to begin.")
    st.stop()

df = pd.read_csv(uploaded)
st.subheader("Preview")
st.dataframe(df.head(20), use_container_width=True)

with st.sidebar:
    st.header("Target selection")
    target_col = st.selectbox("Choose target column", options=["(none)"] + list(df.columns))
    problem_hint = st.selectbox("Problem hint (optional)", ["Auto-detect", "Classification", "Regression"])
    run_baseline = st.toggle("Run baseline model evaluation", value=True)
    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    random_state = st.number_input("Random state", min_value=0, value=42, step=1)

if target_col == "(none)":
    st.warning("Select a target column from the sidebar.")
    st.stop()

profile = profile_dataset(df, target_col=target_col, problem_hint=problem_hint)

col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("Dataset profile")
    st.json(profile)

with col2:
    st.subheader("Recommended models")
    rec = recommend_models(profile)
    st.write("### Primary shortlist")
    st.table(pd.DataFrame(rec["shortlist"]))
    st.write("### Why these?")
    st.write("\n".join([f"- {r}" for r in rec["rationale"]]))

st.divider()

if run_baseline:
    st.subheader("Baseline pipeline (quick eval)")
    try:
        results = build_and_eval_baseline(
            df=df,
            target_col=target_col,
            problem_type=profile["problem_type"],
            test_size=float(test_size),
            random_state=int(random_state),
        )
        st.write("### Results")
        st.table(pd.DataFrame(results["metrics"], index=[0]))
        st.write("### Model used")
        st.code(results["model_name"])
        st.write("### Notes")
        st.write("\n".join([f"- {n}" for n in results["notes"]]))
    except Exception as e:
        st.error(f"Baseline failed: {e}")
else:
    st.info("Baseline eval is turned off in the sidebar.")
