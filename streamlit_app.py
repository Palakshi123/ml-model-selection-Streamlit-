import json
import numpy as np
import pandas as pd
import streamlit as st

from src.profiling import profile_dataset
from src.recommend import recommend_models
from src.baseline import build_and_eval_baseline

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def simplify_rationale(rationale_list):
    mapping = {
        "nonlinear": "Your data has complex patterns (not straight-line relationships).",
        "linear": "Your data follows simpler, predictable patterns.",
        "small dataset": "This model works well even with limited data.",
        "large dataset": "This model scales well with large datasets.",
        "missing": "This model handles missing values well.",
        "categorical": "Your dataset contains categories (labels or names), and this model works well with them.",
        "high dimensional": "You have many columns, and this model can handle that.",
        "imbalanced": "Your target labels are uneven, and this model handles that well.",
        "baseline": "This is a good starting model for beginners.",
        "overfit": "This choice helps reduce overfitting (memorizing instead of learning).",
        "noise": "This model tends to handle noisy real-world data well.",
        "outlier": "This model is less sensitive to extreme values (outliers).",
    }

    results = []
    for r in rationale_list or []:
        r_low = str(r).lower()
        for k, v in mapping.items():
            if k in r_low:
                results.append(v)
                break
        else:
            results.append("This model matches your dataset characteristics well.")
    return results


def problem_type_sentence(profile, df, target_col):
    pt = str(profile.get("problem_type", "")).lower()
    y = df[target_col]

    if pt == "classification":
        n = y.nunique(dropna=True)
        examples = y.dropna().astype(str).unique()[:4]
        examples_str = ", ".join(map(str, examples))
        return (
            f"This is a **classification problem** because your target column "
            f"**'{target_col}'** has **{n} distinct label(s)** "
            f"(for example: {examples_str})."
        )

    if pt == "regression":
        return (
            f"This is a **regression problem** because your target column "
            f"**'{target_col}'** contains numeric values that vary continuously."
        )

    return "The problem type could not be confidently determined."


def format_advanced_block(profile, rec):
    lines = []

    tech = rec.get("rationale", []) or []
    if tech:
        lines.append("**Technical rationale (from the recommender):**")
        for t in tech:
            lines.append(f"- {t}")
    else:
        lines.append("**Technical rationale:** (none provided)")

    lines.append("")
    lines.append("**Key dataset signals used (if available):**")

    for k in [
        "problem_type",
        "n_rows",
        "n_cols",
        "target_dtype",
        "target_nunique",
        "missingness_overall",
        "class_balance",
        "n_numeric",
        "n_categorical",
    ]:
        if k in profile:
            lines.append(f"- `{k}`: {profile[k]}")

    return "\n".join(lines)


def looks_like_id(series):
    s = series.dropna()
    return len(s) > 0 and s.nunique() / len(s) >= 0.9


def is_datetime_like(series):
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    s = series.dropna().astype(str).head(200)
    parsed = pd.to_datetime(s, errors="coerce")
    return parsed.notna().mean() > 0.8


def dataset_guardrails(df, target_col, problem_type):
    warnings = []
    y = df[target_col]

    if looks_like_id(y):
        warnings.append("⚠️ Target looks like an ID column. Models usually can’t learn from IDs.")

    if is_datetime_like(y):
        warnings.append("⚠️ Target looks like a datetime column. Confirm this is intentional.")

    if df.shape[0] < 200:
        warnings.append("⚠️ Dataset is small. Results may vary significantly.")

    if problem_type == "classification":
        vc = y.value_counts(normalize=True)
        if len(vc) > 1 and vc.iloc[0] > 0.8:
            warnings.append(
                f"⚠️ Class imbalance detected (~{vc.iloc[0]*100:.0f}% in one class). "
                "Accuracy may be misleading."
            )

    return warnings


def metric_explainer(problem_type, metrics):
    lines = []
    if problem_type == "classification":
        lines += [
            "- **Accuracy**: Overall correctness (can mislead with imbalance).",
            "- **F1**: Balance of precision and recall.",
            "- **Precision**: How often positive predictions are correct.",
            "- **Recall**: How many positives the model captures.",
        ]
    elif problem_type == "regression":
        lines += [
            "- **RMSE**: Typical prediction error size (lower is better).",
            "- **MAE**: Average absolute error.",
            "- **R²**: Variance explained (closer to 1 is better).",
        ]
    return "\n".join(lines)


# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(page_title="ML Model Selector Advisor", page_icon="🧠", layout="wide")

# -------------------------------------------------
# Header
# -------------------------------------------------
st.markdown(
    """
    <h1>🧠 ML Model Selector Advisor</h1>
    <p>Upload a CSV, choose a target column, and get beginner-friendly ML model recommendations.</p>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------
# Upload CSV
# -------------------------------------------------
uploaded = st.file_uploader("Upload dataset (.csv)", type=["csv"])

if uploaded is None:
    st.info("Upload a CSV file to begin.")
    st.stop()

df = pd.read_csv(uploaded)

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    target_col = st.selectbox("Target column", ["(none)"] + list(df.columns))
    problem_hint = st.selectbox("Problem hint", ["Auto-detect", "Classification", "Regression"])
    run_baseline = st.toggle("Run baseline evaluation", value=True)
    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    random_state = st.number_input("Random state", value=42)

if target_col == "(none)":
    st.error("Select a target column to continue.")
    st.stop()

# -------------------------------------------------
# Dataset summary
# -------------------------------------------------
c1, c2, c3 = st.columns(3)
c1.metric("Rows", df.shape[0])
c2.metric("Columns", df.shape[1])
c3.metric("Missing cells", int(df.isna().sum().sum()))

# -------------------------------------------------
# Compute profile & recommendations
# -------------------------------------------------
with st.spinner("Analyzing dataset..."):
    profile = profile_dataset(df, target_col=target_col, problem_hint=problem_hint)
    rec = recommend_models(profile)

problem_type = str(profile.get("problem_type", "")).lower()
warnings = dataset_guardrails(df, target_col, problem_type)

# -------------------------------------------------
# Tabs
# -------------------------------------------------
tab1, tab2, tab3 = st.tabs(["✅ Recommendations", "📊 Dataset Profile", "🚀 Baseline"])

# ---------------- TAB 1 ----------------
with tab1:
    for w in warnings:
        st.warning(w)

    st.subheader("Recommended models")
    shortlist_df = pd.DataFrame(rec.get("shortlist", []))
    st.dataframe(shortlist_df, use_container_width=True, hide_index=True)

    st.subheader("Why these models were chosen (simple explanation)")
    st.info(problem_type_sentence(profile, df, target_col), icon="🎯")

    for r in dict.fromkeys(simplify_rationale(rec.get("rationale", []))):
        st.info(r, icon="💡")

    with st.expander("Advanced explanation (optional)"):
        st.markdown(format_advanced_block(profile, rec))

# ---------------- TAB 2 ----------------
with tab2:
    st.subheader("Dataset profile")
    st.json(profile)

# ---------------- TAB 3 ----------------
with tab3:
    st.subheader("Baseline evaluation")

    if run_baseline:
        try:
            results = build_and_eval_baseline(
                df=df,
                target_col=target_col,
                problem_type=profile.get("problem_type"),
                test_size=float(test_size),
                random_state=int(random_state),
            )

            st.metric("Model used", results.get("model_name"))
            metrics = results.get("metrics", {})
            st.dataframe(pd.DataFrame([metrics]), hide_index=True)

            with st.expander("What do these metrics mean?"):
                st.markdown(metric_explainer(problem_type, metrics))

        except Exception as e:
            st.error(f"Baseline failed: {e}")
    else:
        st.info("Baseline evaluation is disabled.")

st.caption("— ML Model Selector Advisor")
