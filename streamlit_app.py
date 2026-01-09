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
        # show up to 4 example labels
        examples = y.dropna().astype(str).unique()[:4]
        examples_str = ", ".join(map(str, examples)) if len(examples) else "—"
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
    keys_of_interest = [
        "problem_type",
        "n_rows",
        "n_cols",
        "target_dtype",
        "target_nunique",
        "missingness_overall",
        "class_balance",
        "n_numeric",
        "n_categorical",
    ]

    shown_any = False
    for k in keys_of_interest:
        if k in profile:
            lines.append(f"- `{k}`: {profile[k]}")
            shown_any = True

    if not shown_any:
        preview_keys = list(profile.keys())[:12]
        lines.append(
            f"- Available profile keys: {preview_keys} (schema depends on your profiling module)"
        )

    return "\n".join(lines)


def looks_like_id_column(series: pd.Series) -> bool:
    s = series.dropna()
    if len(s) == 0:
        return False
    # high uniqueness is a strong ID signal
    unique_ratio = s.nunique() / len(s)
    return unique_ratio >= 0.9


def safe_is_datetime(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    # try parse a sample
    s = series.dropna()
    if len(s) < 5:
        return False
    sample = s.astype(str).head(200)
    parsed = pd.to_datetime(sample, errors="coerce", utc=False)
    # if most parse -> likely datetime-ish
    return parsed.notna().mean() >= 0.8


def dataset_guardrails(df: pd.DataFrame, target_col: str, problem_type: str):
    warnings = []

    y = df[target_col]

    # Target: ID-like
    if looks_like_id_column(y):
        warnings.append(
            f"⚠️ **Target looks like an ID** (most values are unique). Models usually can’t learn useful patterns from IDs."
        )

    # Target: datetime-like
    if safe_is_datetime(y):
        warnings.append(
            "⚠️ **Target looks like a date/time**. That’s okay only if you’re truly predicting a timestamp; otherwise choose a different target."
        )

    # Too few rows
    if df.shape[0] < 200:
        warnings.append(
            "⚠️ **Small dataset** (<200 rows). Results may vary a lot with one train/test split. Consider cross-validation (future improvement)."
        )

    # Classification imbalance warning
    if str(problem_type).lower() == "classification":
        vc = y.dropna().value_counts(normalize=True)
        if len(vc) >= 2 and vc.iloc[0] >= 0.8:
            warnings.append(
                f"⚠️ **Class imbalance detected**: largest class is ~{vc.iloc[0]*100:.0f}%. Accuracy can be misleading; prefer F1/ROC-AUC (if available)."
            )

    # Leakage-ish column name hints (features)
    leak_words = ["target", "label", "outcome", "groundtruth", "gt", "result"]
    suspected = [
        c
        for c in df.columns
        if c != target_col and any(w in c.lower() for w in leak_words)
    ]
    if suspected:
        warnings.append(
            f"⚠️ **Potential leakage columns by name**: {suspected[:6]}{'…' if len(suspected) > 6 else ''}. If any of these are created *after* the outcome, drop them."
        )

    return warnings


def metric_explainer(problem_type: str, metrics: dict) -> str:
    pt = str(problem_type).lower()
    keys = set((metrics or {}).keys())

    lines = []
    if pt == "classification":
        lines.append("**Classification metrics (quick meaning):**")
        if "accuracy" in keys:
            lines.append("- **Accuracy**: % predictions correct (can be misleading if classes are imbalanced).")
        if "f1" in keys or "f1_score" in keys:
            lines.append("- **F1**: balances precision + recall (better for imbalance).")
        if "precision" in keys:
            lines.append("- **Precision**: when the model predicts positive, how often it's correct.")
        if "recall" in keys:
            lines.append("- **Recall**: how many true positives the model finds.")
        if "roc_auc" in keys or "auc" in keys:
            lines.append("- **ROC-AUC**: ranking quality across thresholds (higher is better).")
        if len(lines) == 1:
            lines.append("- These metrics summarize how well the model predicts categories.")
    elif pt == "regression":
        lines.append("**Regression metrics (quick meaning):**")
        if "rmse" in keys:
            lines.append("- **RMSE**: average error size (lower is better). Same units as the target.")
        if "mae" in keys:
            lines.append("- **MAE**: average absolute error (lower is better).")
        if "r2" in keys:
            lines.append("- **R²**: how much variance is explained (1.0 is best; 0 means no improvement over mean).")
        if len(lines) == 1:
            lines.append("- These metrics summarize how close predictions are to the numeric target.")
    else:
        lines.append("Metrics explanation depends on whether it’s classification or regression.")

    return "\n".join(lines)


def make_sample_dataset(kind: str) -> pd.DataFrame:
    rng = np.random.default_rng(42)

    if kind == "Classification (sample)":
        n = 600
        age = rng.integers(18, 70, size=n)
        city = rng.choice(["Denver", "Seattle", "Austin", "NYC"], size=n, p=[0.3, 0.2, 0.2, 0.3])
        plan = rng.choice(["Free", "Pro", "Enterprise"], size=n, p=[0.65, 0.3, 0.05])
        sessions = rng.poisson(lam=8, size=n)
        last_active_days = rng.integers(0, 60, size=n)

        # synthetic label with some signal + imbalance
        churn_prob = (
            0.15
            + 0.02 * (plan == "Free")
            + 0.03 * (last_active_days > 20)
            + 0.02 * (sessions < 5)
        )
        churn = rng.random(n) < churn_prob
        churn = np.where(churn, "Churned", "Active")  # labels

        df = pd.DataFrame(
            {
                "age": age,
                "city": city,
                "plan": plan,
                "sessions_last_30d": sessions,
                "days_since_last_active": last_active_days,
                "account_id": [f"A{100000+i}" for i in range(n)],
                "status": churn,  # target
            }
        )
        # add a few missing values
        mask = rng.random(n) < 0.05
        df.loc[mask, "city"] = np.nan
        return df

    # Regression sample
    n = 600
    sqft = rng.normal(1800, 550, size=n).clip(400, 4500).round(0)
    beds = rng.integers(1, 6, size=n)
    neighborhood = rng.choice(["A", "B", "C", "D"], size=n, p=[0.25, 0.35, 0.25, 0.15])
    year_built = rng.integers(1950, 2023, size=n)

    base = 120_000 + sqft * 180 + beds * 15_000
    neigh_adj = np.select(
        [neighborhood == "A", neighborhood == "B", neighborhood == "C", neighborhood == "D"],
        [90_000, 40_000, 10_000, -15_000],
        default=0,
    )
    age_penalty = (2025 - year_built) * 450  # older houses slightly cheaper
    noise = rng.normal(0, 35_000, size=n)
    price = (base + neigh_adj - age_penalty + noise).clip(50_000, None).round(0)

    df = pd.DataFrame(
        {
            "sqft": sqft,
            "beds": beds,
            "neighborhood": neighborhood,
            "year_built": year_built,
            "listing_id": [f"L{900000+i}" for i in range(n)],
            "price": price,  # target
        }
    )
    # add a few missing values
    mask = rng.random(n) < 0.04
    df.loc[mask, "year_built"] = np.nan
    return df


def df_to_csv_download(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# -------------------------------------------------
# Page config + styles
# -------------------------------------------------
st.set_page_config(page_title="ML Model Selector Advisor", page_icon="🧠", layout="wide")

st.markdown(
    """
    <style>
    h1 { font-size: 1.6rem; }
    h2 { font-size: 1.25rem; }
    h3 { font-size: 1.05rem; }
    .hero {
        padding: 1rem;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.15);
        background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(16,185,129,0.1));
    }
    .card {
        padding: 1rem;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.15);
        background: rgba(255,255,255,0.03);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------
# Header
# -------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🧠 ML Model Selector Advisor</h1>
        <p>Upload a CSV (or use a sample), select a target, and get beginner-friendly model recommendations + a runnable baseline.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

# -------------------------------------------------
# Load data: sample or upload
# -------------------------------------------------
choice_left, choice_right = st.columns([1.2, 1])

with choice_left:
    data_mode = st.radio(
        "Start with:",
        ["Use a sample dataset", "Upload my CSV"],
        horizontal=True,
    )

with choice_right:
    st.info("Beginners tip: Try the sample first to understand the flow.", icon="✨")

df = None
uploaded = None

if data_mode == "Use a sample dataset":
    sample_kind = st.selectbox(
        "Choose a sample dataset",
        ["Classification (sample)", "Regression (sample)"],
    )
    df = make_sample_dataset(sample_kind)

    st.download_button(
        "Download this sample CSV",
        data=df_to_csv_download(df),
        file_name=f"sample_{'classification' if 'Class' in sample_kind else 'regression'}.csv",
        mime="text/csv",
        use_container_width=True,
    )
else:
    uploaded = st.file_uploader("Upload dataset (.csv)", type=["csv"])
    if uploaded is None:
        st.warning("Upload a CSV to begin.")
        st.stop()
    df = pd.read_csv(uploaded)

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    target_col = st.selectbox("Target column", ["(none)"] + list(df.columns))
    problem_hint = st.selectbox("Problem hint", ["Auto-detect", "Classification", "Regression"])

    st.divider()
    run_baseline = st.toggle("Run baseline evaluation", value=True)
    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    random_state = st.number_input("Random state", value=42)

if target_col == "(none)":
    st.error("Select a target column to continue.")
    st.stop()

# -------------------------------------------------
# Dataset summary (no preview)
# -------------------------------------------------
m1, m2, m3 = st.columns(3)
m1.metric("Rows", df.shape[0])
m2.metric("Columns", df.shape[1])
m3.metric("Missing cells", int(df.isna().sum().sum()))
st.write("")

# -------------------------------------------------
# Compute
# -------------------------------------------------
with st.spinner("Analyzing dataset..."):
    profile = profile_dataset(df, target_col=target_col, problem_hint=problem_hint)
    rec = recommend_models(profile)

problem_type = str(profile.get("problem_type", "")).lower()

# Guardrails / warnings
guardrail_msgs = dataset_guardrails(df, target_col, problem_type)

# -------------------------------------------------
# Tabs
# -------------------------------------------------
tab1, tab2, tab3 = st.tabs(["✅ Recommendations", "📊 Dataset Profile", "🚀 Baseline"])

# ---------------- TAB 1 ----------------
with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Recommended models")

    if guardrail_msgs:
        for msg in guardrail_msgs:
            st.warning(msg)

    shortlist_df = pd.DataFrame(rec.get("shortlist", []))
    if shortlist_df.empty:
        st.info("No shortlist returned by the recommender. Check your recommend_models() output schema.")
    else:
        st.dataframe(shortlist_df, use_container_width=True, hide_index=True)

        # Export shortlist
        st.download_button(
            "Download recommendations (CSV)",
            data=df_to_csv_download(shortlist_df),
            file_name="model_recommendations.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.write("")
    st.subheader("Why these models were chosen (simple explanation)")
    st.info(problem_type_sentence(profile, df, target_col), icon="🎯")

    reasons = simplify_rationale(rec.get("rationale", []))
    for r in dict.fromkeys(reasons):
        st.info(r, icon="💡")

    with st.expander("Advanced explanation (optional)"):
        st.markdown(format_advanced_block(profile, rec))

    st.caption("Note: Recommendations are starting points. Always validate with domain knowledge and proper evaluation.")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- TAB 2 ----------------
with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Dataset profile")

    st.json(profile)

    st.download_button(
        "Download profile (JSON)",
        data=json.dumps(profile, indent=2).encode("utf-8"),
        file_name="dataset_profile.json",
        mime="application/json",
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- TAB 3 ----------------
with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Baseline evaluation")

    if run_baseline:
        try:
            with st.spinner("Training baseline model..."):
                results = build_and_eval_baseline(
                    df=df,
                    target_col=target_col,
                    problem_type=profile.get("problem_type"),
                    test_size=float(test_size),
                    random_state=int(random_state),
                )

            model_name = results.get("model_name", "—")
            metrics = results.get("metrics", {}) or {}
            notes = results.get("notes", []) or []

            c1, c2 = st.columns([1, 1.2])
            with c1:
                st.metric("Model used", model_name)

            with c2:
                # Add a quick “target scale” hint for regression
                if problem_type == "regression" and pd.api.types.is_numeric_dtype(df[target_col]):
                    y = df[target_col].dropna()
                    if len(y) > 0:
                        st.caption(
                            f"Target scale hint: mean={y.mean():.2f}, std={y.std():.2f} (helps interpret RMSE/MAE)."
                        )

            if metrics:
                st.dataframe(pd.DataFrame([metrics]), hide_index=True, use_container_width=True)
            else:
                st.info("No metrics returned by build_and_eval_baseline().")

            with st.expander("What do these metrics mean?"):
                st.markdown(metric_explainer(problem_type, metrics))

            if notes:
                st.write("**Notes:**")
                for n in notes:
                    st.write("- ", n)

            # Export results
            st.download_button(
                "Download baseline results (JSON)",
                data=json.dumps(results, indent=2, default=str).encode("utf-8"),
                file_name="baseline_results.json",
                mime="application/json",
                use_container_width=True,
            )

            with st.expander("Baseline transparency (what usually happens)"):
                st.markdown(
                    """
- Split into train/test (with stratification for classification when possible).
- Preprocess features:
  - Missing values handled (imputed).
  - Categorical columns encoded (e.g., one-hot).
  - Numeric columns optionally scaled (depends on your baseline implementation).
- Train a simple, reliable starter model and report quick metrics.

If you want maximum trust, have your baseline code return a `preprocessing` field describing exact steps.
                    """.strip()
                )

        except Exception as e:
            st.error(f"Baseline failed: {e}")
    else:
        st.info("Baseline evaluation is disabled.")

    st.markdown("</div>", unsafe_allow_html=True)

st.caption("— ML Model Selector Advisor")
