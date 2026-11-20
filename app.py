from __future__ import annotations

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.base import clone

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

# ============================================================
# 1. PAGE CONFIGURATION AND PATHS
# ============================================================

st.set_page_config(
    page_title="EduPredict | UCI Student Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_CANDIDATES = [
    DATA_DIR / "student-mat.csv",
    DATA_DIR / "student_mat.csv",
    DATA_DIR / "Original_Dataset.csv",
    BASE_DIR / "student-mat.csv",
]

TARGET_COLUMN = "G3"
TARGET_LABELS = ["At-Risk", "Pass"]
RANDOM_STATE = 42

# UCI Student Performance Mathematics predictors.
FEATURES = [
    "school", "sex", "age", "address", "famsize", "Pstatus",
    "Medu", "Fedu", "Mjob", "Fjob", "reason", "guardian",
    "traveltime", "studytime", "failures", "schoolsup", "famsup",
    "paid", "activities", "nursery", "higher", "internet", "romantic",
    "famrel", "freetime", "goout", "Dalc", "Walc", "health",
    "absences", "G1", "G2",
]

CATEGORICAL_FEATURES = [
    "school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob",
    "reason", "guardian", "schoolsup", "famsup", "paid", "activities",
    "nursery", "higher", "internet", "romantic",
]
NUMERIC_FEATURES = [c for c in FEATURES if c not in CATEGORICAL_FEATURES]

# ============================================================
# 2. PROFESSIONAL WHITE UI
# ============================================================

st.markdown(
    """
    <style>
    :root {
        --navy: #17365D;
        --blue: #2563EB;
        --blue-dark: #1D4ED8;
        --text: #24364B;
        --muted: #65758B;
        --border: #E3EAF3;
        --soft: #F5F8FC;
    }
    .stApp, [data-testid="stAppViewContainer"] {background:#FFFFFF;color:var(--text);}
    .block-container {max-width:1450px;padding:1.5rem 2.2rem 3rem;}
    [data-testid="stHeader"] {background:rgba(255,255,255,.94);}
    [data-testid="stSidebar"] {background:#F7F9FC;border-right:1px solid var(--border);}
    [data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label {color:var(--navy)!important;}
    h1,h2,h3,h4 {color:var(--navy)!important;font-weight:750!important;letter-spacing:-.02em;}
    p,li,[data-testid="stCaptionContainer"] {color:var(--muted);}
    .hero {
      display:flex;align-items:center;justify-content:space-between;gap:24px;
      background:linear-gradient(115deg,#F0F6FF 0%,#F8FBFF 62%,#EAF2FF 100%);
      border:1px solid #DCE7F5;border-radius:22px;padding:34px 38px;margin:4px 0 26px;
      box-shadow:0 8px 28px rgba(27,69,125,.055);
    }
    .hero-eyebrow {color:#2563EB;font-size:11px;font-weight:800;letter-spacing:1.7px;margin-bottom:12px;}
    .hero-title {color:#17365D;font-size:clamp(34px,4vw,46px);line-height:1.12;font-weight:850;margin:0 0 8px;}
    .hero-subtitle {color:#38536F;font-size:18px;font-weight:650;margin-bottom:8px;}
    .hero-description {color:#64748B;font-size:14px;line-height:1.7;max-width:760px;}
    .hero-mark {width:88px;height:88px;flex:0 0 88px;display:grid;place-items:center;border:1px solid #D6E4F8;
      border-radius:22px;background:rgba(255,255,255,.8);font-size:40px;box-shadow:0 7px 18px rgba(27,69,125,.06);}
    .section-kicker {color:#2563EB;font-size:11px;font-weight:800;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:5px;}
    .section-description {color:#64748B;font-size:14px;margin-bottom:18px;}
    [data-testid="stMetric"] {background:#fff;border:1px solid var(--border);border-radius:15px;padding:16px 18px;
      box-shadow:0 3px 12px rgba(15,42,75,.035);}
    [data-testid="stMetricLabel"] {color:#64748B!important;}
    [data-testid="stMetricValue"] {color:#17365D!important;font-weight:750!important;}
    .stTabs [data-baseweb="tab-list"] {gap:6px;background:#F5F8FC;border:1px solid var(--border);padding:6px;border-radius:13px;}
    .stTabs [data-baseweb="tab"] {color:#526780;border-radius:9px;font-weight:650;padding:9px 13px;}
    .stTabs [aria-selected="true"] {background:#E5EFFF!important;color:#174A8B!important;}
    .stButton>button,.stDownloadButton>button {background:#2459A6;color:#fff!important;border:1px solid #2459A6;
      border-radius:9px;min-height:42px;font-weight:650;}
    .stButton>button:hover,.stDownloadButton>button:hover {background:#174580;border-color:#174580;color:#fff!important;}
    [data-testid="stDataFrame"] {border:1px solid var(--border);border-radius:10px;}
    [data-testid="stAlert"] {border-radius:11px;}
    hr {border-color:var(--border);}
    .welcome-panel {background:#F7FAFF;border:1px solid #E0EAF8;border-radius:18px;padding:24px 26px;margin:12px 0 20px;}
    .welcome-title {color:#17365D;font-size:24px;font-weight:750;margin-bottom:7px;}
    .welcome-text {color:#64748B;font-size:14px;line-height:1.7;}
    .workflow-card {background:#fff;border:1px solid var(--border);border-radius:14px;padding:18px;min-height:140px;
      box-shadow:0 3px 12px rgba(15,42,75,.025);}
    .workflow-number {color:#2563EB;font-size:11px;font-weight:800;letter-spacing:1px;margin-bottom:8px;}
    .workflow-title {color:#17365D;font-size:16px;font-weight:700;margin-bottom:5px;}
    .workflow-description {color:#64748B;font-size:13px;line-height:1.55;}
    .footer {text-align:center;color:#8492A6;font-size:12px;line-height:1.8;padding:20px 0 4px;}
    @media(max-width:800px){.block-container{padding-left:1rem;padding-right:1rem}.hero{padding:24px}.hero-mark{display:none;}}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# 3. DATASET LOADING AND VALIDATION
# ============================================================

@st.cache_data(show_spinner=False)
def read_uci_dataset(path_string: str, mtime: float) -> pd.DataFrame:
    """Read the standard UCI Mathematics CSV (normally semicolon-delimited)."""
    path = Path(path_string)
    df = pd.read_csv(path, sep=";")
    if len(df.columns) == 1:
        df = pd.read_csv(path, sep=",")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def find_dataset_path() -> Path | None:
    for candidate in DATA_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def validate_dataset(df: pd.DataFrame) -> list[str]:
    required = set(FEATURES + [TARGET_COLUMN])
    return sorted(required - set(df.columns))


def prepare_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    for column in NUMERIC_FEATURES + [TARGET_COLUMN]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    for column in CATEGORICAL_FEATURES:
        # Keep categorical columns as ordinary object dtype so sklearn's
        # SimpleImputer handles missing values consistently across versions.
        df[column] = df[column].astype("object")
        df[column] = df[column].apply(
            lambda value: np.nan if pd.isna(value) else str(value).strip()
        )

    df = df.dropna(subset=[TARGET_COLUMN]).copy()
    # UCI convention: G3 >= 10 is Pass; G3 < 10 is At-Risk.
    df["Outcome"] = np.where(df[TARGET_COLUMN] >= 10, "Pass", "At-Risk")
    return df

# ============================================================
# 4. PREPROCESSING AND MODEL TRAINING
# ============================================================

def make_pipeline(X: pd.DataFrame, estimator) -> Pipeline:
    """Build preprocessing only for the features selected in this experiment."""
    categorical_columns = [c for c in X.columns if c in CATEGORICAL_FEATURES]
    numeric_columns = [c for c in X.columns if c not in CATEGORICAL_FEATURES]

    transformers = []
    if numeric_columns:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", numeric_pipeline, numeric_columns))
    if categorical_columns:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("cat", categorical_pipeline, categorical_columns))

    preprocessor = ColumnTransformer(transformers, remainder="drop")
    return Pipeline([
        ("preprocess", preprocessor),
        ("model", estimator),
    ])


def train_models(df: pd.DataFrame, test_size: float, features: list[str]) -> dict:
    features = list(features)
    if not features:
        raise ValueError("At least one predictor feature must be selected.")
    X = df[features].copy()
    y = df["Outcome"].astype(str)
    counts = y.value_counts()
    if len(counts) != 2 or counts.min() < 5:
        raise ValueError("Both Pass and At-Risk classes need at least five records.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    estimators = {
        "Logistic Regression": LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=3, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
    }

    min_train_class = int(y_train.value_counts().min())
    n_splits = min(5, min_train_class)
    if n_splits < 2:
        raise ValueError("Not enough training records per class for stratified cross-validation.")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    fitted, predictions, metric_rows, cv_rows = {}, {}, [], []
    for name, estimator in estimators.items():
        pipe = make_pipeline(X_train, clone(estimator))
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        fitted[name] = pipe
        predictions[name] = pred

        # Treat At-Risk as the positive class for early-intervention metrics.
        class_order = list(pipe.named_steps["model"].classes_)
        risk_probability = pipe.predict_proba(X_test)[:, class_order.index("At-Risk")]
        y_test_risk = (y_test == "At-Risk").astype(int)

        metric_rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Balanced Accuracy": balanced_accuracy_score(y_test, pred),
            "At-Risk Precision": precision_score(y_test, pred, pos_label="At-Risk", zero_division=0),
            "At-Risk Recall": recall_score(y_test, pred, pos_label="At-Risk", zero_division=0),
            "At-Risk F1": f1_score(y_test, pred, pos_label="At-Risk", zero_division=0),
            "Pass Precision": precision_score(y_test, pred, pos_label="Pass", zero_division=0),
            "Pass Recall": recall_score(y_test, pred, pos_label="Pass", zero_division=0),
            "Pass F1": f1_score(y_test, pred, pos_label="Pass", zero_division=0),
            "Macro F1": f1_score(y_test, pred, average="macro", zero_division=0),
            "ROC-AUC (At-Risk)": roc_auc_score(y_test_risk, risk_probability),
            "PR-AUC (At-Risk)": average_precision_score(y_test_risk, risk_probability),
            "Matthews Correlation": matthews_corrcoef(y_test, pred),
            "Brier Score (At-Risk)": brier_score_loss(y_test_risk, risk_probability),
        })

        cv_pipe = make_pipeline(X_train, clone(estimator))
        at_risk_recall_scorer = make_scorer(
            recall_score, pos_label="At-Risk", zero_division=0
        )
        scores = cross_validate(
            cv_pipe, X_train, y_train, cv=cv,
            scoring={
                "accuracy": "accuracy",
                "balanced_accuracy": "balanced_accuracy",
                "macro_f1": "f1_macro",
                "at_risk_recall": at_risk_recall_scorer,
            },
            n_jobs=-1,
        )
        cv_rows.append({
            "Model": name,
            "CV Accuracy Mean": scores["test_accuracy"].mean(),
            "CV Accuracy Std": scores["test_accuracy"].std(ddof=1),
            "CV Balanced Accuracy Mean": scores["test_balanced_accuracy"].mean(),
            "CV Macro F1 Mean": scores["test_macro_f1"].mean(),
            "CV Macro F1 Std": scores["test_macro_f1"].std(ddof=1),
            "CV At-Risk Recall Mean": scores["test_at_risk_recall"].mean(),
            "CV At-Risk Recall Std": scores["test_at_risk_recall"].std(ddof=1),
        })

    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "fitted": fitted, "predictions": predictions,
        "metrics": pd.DataFrame(metric_rows),
        "cv": pd.DataFrame(cv_rows), "n_splits": n_splits,
        "data": df, "features": features,
    }

# ============================================================
# 5. SHAP UTILITIES
# ============================================================

def to_dense(matrix):
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


def class_shap_values(shap_output, class_index: int, n_features: int) -> np.ndarray:
    if isinstance(shap_output, list):
        return np.asarray(shap_output[class_index])
    values = np.asarray(getattr(shap_output, "values", shap_output))
    if values.ndim == 2:
        return values
    if values.ndim == 3:
        # Typical formats: samples x features x classes OR classes x samples x features.
        if values.shape[1] == n_features:
            return values[:, :, class_index]
        if values.shape[2] == n_features:
            return values[class_index, :, :]
    raise ValueError(f"Unsupported SHAP output shape: {values.shape}")


def class_base_value(explainer, class_index: int, n_classes: int) -> float:
    expected = np.asarray(explainer.expected_value)
    if expected.ndim == 0:
        return float(expected)
    if expected.size == n_classes:
        return float(expected.reshape(-1)[class_index])
    raise ValueError(f"Unsupported SHAP expected_value shape: {expected.shape}")


def calculate_global_shap(rf_pipe: Pipeline, sample: pd.DataFrame) -> pd.DataFrame:
    import shap
    preprocessor = rf_pipe.named_steps["preprocess"]
    model = rf_pipe.named_steps["model"]
    transformed = to_dense(preprocessor.transform(sample))
    names = preprocessor.get_feature_names_out()
    explainer = shap.TreeExplainer(model)
    output = explainer.shap_values(transformed, check_additivity=False)
    class_index = list(model.classes_).index("Pass")
    values = class_shap_values(output, class_index, len(names))
    importance = np.abs(values).mean(axis=0)
    return pd.DataFrame({
        "Feature": names,
        "Mean Absolute SHAP (Pass)": importance,
    }).sort_values("Mean Absolute SHAP (Pass)", ascending=False)


def calculate_individual_shap(rf_pipe: Pipeline, row: pd.DataFrame):
    import shap
    preprocessor = rf_pipe.named_steps["preprocess"]
    model = rf_pipe.named_steps["model"]
    transformed = to_dense(preprocessor.transform(row))
    names = preprocessor.get_feature_names_out()
    predicted_class = rf_pipe.predict(row)[0]
    classes = list(model.classes_)
    class_index = classes.index(predicted_class)
    explainer = shap.TreeExplainer(model)
    output = explainer.shap_values(transformed, check_additivity=False)
    values = class_shap_values(output, class_index, len(names))[0]
    base = class_base_value(explainer, class_index, len(classes))
    explanation = shap.Explanation(
        values=np.asarray(values).reshape(-1),
        base_values=base,
        data=transformed[0],
        feature_names=names,
    )
    return explanation, predicted_class

# ============================================================
# 6. HERO HEADER
# ============================================================

st.markdown(
    '<div class="hero"><div><div class="hero-eyebrow">STUDENT PERFORMANCE ANALYTICS · UCI RESEARCH EDITION</div><div class="hero-title">EduPredict</div><div class="hero-subtitle">Understand. Predict. Improve.</div><div class="hero-description">A machine-learning platform for predicting student academic outcomes using educational, family-background, social, and study-related factors from the UCI Student Performance Mathematics dataset.</div></div><div class="hero-mark">🎓</div></div>',
    unsafe_allow_html=True,
)

# ============================================================
# 7. SIDEBAR SETTINGS
# ============================================================

with st.sidebar:
    st.markdown("### EduPredict")
    st.caption("UCI Student Performance · Mathematics")
    st.divider()
    st.markdown("#### Experiment settings")
    prediction_stage = st.radio(
        "Prediction stage",
        ["Early-stage (without G1/G2)", "Later-stage (with G1/G2)"],
        index=1,
        help="Early-stage uses the 30 background and behaviour predictors. Later-stage additionally uses first- and second-period grades G1 and G2.",
    )
    active_features = [
        feature for feature in FEATURES
        if prediction_stage.startswith("Later") or feature not in {"G1", "G2"}
    ]
    test_fraction = st.slider("Test dataset proportion", 0.20, 0.35, 0.25, 0.05)
    run_button = st.button("Run ML Experiment", use_container_width=True, type="primary")
    st.divider()
    st.markdown("#### Prediction target")
    st.write("**Pass:** G3 ≥ 10")
    st.write("**At-Risk:** G3 < 10")
    st.caption("G3 is the target and is never used as a predictor.")
    if prediction_stage.startswith("Early"):
        st.success("Early-stage mode: G1 and G2 are excluded from model inputs.")
    else:
        st.warning("Later-stage mode: G1 and G2 are included. This is not a prediction made before earlier-period grades exist.")
    with st.expander("About the dataset"):
        st.write("UCI Student Performance — Mathematics (student-mat.csv).")
        st.write("Includes family education, parent occupations, family support, study habits, absences, and earlier grades G1/G2.")

# ============================================================
# 8. DATASET DISCOVERY AND STATUS
# ============================================================

st.markdown('<div class="section-kicker">WORKSPACE</div>', unsafe_allow_html=True)
st.markdown("## Research dashboard")
st.markdown('<div class="section-description">UCI Mathematics dataset · family background · academic history · student prediction · explainable AI.</div>', unsafe_allow_html=True)

dataset_path = find_dataset_path()
if dataset_path is None:
    st.error("UCI Mathematics dataset not found.")
    st.markdown(
        '<div class="welcome-panel"><div class="welcome-title">Add your UCI dataset</div><div class="welcome-text">Place the Mathematics dataset file <b>student-mat.csv</b> inside the <b>data</b> folder beside app.py. The standard UCI file is semicolon-delimited. You may also use data/Original_Dataset.csv.</div></div>',
        unsafe_allow_html=True,
    )
    st.code("EduPredict/\n├── app.py\n└── data/\n    └── student-mat.csv", language="text")
    st.stop()

try:
    raw_data = read_uci_dataset(str(dataset_path), dataset_path.stat().st_mtime)
except Exception as error:
    st.error(f"Could not read the UCI dataset: {error}")
    st.stop()

missing = validate_dataset(raw_data)
if missing:
    st.error("The selected file is missing required UCI columns: " + ", ".join(missing))
    st.info("Use the UCI Student Performance Mathematics file (student-mat.csv), not the Portuguese file or OULAD files.")
    st.stop()

try:
    data = prepare_dataset(raw_data)
except Exception as error:
    st.error(f"Dataset preparation failed: {error}")
    st.stop()

status_col1, status_col2, status_col3 = st.columns(3)
status_col1.metric("Dataset", "UCI Mathematics")
status_col2.metric("Records", f"{len(data):,}")
status_col3.metric("Predictor inputs", len(active_features))

with st.expander("Dataset file details", expanded=False):
    st.write(f"**File:** `{dataset_path.name}`")
    st.write(f"**Rows:** {len(raw_data):,}")
    st.write(f"**Columns:** {len(raw_data.columns)}")
    st.write("**Target:** G3 converted to Pass / At-Risk using the threshold of 10.")

# ============================================================
# 9. RUN THE EXPERIMENT
# ============================================================

signature = (str(dataset_path), dataset_path.stat().st_mtime, test_fraction, prediction_stage)
if run_button:
    try:
        with st.spinner("Training models and evaluating the held-out test dataset..."):
            result = train_models(data, test_fraction, active_features)
        st.session_state["uci_result"] = result
        st.session_state["uci_signature"] = signature
        for key in ["uci_global_shap", "uci_permutation", "uci_new_prediction", "uci_new_student_shap", "uci_test_shap"]:
            st.session_state.pop(key, None)
        st.success("UCI experiment completed successfully.")
    except Exception as error:
        st.error("Experiment failed. Review the error below.")
        st.exception(error)
        st.stop()

if "uci_result" not in st.session_state:
    st.markdown(
        '<div class="welcome-panel"><div class="welcome-title">Welcome to EduPredict</div><div class="welcome-text">Explore how family background, parental education, student support, study habits, attendance, and earlier grades relate to student outcomes. Run the experiment from the sidebar to train and evaluate the models.</div></div>',
        unsafe_allow_html=True,
    )
    a, b, c = st.columns(3)
    with a:
        st.markdown('<div class="workflow-card"><div class="workflow-number">01 · PREPARE</div><div class="workflow-title">Student & family profile</div><div class="workflow-description">Use parental education, occupations, family size, family support, and student background.</div></div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="workflow-card"><div class="workflow-number">02 · PREDICT</div><div class="workflow-title">Machine learning</div><div class="workflow-description">Compare Logistic Regression, Decision Tree, and Random Forest.</div></div>', unsafe_allow_html=True)
    with c:
        st.markdown('<div class="workflow-card"><div class="workflow-number">03 · EXPLAIN</div><div class="workflow-title">Explainable AI</div><div class="workflow-description">Inspect global feature importance and individual SHAP explanations.</div></div>', unsafe_allow_html=True)
    st.markdown("### UCI dataset preview")
    st.dataframe(data[active_features + [TARGET_COLUMN, "Outcome"]].head(10), use_container_width=True, hide_index=True)
    st.info("Click **Run ML Experiment** in the sidebar to train the models. You can use the New Student Prediction tab after the experiment finishes.")
    st.stop()

result = st.session_state["uci_result"]
if st.session_state.get("uci_signature") != signature:
    st.warning("The dataset or test proportion changed. Run the experiment again to refresh all results.")
    st.stop()

# ============================================================
# 10. NAVIGATION TABS
# ============================================================

tabs = st.tabs([
    "Overview",
    "Model Lab",
    "Explainable AI",
    "Student Insight",
    "New Student Prediction",
    "Research Export",
])

# ============================================================
# TAB 1 — OVERVIEW
# ============================================================

with tabs[0]:
    st.markdown("## Student Performance Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Students", f"{len(data):,}")
    c2.metric("Predictor features", len(result["features"]))
    c3.metric("Training records", f"{len(result['X_train']):,}")
    c4.metric("Test records", f"{len(result['X_test']):,}")

    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("### Dataset preview")
        st.dataframe(data[result["features"] + ["G3", "Outcome"]].head(12), use_container_width=True, hide_index=True)
    with right:
        st.markdown("### Outcome distribution")
        counts = data["Outcome"].value_counts().rename_axis("Outcome").reset_index(name="Students")
        fig = px.pie(counts, names="Outcome", values="Students", hole=0.58, color="Outcome", color_discrete_map={"Pass": "#2563EB", "At-Risk": "#F59E0B"}, template="plotly_white")
        fig.update_layout(paper_bgcolor="white", font_color="#17365D", margin=dict(l=10, r=10, t=20, b=10), legend_title_text="Final outcome")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("### Family and student background available in UCI")
    info_cols = st.columns(4)
    info_cols[0].metric("Mother's education", "Medu")
    info_cols[1].metric("Father's education", "Fedu")
    info_cols[2].metric("Family support", "famsup")
    info_cols[3].metric("Parent occupation", "Mjob / Fjob")
    st.caption("These are real predictors in the UCI Mathematics dataset. The model can therefore use them after training.")

# ============================================================
# TAB 2 — MODEL LAB
# ============================================================

with tabs[1]:
    st.markdown("## Machine Learning Model Evaluation")
    st.caption("Final performance metrics are calculated on the held-out test dataset. Cross-validation uses training data only.")

    metrics = result["metrics"]
    st.markdown("### Held-out test performance")
    st.dataframe(metrics.style.format({c: "{:.4f}" for c in metrics.columns if c != "Model"}), use_container_width=True, hide_index=True)
    st.caption("At-Risk is treated as the positive class for At-Risk Recall, ROC-AUC, and PR-AUC. Higher recall means fewer at-risk students are missed. Lower Brier Score is better; it evaluates probability error, not class accuracy.")

    plot_df = metrics.melt(id_vars="Model", var_name="Metric", value_name="Score")
    fig = px.bar(plot_df, x="Model", y="Score", color="Metric", barmode="group", range_y=[0, 1], text_auto=".3f", template="plotly_white")
    fig.update_layout(paper_bgcolor="white", plot_bgcolor="white", font_color="#17365D")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("### Stratified cross-validation")
    st.caption(f"{result['n_splits']}-fold stratified cross-validation on training data only. At-Risk Recall is included to measure risk detection.")
    st.dataframe(result["cv"].style.format({c: "{:.4f}" for c in result["cv"].columns if c != "Model"}), use_container_width=True, hide_index=True)

    st.divider()
    selected_eval_model = st.selectbox("Model for confusion matrix and classification report", list(result["fitted"]), key="uci_eval_model")
    labels = ["At-Risk", "Pass"]
    pred = result["predictions"][selected_eval_model]
    cm = confusion_matrix(result["y_test"], pred, labels=labels)
    st.markdown("### Confusion matrix")
    fig = px.imshow(cm, x=labels, y=labels, text_auto=True, color_continuous_scale="Blues", labels={"x": "Predicted", "y": "Actual", "color": "Students"}, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Classification report")
    report = classification_report(result["y_test"], pred, labels=labels, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T, use_container_width=True)

    st.download_button("Download test metrics", metrics.to_csv(index=False).encode("utf-8"), "uci_test_metrics.csv", "text/csv", key="uci_model_test_metrics_download")
    st.download_button("Download cross-validation metrics", result["cv"].to_csv(index=False).encode("utf-8"), "uci_cv_metrics.csv", "text/csv", key="uci_model_cv_metrics_download")

# ============================================================
# TAB 3 — EXPLAINABLE AI
# ============================================================

with tabs[2]:
    st.markdown("## Explainable Artificial Intelligence")
    st.caption("Understand which family, academic, and behavioural factors influence model predictions.")

    st.markdown("### Permutation feature importance")
    perm_model = st.selectbox("Select model", list(result["fitted"]), key="uci_perm_model")
    if st.button("Calculate permutation importance", key="uci_perm_button"):
        try:
            with st.spinner("Calculating permutation importance on held-out test data..."):
                pi = permutation_importance(result["fitted"][perm_model], result["X_test"], result["y_test"], n_repeats=10, random_state=RANDOM_STATE, scoring="f1_macro", n_jobs=-1)
                imp = pd.DataFrame({"Feature": result["features"], "Importance Mean": pi.importances_mean, "Importance Std": pi.importances_std}).sort_values("Importance Mean", ascending=False)
                st.session_state["uci_permutation"] = imp
                st.session_state["uci_permutation_model"] = perm_model
            st.success("Permutation importance calculated.")
        except Exception as error:
            st.error("Permutation importance failed.")
            st.exception(error)

    if "uci_permutation" in st.session_state:
        imp = st.session_state["uci_permutation"]
        st.caption(f"Model: {st.session_state.get('uci_permutation_model', 'Selected model')} · Scoring: macro F1 · Held-out test data")
        fig = px.bar(imp.sort_values("Importance Mean"), x="Importance Mean", y="Feature", orientation="h", error_x="Importance Std", template="plotly_white")
        fig.update_layout(paper_bgcolor="white", plot_bgcolor="white", font_color="#17365D")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(imp, use_container_width=True, hide_index=True)
        st.download_button("Download permutation importance", imp.to_csv(index=False).encode("utf-8"), "uci_permutation_importance.csv", "text/csv", key="uci_perm_download")

    st.divider()
    st.markdown("### Global SHAP feature importance")
    st.info("Global SHAP uses the Random Forest model and a reproducible sample of held-out test students. It explains the Pass-class output.")
    if st.button("Calculate global SHAP importance", key="uci_global_shap_button"):
        try:
            with st.spinner("Calculating SHAP values..."):
                sample = result["X_test"].sample(min(120, len(result["X_test"])), random_state=RANDOM_STATE)
                shap_df = calculate_global_shap(result["fitted"]["Random Forest"], sample)
                st.session_state["uci_global_shap"] = shap_df
            st.success("Global SHAP calculated successfully.")
        except Exception as error:
            st.error("Global SHAP calculation failed.")
            st.exception(error)

    if "uci_global_shap" in st.session_state:
        shap_df = st.session_state["uci_global_shap"]
        st.markdown("#### Top 20 SHAP features")
        fig = px.bar(shap_df.head(20).sort_values("Mean Absolute SHAP (Pass)"), x="Mean Absolute SHAP (Pass)", y="Feature", orientation="h", template="plotly_white")
        fig.update_layout(paper_bgcolor="white", plot_bgcolor="white", font_color="#17365D")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(shap_df, use_container_width=True, hide_index=True)
        st.download_button("Download global SHAP CSV", shap_df.to_csv(index=False).encode("utf-8"), "uci_shap_global.csv", "text/csv", key="uci_global_shap_download")
        st.caption("Mean absolute SHAP measures average influence magnitude. It does not indicate whether a feature raises or lowers the Pass-class output.")

# ============================================================
# TAB 4 — TEST STUDENT INSIGHT
# ============================================================

with tabs[3]:
    st.markdown("## Held-out Test Student Insight")
    st.caption("Choose an existing test-set student to compare predicted and actual outcomes.")
    insight_model = st.selectbox("Prediction model", list(result["fitted"]), key="uci_insight_model")
    test_indices = list(result["X_test"].index)
    idx = st.selectbox("Select test student record", test_indices, format_func=lambda x: f"Test record {x}", key="uci_test_record")
    row = result["X_test"].loc[[idx]]
    pred = result["fitted"][insight_model].predict(row)[0]
    actual = result["y_test"].loc[idx]
    c1, c2, c3 = st.columns(3)
    c1.metric("Selected model", insight_model)
    c2.metric("Predicted outcome", pred)
    c3.metric("Actual outcome", actual)
    probabilities = result["fitted"][insight_model].predict_proba(row)[0]
    classes = result["fitted"][insight_model].named_steps["model"].classes_
    prob_df = pd.DataFrame({"Outcome": classes, "Probability": probabilities}).sort_values("Probability", ascending=False)
    fig = px.bar(prob_df, x="Outcome", y="Probability", color="Outcome", range_y=[0, 1], template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("### Student input features")
    st.dataframe(row, use_container_width=True)

    st.divider()
    st.markdown("### Random Forest individual SHAP explanation")
    if st.button("Generate SHAP explanation for this test student", key="uci_test_shap_button"):
        try:
            with st.spinner("Generating individual SHAP explanation..."):
                explanation, explained_class = calculate_individual_shap(result["fitted"]["Random Forest"], row)
                st.session_state["uci_test_shap"] = {"index": idx, "explanation": explanation, "class": explained_class}
            st.success(f"Random Forest explanation generated for {explained_class}.")
        except Exception as error:
            st.error("Individual SHAP calculation failed.")
            st.exception(error)

    local = st.session_state.get("uci_test_shap")
    if local is not None and local["index"] == idx:
        import shap
        import matplotlib.pyplot as plt
        st.caption(f"Explained Random Forest class: {local['class']}")
        shap.plots.waterfall(local["explanation"], max_display=15, show=False)
        fig = plt.gcf()
        st.pyplot(fig, clear_figure=False, use_container_width=True)
        plt.close(fig)
        contrib = pd.DataFrame({"Feature": local["explanation"].feature_names, "Transformed Value": local["explanation"].data, "SHAP Value": local["explanation"].values})
        contrib["Absolute SHAP"] = contrib["SHAP Value"].abs()
        contrib = contrib.sort_values("Absolute SHAP", ascending=False)
        st.dataframe(contrib.drop(columns="Absolute SHAP"), use_container_width=True, hide_index=True)
        st.download_button("Download test-student SHAP", contrib.drop(columns="Absolute SHAP").to_csv(index=False).encode("utf-8"), f"uci_test_student_shap_{idx}.csv", "text/csv", key="uci_test_individual_shap_download")

    st.divider()
    st.markdown("### Test prediction audit")
    audit = result["X_test"].copy()
    audit.insert(0, "Actual Outcome", result["y_test"])
    for model_name, model_predictions in result["predictions"].items():
        audit[f"{model_name} Prediction"] = model_predictions
    st.dataframe(audit, use_container_width=True)
    st.download_button("Download test prediction audit", audit.to_csv(index=False).encode("utf-8"), "uci_test_predictions.csv", "text/csv", key="uci_test_audit_download")

# ============================================================
# TAB 5 — NEW STUDENT PREDICTION
# ============================================================

with tabs[4]:
    st.markdown("## New Student Prediction")
    st.markdown('<div class="section-description">Enter a new student profile. Family education, family structure, parent occupation, support, study habits, and earlier grades are included as model inputs.</div>', unsafe_allow_html=True)
    st.info("The form uses the UCI dataset's real predictors. Mother/father education and family background are therefore used by the trained model, not just displayed in the form.")

    # Defaults from the full training data; all values are editable.
    train = result["X_train"]
    defaults = {}
    for col in result["features"]:
        if col in NUMERIC_FEATURES:
            defaults[col] = float(pd.to_numeric(train[col], errors="coerce").median())
        else:
            modes = train[col].mode(dropna=True)
            defaults[col] = str(modes.iloc[0]) if not modes.empty else "unknown"

    with st.form("uci_new_student_form", clear_on_submit=False):
        st.markdown("### A. Student profile")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            school = st.selectbox("School", ["GP", "MS"], index=0 if defaults["school"] == "GP" else 1)
        with c2:
            sex = st.selectbox("Gender", ["F", "M"], index=0 if defaults["sex"] == "F" else 1)
        with c3:
            age = st.number_input("Age", min_value=15, max_value=22, value=int(np.clip(round(defaults["age"]), 15, 22)))
        with c4:
            address = st.selectbox("Home address", ["U", "R"], index=0 if defaults["address"] == "U" else 1, format_func=lambda x: "Urban" if x == "U" else "Rural")

        st.divider()
        st.markdown("### B. Family background and parental education")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            famsize = st.selectbox("Family size", ["GT3", "LE3"], index=0 if defaults["famsize"] == "GT3" else 1, format_func=lambda x: "More than 3" if x == "GT3" else "3 or fewer")
        with c2:
            pstatus = st.selectbox("Parents living together", ["T", "A"], index=0 if defaults["Pstatus"] == "T" else 1, format_func=lambda x: "Together" if x == "T" else "Apart")
        with c3:
            medu = st.selectbox("Mother's education", [0, 1, 2, 3, 4], index=int(np.clip(round(defaults["Medu"]), 0, 4)), format_func=lambda x: {0:"None",1:"Primary (4th grade)",2:"5th–9th grade",3:"Secondary",4:"Higher education"}[x])
        with c4:
            fedu = st.selectbox("Father's education", [0, 1, 2, 3, 4], index=int(np.clip(round(defaults["Fedu"]), 0, 4)), format_func=lambda x: {0:"None",1:"Primary (4th grade)",2:"5th–9th grade",3:"Secondary",4:"Higher education"}[x])

        c1, c2, c3 = st.columns(3)
        with c1:
            mjob = st.selectbox("Mother's occupation", ["teacher", "health", "services", "at_home", "other"], index=["teacher", "health", "services", "at_home", "other"].index(defaults["Mjob"]) if defaults["Mjob"] in ["teacher", "health", "services", "at_home", "other"] else 4)
        with c2:
            fjob = st.selectbox("Father's occupation", ["teacher", "health", "services", "at_home", "other"], index=["teacher", "health", "services", "at_home", "other"].index(defaults["Fjob"]) if defaults["Fjob"] in ["teacher", "health", "services", "at_home", "other"] else 4)
        with c3:
            guardian = st.selectbox("Main guardian", ["mother", "father", "other"], index=["mother", "father", "other"].index(defaults["guardian"]) if defaults["guardian"] in ["mother", "father", "other"] else 0)

        c1, c2, c3 = st.columns(3)
        with c1:
            famsup = st.selectbox("Family educational support", ["yes", "no"], index=0 if defaults["famsup"] == "yes" else 1)
        with c2:
            famrel = st.slider("Family relationship quality (1–5)", 1, 5, int(np.clip(round(defaults["famrel"]), 1, 5)))
        with c3:
            schoolsup = st.selectbox("Extra educational support", ["yes", "no"], index=0 if defaults["schoolsup"] == "yes" else 1)

        st.divider()
        st.markdown("### C. Academic and school background")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            reason = st.selectbox("Reason for choosing school", ["course", "reputation", "home", "other"], index=["course", "reputation", "home", "other"].index(defaults["reason"]) if defaults["reason"] in ["course", "reputation", "home", "other"] else 0)
        with c2:
            traveltime = st.selectbox("Travel time to school", [1, 2, 3, 4], index=int(np.clip(round(defaults["traveltime"]), 1, 4))-1, format_func=lambda x: {1:"< 15 min",2:"15–30 min",3:"30–60 min",4:"> 60 min"}[x])
        with c3:
            studytime = st.selectbox("Weekly study time", [1, 2, 3, 4], index=int(np.clip(round(defaults["studytime"]), 1, 4))-1, format_func=lambda x: {1:"< 2 hours",2:"2–5 hours",3:"5–10 hours",4:"> 10 hours"}[x])
        with c4:
            failures = st.selectbox("Past class failures", [0, 1, 2, 3], index=int(np.clip(round(defaults["failures"]), 0, 3)))

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            paid = st.selectbox("Extra paid classes", ["yes", "no"], index=0 if defaults["paid"] == "yes" else 1)
        with c2:
            activities = st.selectbox("Extracurricular activities", ["yes", "no"], index=0 if defaults["activities"] == "yes" else 1)
        with c3:
            higher = st.selectbox("Wants higher education", ["yes", "no"], index=0 if defaults["higher"] == "yes" else 1)
        with c4:
            nursery = st.selectbox("Attended nursery school", ["yes", "no"], index=0 if defaults["nursery"] == "yes" else 1)

        c1, c2, c3 = st.columns(3)
        with c1:
            internet = st.selectbox("Internet access at home", ["yes", "no"], index=0 if defaults["internet"] == "yes" else 1)
        with c2:
            romantic = st.selectbox("In a romantic relationship", ["yes", "no"], index=0 if defaults["romantic"] == "yes" else 1)
        with c3:
            absences = st.number_input("School absences", min_value=0, max_value=100, value=int(np.clip(round(defaults["absences"]), 0, 100)))

        st.divider()
        st.markdown("### D. Social habits and wellbeing")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            freetime = st.slider("Free time (1–5)", 1, 5, int(np.clip(round(defaults["freetime"]), 1, 5)))
        with c2:
            goout = st.slider("Going out with friends (1–5)", 1, 5, int(np.clip(round(defaults["goout"]), 1, 5)))
        with c3:
            dalc = st.slider("Weekday alcohol consumption (1–5)", 1, 5, int(np.clip(round(defaults["Dalc"]), 1, 5)))
        with c4:
            walc = st.slider("Weekend alcohol consumption (1–5)", 1, 5, int(np.clip(round(defaults["Walc"]), 1, 5)))
        with c5:
            health = st.slider("Current health (1–5)", 1, 5, int(np.clip(round(defaults["health"]), 1, 5)))

        st.divider()
        st.markdown("### E. Previous academic grades")
        if {"G1", "G2"}.issubset(result["features"]):
            st.caption("G1 and G2 are earlier-period grades on the UCI 0–20 scale. G3, the final grade, is never entered as a predictor.")
            c1, c2 = st.columns(2)
            with c1:
                g1 = st.number_input("First period grade (G1)", min_value=0, max_value=20, value=int(np.clip(round(defaults["G1"]), 0, 20)))
            with c2:
                g2 = st.number_input("Second period grade (G2)", min_value=0, max_value=20, value=int(np.clip(round(defaults["G2"]), 0, 20)))
        else:
            g1, g2 = None, None
            st.info("Early-stage mode excludes G1 and G2. The prediction uses only student, family, school, study, and behaviour information.")

        submitted = st.form_submit_button("Predict Student Outcome", use_container_width=True, type="primary")

    if submitted:
        student_input = {
            "school": school,
            "sex": sex,
            "age": age,
            "address": address,
            "famsize": famsize,
            "Pstatus": pstatus,
            "Medu": medu,
            "Fedu": fedu,
            "Mjob": mjob,
            "Fjob": fjob,
            "reason": reason,
            "guardian": guardian,
            "traveltime": traveltime,
            "studytime": studytime,
            "failures": failures,
            "schoolsup": schoolsup,
            "famsup": famsup,
            "paid": paid,
            "activities": activities,
            "nursery": nursery,
            "higher": higher,
            "internet": internet,
            "romantic": romantic,
            "famrel": famrel,
            "freetime": freetime,
            "goout": goout,
            "Dalc": dalc,
            "Walc": walc,
            "health": health,
            "absences": absences,
            "G1": g1,
            "G2": g2,
        }
        new_row = pd.DataFrame([student_input], columns=result["features"])
        st.session_state["uci_new_prediction"] = new_row

    if "uci_new_prediction" in st.session_state:
        new_row = st.session_state["uci_new_prediction"]
        st.divider()
        st.markdown("### Prediction results")
        rf_pipe = result["fitted"]["Random Forest"]
        rf_pred = rf_pipe.predict(new_row)[0]
        rf_proba = rf_pipe.predict_proba(new_row)[0]
        rf_classes = rf_pipe.named_steps["model"].classes_
        predicted_probability = float(rf_proba[list(rf_classes).index(rf_pred)])

        col1, col2, col3 = st.columns(3)
        col1.metric("Random Forest prediction", rf_pred)
        col2.metric("Predicted-class probability", f"{predicted_probability:.1%}")
        col3.metric("Prediction type", "New student")

        probability_df = pd.DataFrame({"Outcome": rf_classes, "Probability": rf_proba}).sort_values("Probability", ascending=False)
        fig = px.bar(probability_df, x="Outcome", y="Probability", color="Outcome", range_y=[0, 1], text_auto=".1%", template="plotly_white", color_discrete_map={"Pass": "#2563EB", "At-Risk": "#F59E0B"})
        fig.update_layout(paper_bgcolor="white", plot_bgcolor="white", font_color="#17365D")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Compare all trained models")
        comparison_rows = []
        for model_name, model_pipe in result["fitted"].items():
            p = model_pipe.predict(new_row)[0]
            pr = model_pipe.predict_proba(new_row)[0]
            cls = model_pipe.named_steps["model"].classes_
            conf = float(pr[list(cls).index(p)])
            comparison_rows.append({"Model": model_name, "Prediction": p, "Probability of predicted class": conf})
        st.dataframe(pd.DataFrame(comparison_rows).style.format({"Probability of predicted class": "{:.1%}"}), use_container_width=True, hide_index=True)

        st.caption("This is a new inference example, not part of the training or held-out test dataset. Probabilities are model outputs and have not been independently calibrated.")
        st.download_button("Download new student input and prediction", pd.concat([new_row.reset_index(drop=True), pd.DataFrame([{"Predicted Outcome": rf_pred, "Predicted Class Probability": predicted_probability}])], axis=1).to_csv(index=False).encode("utf-8"), "uci_new_student_prediction.csv", "text/csv", key="uci_new_student_prediction_download")

        st.divider()
        st.markdown("### Individual SHAP explanation for this new student")
        st.caption("This explanation uses the Random Forest prediction, irrespective of the comparison results from the other models.")
        if st.button("Generate SHAP explanation for new student", key="uci_new_student_shap_button"):
            try:
                with st.spinner("Calculating new-student SHAP explanation..."):
                    explanation, explained_class = calculate_individual_shap(rf_pipe, new_row)
                    st.session_state["uci_new_student_shap"] = {"explanation": explanation, "class": explained_class}
                st.success(f"Explanation generated for Random Forest class: {explained_class}")
            except Exception as error:
                st.error("New-student SHAP calculation failed.")
                st.exception(error)

        local_new = st.session_state.get("uci_new_student_shap")
        if local_new is not None:
            import shap
            import matplotlib.pyplot as plt
            st.caption(f"Explained outcome: {local_new['class']}")
            shap.plots.waterfall(local_new["explanation"], max_display=15, show=False)
            fig = plt.gcf()
            st.pyplot(fig, clear_figure=False, use_container_width=True)
            plt.close(fig)
            local_df = pd.DataFrame({"Feature": local_new["explanation"].feature_names, "Transformed Value": local_new["explanation"].data, "SHAP Value": local_new["explanation"].values})
            local_df["Absolute SHAP"] = local_df["SHAP Value"].abs()
            local_df = local_df.sort_values("Absolute SHAP", ascending=False)
            st.dataframe(local_df.drop(columns="Absolute SHAP"), use_container_width=True, hide_index=True)
            st.download_button("Download new-student SHAP explanation", local_df.drop(columns="Absolute SHAP").to_csv(index=False).encode("utf-8"), "uci_new_student_shap.csv", "text/csv", key="uci_new_student_shap_download")

# ============================================================
# TAB 6 — RESEARCH EXPORT
# ============================================================

with tabs[5]:
    st.markdown("## Research Data and Exports")
    st.caption("Export your prepared dataset, held-out test results, cross-validation, explainability values, and experiment configuration.")

    st.download_button("Download prepared UCI dataset", data.to_csv(index=False).encode("utf-8"), "uci_student_performance_prepared.csv", "text/csv", key="uci_export_dataset")
    st.download_button("Download test metrics", result["metrics"].to_csv(index=False).encode("utf-8"), "uci_test_metrics.csv", "text/csv", key="uci_export_test_metrics")
    st.download_button("Download cross-validation metrics", result["cv"].to_csv(index=False).encode("utf-8"), "uci_cv_metrics.csv", "text/csv", key="uci_export_cv_metrics")

    if "uci_global_shap" in st.session_state:
        st.download_button("Download global SHAP results", st.session_state["uci_global_shap"].to_csv(index=False).encode("utf-8"), "uci_shap_global.csv", "text/csv", key="uci_export_global_shap")
    if "uci_permutation" in st.session_state:
        st.download_button("Download permutation importance", st.session_state["uci_permutation"].to_csv(index=False).encode("utf-8"), "uci_permutation_importance.csv", "text/csv", key="uci_export_permutation")

    config = {
        "project": "EduPredict",
        "dataset": "UCI Student Performance Mathematics (student-mat.csv)",
        "target": "G3 >= 10: Pass; G3 < 10: At-Risk",
        "prediction_stage": prediction_stage,
        "predictors": result["features"],
        "g3_used_as_predictor": False,
        "earlier_grades_used": [g for g in ["G1", "G2"] if g in result["features"]],
        "train_test_split": "Stratified 75:25 by default",
        "random_state": RANDOM_STATE,
        "training_records": len(result["X_train"]),
        "testing_records": len(result["X_test"]),
        "cross_validation": f"Stratified {result['n_splits']}-fold CV on training data only",
        "models": list(result["fitted"].keys()),
        "evaluation": "Held-out test data",
        "new_student_input": "Manual inputs are separate inference records and are not used to retrain or score the model.",
    }
    st.markdown("### Experiment configuration")
    st.json(config)
    st.download_button("Download experiment configuration", json.dumps(config, indent=4), "uci_experiment_config.json", "application/json", key="uci_export_config")

    if prediction_stage.startswith("Early"):
        st.info("Research note: This experiment excludes G1 and G2. Describe it as early-stage prediction using background, school, study, and behavioural variables. G3 is used only to derive the Pass / At-Risk target.")
    else:
        st.warning("Research note: This experiment includes G1 and G2 as predictors of G3. Describe it as later-stage prediction after earlier grading periods, not as prediction before any grades are available. Verify all metrics from your own run before reporting them.")

# ============================================================
# 11. FOOTER
# ============================================================

st.markdown("---")
st.markdown(
    '<div class="footer"><strong>EduPredict</strong><br>UCI Student Performance Mathematics · Student & Family Background · Machine Learning · Explainable AI</div>',
    unsafe_allow_html=True,
)
