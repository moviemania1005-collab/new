"""
app.py - Single-file Academic-Pro Edition
Features:
- Download UCI Student dataset (math)
- Preprocess & feature pipeline
- Train models (RandomForest default), optional RandomizedSearchCV tuning
- Save/load pipeline (joblib) in models/
- Compute SHAP summary (if shap available)
- Streamlit UI with tabs: Home, Predict (single + CSV batch), Insights, About
- Command-line flags:
    python app.py --train        # trains full (with tuning by default)
    python app.py --train --fast # trains quickly (no tuning)
    python app.py --explain      # compute SHAP summary (after training)
    streamlit run app.py         # launches Streamlit UI
"""

import os
import sys
import argparse
import json
from io import BytesIO
from zipfile import ZipFile
from datetime import datetime

# Data libs
import pandas as pd
import numpy as np
import requests
import joblib

# ML libs
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.stats import randint

# Optional visualization/library imports inside try blocks
try:
    import streamlit as st
    import plotly.express as px
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:
    st = None
    px = None
    plt = None
    sns = None

# SHAP (optional)
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# Directories
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Constants
UCI_BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/"
DEFAULT_CSV = "student-mat.csv"
PIPE_FILENAME = os.path.join(MODELS_DIR, "best_pipeline.joblib")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")
SHAP_SUM_PATH = os.path.join(MODELS_DIR, "shap_summary.csv")

#########################
# Data utilities
#########################
def download_if_missing(filename=DEFAULT_CSV, force=False):
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path) and not force:
        print(f"[data] Found {path}")
        return path
    print("[data] Downloading dataset from UCI...")
    zip_url = UCI_BASE + "student.zip"
    r = requests.get(zip_url, timeout=30)
    r.raise_for_status()
    z = ZipFile(BytesIO(r.content))
    members = [m for m in z.namelist() if m.endswith(filename)]
    if not members:
        raise FileNotFoundError(f"{filename} not found in zip archive.")
    z.extract(members[0], path=DATA_DIR)
    extracted = os.path.join(DATA_DIR, members[0])
    final = os.path.join(DATA_DIR, filename)
    if extracted != final:
        os.replace(extracted, final)
    print(f"[data] Saved dataset to {final}")
    return final

def load_data(filename=DEFAULT_CSV):
    path = download_if_missing(filename)
    df = pd.read_csv(path, sep=';')
    return df

#########################
# Preprocessing
#########################
def label_performance(g3):
    if g3 < 10:
        return "Low"
    elif g3 < 15:
        return "Medium"
    else:
        return "High"

def prepare_dataframe(df, drop_cols=None):
    df = df.copy()
    df['performance'] = df['G3'].apply(label_performance)
    if drop_cols is None:
        drop_cols = []
    drop = ['G3'] + drop_cols
    df = df.drop(columns=[c for c in drop if c in df.columns], errors='ignore')
    return df

def get_feature_lists(df):
    cat = df.select_dtypes(include=['object', 'category']).columns.tolist()
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'performance' in cat: cat.remove('performance')
    if 'performance' in num: num.remove('performance')
    return cat, num

def build_preprocessor(df):
    cat, num = get_feature_lists(df)
    numeric_transformer = Pipeline([('scaler', StandardScaler())])
    categorical_transformer = Pipeline([('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))])
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, num),
        ('cat', categorical_transformer, cat)
    ], remainder='drop')
    return preprocessor

#########################
# Train / Evaluate
#########################
def train_models(df, tune=True, n_iter=30, random_state=42, n_jobs=-1):
    """
    Train models and optionally perform RandomizedSearchCV for RandomForest.
    Returns best_pipeline, metrics dict
    """
    print("[train] Preparing data...")
    dfp = prepare_dataframe(df)
    X = dfp.drop(columns=['performance'])
    y = dfp['performance']

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.25,
                                                        random_state=random_state,
                                                        stratify=y)
    preprocessor = build_preprocessor(X_train)

    # Candidate base models
    candidates = {
        "RandomForest": RandomForestClassifier(random_state=random_state),
        "DecisionTree": DecisionTreeClassifier(random_state=random_state),
        "SVM": SVC(probability=True, random_state=random_state)
    }

    results = {}

    # If tuning, perform RandomizedSearchCV on RandomForest pipeline
    if tune:
        print("[train] Running RandomizedSearchCV for RandomForest (this may take time)...")
        pipe = Pipeline([('preprocessor', preprocessor),
                         ('classifier', RandomForestClassifier(random_state=random_state))])
        param_dist = {
            "classifier__n_estimators": randint(50, 400),
            "classifier__max_depth": randint(3, 20),
            "classifier__min_samples_split": randint(2, 10),
            "classifier__min_samples_leaf": randint(1, 5),
            "classifier__max_features": ['auto', 'sqrt', 0.3, 0.5, 0.7]
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        rs = RandomizedSearchCV(pipe, param_distributions=param_dist, n_iter=n_iter,
                                cv=cv, scoring='accuracy', n_jobs=n_jobs, verbose=1, random_state=random_state)
        rs.fit(X_train, y_train)
        best_pipeline = rs.best_estimator_
        print(f"[train] RandomizedSearchCV best CV score: {rs.best_score_:.4f}")
        results['RandomForest_tuned'] = {
            "accuracy": float(rs.score(X_test, y_test)),
            "cv_best_score": float(rs.best_score_),
            "best_params": rs.best_params_
        }
        # still evaluate other baseline models without tuning
        for name, clf in candidates.items():
            if name == "RandomForest":
                # evaluate tuned model above
                preds = best_pipeline.predict(X_test)
                results['RandomForest'] = {
                    "accuracy": float(accuracy_score(y_test, preds)),
                    "report": classification_report(y_test, preds, output_dict=True)
                }
            else:
                pipe = Pipeline([('preprocessor', preprocessor), ('classifier', clf)])
                pipe.fit(X_train, y_train)
                preds = pipe.predict(X_test)
                results[name] = {
                    "accuracy": float(accuracy_score(y_test, preds)),
                    "report": classification_report(y_test, preds, output_dict=True)
                }
    else:
        # quick training without heavy search
        pre = preprocessor
        pipelines = {}
        for name, clf in candidates.items():
            pipe = Pipeline([('preprocessor', pre), ('classifier', clf)])
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)
            results[name] = {
                "accuracy": float(accuracy_score(y_test, preds)),
                "report": classification_report(y_test, preds, output_dict=True)
            }
        # choose best by accuracy
        best_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
        print(f"[train] Best model (no tuning): {best_name}")
        best_pipeline = Pipeline([('preprocessor', pre), ('classifier', candidates[best_name])])
        best_pipeline.fit(X_train, y_train)

    # Save pipeline & metrics
    print("[train] Saving pipeline and metrics...")
    joblib.dump(best_pipeline, PIPE_FILENAME)
    with open(METRICS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print("[train] Saved pipeline to:", PIPE_FILENAME)
    print("[train] Saved metrics to:", METRICS_PATH)
    return best_pipeline, results

#########################
# SHAP explainability
#########################
def compute_shap_summary(pipeline_path=PIPE_FILENAME, nsample=500):
    if not SHAP_AVAILABLE:
        print("[shap] SHAP library not available. Install 'shap' to enable explainability.")
        return None
    print("[shap] Computing SHAP summary. Loading pipeline...")
    pipeline = joblib.load(pipeline_path)
    # get data
    df = load_data()
    dfp = prepare_dataframe(df)
    X = dfp.drop(columns=['performance'])
    # sample
    Xs = X.sample(n=min(nsample, X.shape[0]), random_state=42)
    pre = pipeline.named_steps['preprocessor']
    clf = pipeline.named_steps['classifier']
    # transform
    X_trans = pre.transform(Xs)
    try:
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_trans)
        if isinstance(shap_values, list):
            mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        else:
            mean_abs = np.abs(shap_values).mean(axis=0)
    except Exception as e:
        print("[shap] TreeExplainer failed, attempting KernelExplainer (slow)...", e)
        explainer = shap.KernelExplainer(clf.predict_proba, shap.kmeans(X_trans, 10))
        shap_values = explainer.shap_values(X_trans[:100])
        if isinstance(shap_values, list):
            mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        else:
            mean_abs = np.abs(shap_values).mean(axis=0)

    # get feature names from preprocessor
    try:
        num_cols = pre.transformers_[0][2]
        cat_pipe = pre.transformers_[1][1]
        cat_cols = pre.transformers_[1][2]
        ohe = cat_pipe.named_steps['ohe']
        cat_feature_names = list(ohe.get_feature_names_out(cat_cols))
        feature_names = list(num_cols) + cat_feature_names
    except Exception:
        feature_names = [f"f{i}" for i in range(len(mean_abs))]
    df_shap = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    df_shap = df_shap.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    df_shap.to_csv(SHAP_SUM_PATH, index=False)
    print("[shap] Saved SHAP summary to:", SHAP_SUM_PATH)
    return df_shap

#########################
# Prediction utilities
#########################
def load_pipeline(pipeline_path=PIPE_FILENAME):
    if not os.path.exists(pipeline_path):
        raise FileNotFoundError("Pipeline file not found. Train first using --train.")
    return joblib.load(pipeline_path)

def predict_single(sample_dict, pipeline=None):
    if pipeline is None:
        pipeline = load_pipeline()
    X = pd.DataFrame([sample_dict])
    pred = pipeline.predict(X)[0]
    probs = pipeline.predict_proba(X)[0].tolist() if hasattr(pipeline, "predict_proba") else None
    classes = pipeline.classes_.tolist()
    return {"prediction": pred, "probabilities": probs, "classes": classes}

def predict_batch(df_input, pipeline=None):
    if pipeline is None:
        pipeline = load_pipeline()
    out = df_input.copy()
    preds = pipeline.predict(df_input)
    out['predicted_performance'] = preds
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(df_input)
        for i, cls in enumerate(pipeline.classes_):
            out[f"prob_{cls}"] = probs[:, i]
    return out

#########################
# Streamlit UI
#########################
def run_streamlit_app():
    if st is None:
        print("Streamlit not installed. Install via: pip install streamlit")
        return
    st.set_page_config(page_title="Student Performance — Academic Pro", layout="wide")
    st.markdown("""<style>
        .reportview-container .main .block-container{max-width:1400px; padding:1rem 2rem;}
        .stButton>button {background-color: #0066cc; color: white;}
        </style>""", unsafe_allow_html=True)
    st.title("🎓 Student Performance Predictor — Academic Pro")
    st.markdown("A polished educational analytics demo: model training, prediction, SHAP explainability, and batch exporting.")

    # Load dataset
    try:
        df = load_data()
        dfp = prepare_dataframe(df)
    except Exception as e:
        st.error("Cannot load dataset: " + str(e))
        return

    tabs = st.tabs(["Home", "Predict", "Insights", "About"])

    with tabs[0]:
        st.header("Home — Project Overview")
        st.write("**Dataset:** UCI Student Performance (math). Model predicts final performance category (Low/Medium/High).")
        st.write("**How to use:** Train locally with `python app.py --train` (or use pre-trained pipeline if present). Then use the Predict tab or upload a CSV for batch predictions.")
        st.subheader("Dataset snapshot")
        st.dataframe(df.head(10))
        st.subheader("Quick feature visualizations")
        col1, col2 = st.columns(2)
        with col1:
            if px:
                num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                sel = st.selectbox("Numeric feature for histogram", num_cols, index=0)
                fig = px.histogram(df, x=sel, nbins=20)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("Plotly not available.")
        with col2:
            cat_cols = df.select_dtypes(include=['object']).columns.tolist()
            selc = st.selectbox("Categorical feature for counts", cat_cols, index=0)
            if px:
                fig2 = px.histogram(df, x=selc)
                st.plotly_chart(fig2, use_container_width=True)

    with tabs[1]:
        st.header("Predict")
        st.write("Single prediction (left) or batch CSV upload (right).")

        col1, col2 = st.columns([1,1])
        with col1:
            st.subheader("Single prediction")
            example = dfp.drop(columns=['performance']).iloc[0].to_dict()
            with st.form("single_form"):
                sample = {}
                for k, v in example.items():
                    if isinstance(v, (int, float, np.integer, np.floating)):
                        sample[k] = st.number_input(k, value=float(v))
                    else:
                        opts = sorted(df[k].unique())
                        sample[k] = st.selectbox(k, opts, index=opts.index(v))
                submit = st.form_submit_button("Predict")
                if submit:
                    try:
                        pipeline = load_pipeline()
                        res = predict_single(sample, pipeline)
                        st.success(f"Predicted class: **{res['prediction']}**")
                        if res['probabilities'] is not None:
                            probs = {cls: f"{p:.3f}" for cls, p in zip(res['classes'], res['probabilities'])}
                            st.write("Probabilities:", probs)
                    except Exception as e:
                        st.error("Model not available. Train first. Error: " + str(e))

        with col2:
            st.subheader("Batch prediction (CSV)")
            uploaded = st.file_uploader("Upload CSV (features only, no G3 column)", type=['csv'])
            if uploaded is not None:
                try:
                    df_upload = pd.read_csv(uploaded)
                    st.write("Preview uploaded data:")
                    st.dataframe(df_upload.head())
                    if st.button("Run batch predictions"):
                        pipeline = load_pipeline()
                        out = predict_batch(df_upload, pipeline)
                        st.success("Batch prediction completed.")
                        st.dataframe(out.head())
                        # download
                        csv_bytes = out.to_csv(index=False).encode('utf-8')
                        st.download_button("Download predictions CSV", data=csv_bytes, file_name="predictions.csv", mime="text/csv")
                except Exception as e:
                    st.error("Error processing uploaded file: " + str(e))

    with tabs[2]:
        st.header("Insights & Explainability")
        # show metrics if available
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, "r") as f:
                metrics = json.load(f)
            st.subheader("Model metrics (from training)")
            st.json(metrics)
        else:
            st.info("No metrics found. Train with `python app.py --train` to generate metrics.")

        st.subheader("SHAP feature importance (mean |SHAP|)")
        if os.path.exists(SHAP_SUM_PATH):
            df_shap = pd.read_csv(SHAP_SUM_PATH)
            st.table(df_shap.head(15))
            if px:
                fig = px.bar(df_shap.head(15), x='mean_abs_shap', y='feature', orientation='h')
                st.plotly_chart(fig, use_container_width=True)
        else:
            if SHAP_AVAILABLE:
                if st.button("Compute SHAP summary (may take time)"):
                    st.info("Computing SHAP... this may take a while.")
                    df_shap = compute_shap_summary()
                    if df_shap is not None:
                        st.success("SHAP summary computed.")
                        st.table(df_shap.head(15))
            else:
                st.warning("SHAP not installed. To enable, pip install shap and compute via --explain")

    with tabs[3]:
        st.header("About & Notes")
        st.markdown("""
        **Project:** Predictive Analysis of Student Academic Performance (Academic-Pro Edition)  
        **Author:** X (replace placeholder)  
        **Usage:** Train locally (python app.py --train) then run the Streamlit UI (streamlit run app.py) or deploy to Streamlit Cloud.  
        **Privacy:** Do not upload personal or sensitive student data to public deploys without consent and compliance.
        """)
        st.markdown("Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    st.markdown("---")
    st.write("Notes: This demo saves model artifacts in the `models/` directory. For production, secure storage and access controls are required.")

#########################
# CLI entrypoint
#########################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train model (with tuning by default).")
    parser.add_argument("--fast", action="store_true", help="Fast training (no RandomizedSearchCV).")
    parser.add_argument("--explain", action="store_true", help="Compute SHAP summary (requires shap).")
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV, help="Which CSV inside UCI zip to use (student-mat.csv or student-por.csv).")
    parser.add_argument("--niter", type=int, default=30, help="n_iter for RandomizedSearchCV (if tuning). Lower for faster runs.")
    args = parser.parse_args()

    if args.train:
        print("[main] Loading data...")
        df = load_data(args.csv)
        tune = not args.fast
        pipeline, metrics = train_models(df, tune=tune, n_iter=args.niter)
        print("[main] Training finished.")
        if args.explain:
            if SHAP_AVAILABLE:
                compute_shap_summary()
            else:
                print("[main] SHAP not available. Install shap to compute explanations.")
    elif args.explain:
        if SHAP_AVAILABLE:
            compute_shap_summary()
        else:
            print("[main] SHAP not available. Install shap to compute explanations.")
    else:
        # If running via streamlit, the run_streamlit_app will be invoked by Streamlit process.
        # If user runs python app.py (no args) we attempt to launch streamlit if available.
        if st is not None and ("streamlit" in sys.argv[0] or os.environ.get("STREAMLIT_RUNNING")):
            run_streamlit_app()
        else:
            # Provide helpful message
            print("No arguments provided. To train: python app.py --train")
            print("To run Streamlit UI: streamlit run app.py")
            print("To compute SHAP summary: python app.py --explain")

if __name__ == "__main__":
    main()
