import streamlit as st
import pandas as pd
import plotly.express as px
import os, io, json
from src.data_loader import load_dataset
from src.preprocess import prepare_df
from src.model_utils import load_pipeline
from src.predict import predict_batch, predict_single

ROOT = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(ROOT, "models")
st.set_page_config(page_title="Student Performance — Academic Pro", layout="wide",
                   initial_sidebar_state="expanded")

# custom CSS
st.markdown("""
<style>
body { background-color: #f8f9fb; }
h1 { color: #003366; }
.sidebar .sidebar-content { background-image: linear-gradient(#ffffff, #f1f5fb); }
.card { background: white; padding: 1rem; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.08);}
</style>
""", unsafe_allow_html=True)

st.title("🎓 Student Performance Predictor — Academic Pro Edition")
st.markdown("**Purpose:** Predict student final-performance category using an explainable ML pipeline. Use the tabs to explore, predict and export results.")

# Load dataset (sample)
@st.cache_data
def load_data_cached():
    return load_dataset()

df = load_data_cached()
dfp = prepare_df(df)
feature_cols = dfp.drop(columns=['performance']).columns.tolist()

tabs = st.tabs(["Home", "Predict", "Insights", "About"])

with tabs[0]:
    st.header("Overview")
    st.write("This application models student academic performance (Low / Medium / High) using a robust ML pipeline.")
    st.subheader("Dataset snapshot")
    st.dataframe(df.head(8))
    st.subheader("Feature distribution examples")
    col1, col2 = st.columns(2)
    with col1:
        num_cols = df.select_dtypes(include=['int64','float']).columns.tolist()
        sel = st.selectbox("Numeric feature", num_cols, index=0)
        fig = px.histogram(df, x=sel, nbins=20, title=f"Distribution of {sel}")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        selc = st.selectbox("Categorical feature", cat_cols, index=0)
        fig2 = px.histogram(df, x=selc, title=f"Counts of {selc}")
        st.plotly_chart(fig2, use_container_width=True)

with tabs[1]:
    st.header("Predict")
    st.write("Single student prediction or upload a CSV for batch inference.")

    col1, col2 = st.columns([1,1])
    with col1:
        st.subheader("Single prediction")
        example = dfp.drop(columns=['performance']).iloc[0].to_dict()
        form = st.form(key="single_form")
        inputs = {}
        for c, val in example.items():
            if pd.api.types.is_numeric_dtype(df[c].dtype):
                inputs[c] = form.number_input(c, value=float(val))
            else:
                options = sorted(df[c].unique().tolist())
                inputs[c] = form.selectbox(c, options, index=options.index(val))
        submitted = form.form_submit_button("Predict single")
        if submitted:
            try:
                pipeline = load_pipeline()
                res = predict_single(inputs)
                st.success(f"Predicted: **{res['prediction']}**")
                if res['probability'] is not None:
                    probs = dict(zip(res['classes'], [f"{p:.3f}" for p in res['probability']]))
                    st.write("Probabilities:", probs)
            except Exception as e:
                st.error("Model not available. Train first. Error: " + str(e))

    with col2:
        st.subheader("Batch prediction (CSV upload)")
        uploaded = st.file_uploader("Upload CSV with columns matching training features", type=['csv'])
        if uploaded is not None:
            try:
                df_upload = pd.read_csv(uploaded)
                st.write("Preview uploaded data:")
                st.dataframe(df_upload.head())
                if st.button("Run batch prediction"):
                    pipeline = load_pipeline()
                    out = predict_batch(df_upload)
                    st.success("Batch prediction complete.")
                    st.dataframe(out.head())
                    # allow download as Excel
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                        out.to_excel(writer, index=False, sheet_name="predictions")
                        writer.save()
                    buffer.seek(0)
                    st.download_button("Download predictions (xlsx)", data=buffer, file_name="predictions.xlsx")
            except Exception as e:
                st.error("Error processing uploaded file: " + str(e))

with tabs[2]:
    st.header("Insights & Model Explanation")
    st.write("Model performance metrics and SHAP feature importance (if computed).")
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        st.subheader("Cross-validated metrics")
        st.json({
            "best_cv_score": metrics.get("best_cv_score"),
            "cv_mean_accuracy": metrics.get("cv_mean_accuracy"),
            "cv_std": metrics.get("cv_std")
        })
        st.write("Full-data classification report (in-sample):")
        st.json(metrics.get("full_data_report"))
    else:
        st.info("Metrics not found. Train using src/train.py to produce metrics.")

    shap_path = os.path.join(MODELS_DIR, "shap_summary.csv")
    if os.path.exists(shap_path):
        shap_df = pd.read_csv(shap_path)
        st.subheader("Top features by mean |SHAP|")
        st.table(shap_df.head(15))
        fig = px.bar(shap_df.head(15), x="mean_abs_shap", y="feature", orientation='h')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("SHAP summary not found. Compute with: python -m src.explain")

with tabs[3]:
    st.header("About")
    st.markdown("""
    **Author:** X  
    **Project:** Predictive Analysis of Student Academic Performance  
    **Tools:** Python, scikit-learn, SHAP, Streamlit, Plotly  
    **Note:** Data privacy: do not upload sensitive real student data without consent.
    """)
