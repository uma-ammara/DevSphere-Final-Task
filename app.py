import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------- Page setup
st.set_page_config(
    page_title="SkillMatch | Resume Job Category Predictor",
    page_icon="📄",
    layout="wide",
)

MODEL_PATH = Path(__file__).parent / "model" / "skillmatch_model.pkl"
MIN_WORDS = 20

st.markdown(
    """
    <style>
        .block-container {padding-top: 2rem; max-width: 1150px;}
        .result-card {
            background: #f4f7fb; border-left: 6px solid #2563eb;
            padding: 1.1rem 1.4rem; border-radius: 8px; margin-bottom: 1rem;
        }
        .result-card .label {color: #64748b; font-size: 0.85rem; margin: 0;}
        .result-card .value {color: #0f172a; font-size: 1.8rem; font-weight: 700; margin: 0;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------- Helpers
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def clean_text(text):
    text = str(text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def set_example(text):
    st.session_state["resume_text"] = text


EXAMPLES = {
    "HR": (
        "HR manager with 8 years of experience in recruitment, talent acquisition, "
        "employee relations, payroll administration and onboarding. Developed HR "
        "policies, managed performance reviews, handled benefits administration and "
        "ensured compliance with labor laws. Skilled in HRIS systems and training programs."
    ),
    "IT": (
        "Software engineer with 6 years of experience in Python, Java and SQL. Built "
        "web applications, managed databases, configured servers and networks, and "
        "provided technical support. Experienced with cloud platforms, Linux "
        "administration, API development and agile software development."
    ),
    "Chef": (
        "Executive chef with 10 years of experience in fine dining kitchens. Designed "
        "seasonal menus, supervised kitchen staff, managed food costs and inventory, "
        "and ensured food safety and sanitation standards. Skilled in butchery, sauces, "
        "pastry preparation and banquet catering."
    ),
}

# ------------------------------------------------------------------- Sidebar
with st.sidebar:
    st.title("📄 SkillMatch")
    st.caption("Resume-Based Job Category Prediction")
    st.divider()
    st.subheader("About")
    st.write(
        "This app predicts the most likely job category of a resume using "
        "traditional machine learning: **TF-IDF** text features and a "
        "**Logistic Regression** classifier."
    )
    st.subheader("How to use")
    st.markdown(
        "1. Paste resume text or upload a `.txt` file\n"
        "2. Click **Predict Category**\n"
        "3. Review the top predictions"
    )
    st.divider()
    top_n = st.slider("Top predictions to show", 1, 5, 3)

# ---------------------------------------------------------------- Load model
try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load the model at `{MODEL_PATH}`.\n\n{e}")
    st.stop()

# -------------------------------------------------------------------- Header
st.title("Resume Job Category Predictor")
st.write("Paste a resume below to find out which job category it best matches.")

if "resume_text" not in st.session_state:
    st.session_state["resume_text"] = ""

left, right = st.columns([3, 2], gap="large")

# --------------------------------------------------------------- Input column
with left:
    st.subheader("Resume input")

    uploaded = st.file_uploader("Upload a .txt resume (optional)", type=["txt"])
    if uploaded is not None and st.session_state.get("last_file") != uploaded.name:
        st.session_state["resume_text"] = uploaded.read().decode("utf-8", errors="ignore")
        st.session_state["last_file"] = uploaded.name

    st.caption("Or try an example:")
    ex_cols = st.columns(len(EXAMPLES))
    for col, (name, text) in zip(ex_cols, EXAMPLES.items()):
        col.button(name, on_click=set_example, args=(text,), use_container_width=True)

    resume_text = st.text_area(
        "Resume text",
        key="resume_text",
        height=300,
        placeholder="Paste the full resume text here...",
    )

    word_count = len(resume_text.split())
    st.caption(f"Word count: {word_count}")

    predict = st.button("Predict Category", type="primary", use_container_width=True)

# -------------------------------------------------------------- Result column
with right:
    st.subheader("Prediction")

    if not predict:
        st.info("Your prediction will appear here.")
    else:
        cleaned = clean_text(resume_text)

        if len(cleaned.split()) < MIN_WORDS:
            st.warning(
                f"Please enter at least {MIN_WORDS} words for a useful prediction."
            )
        else:
            prediction = model.predict([cleaned])[0]

            st.markdown(
                f"""
                <div class="result-card">
                    <p class="label">Predicted job category</p>
                    <p class="value">{prediction}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba([cleaned])[0]
                idx = np.argsort(probs)[::-1][:top_n]
                top_df = pd.DataFrame(
                    {
                        "Category": [model.classes_[i] for i in idx],
                        "Confidence": [float(probs[i]) for i in idx],
                    }
                )

                st.metric("Top confidence", f"{top_df['Confidence'].iloc[0]:.1%}")
                st.write("**Top predictions**")
                st.bar_chart(top_df.set_index("Category"), horizontal=True)

                table = top_df.copy()
                table["Confidence"] = table["Confidence"].map("{:.1%}".format)
                st.dataframe(table, hide_index=True, use_container_width=True)

# -------------------------------------------------------------------- Footer
st.divider()
st.caption(
    "Predictions are suggestions from a machine learning model trained on a "
    "limited dataset. Some categories are harder to predict than others, so "
    "treat results as guidance, not a final decision."
)
