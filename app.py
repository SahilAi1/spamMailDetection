"""Spam Mail Detection System - Streamlit Web Application.

A production-grade, interactive web interface for classifying emails as Spam
or Not Spam using a trained Multinomial Naive Bayes pipeline with TF-IDF features.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# Ensure project root is in python path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preprocessing import clean_text

MODEL_PATH = PROJECT_ROOT / "models" / "spam_classifier.joblib"
METRICS_PATH = PROJECT_ROOT / "models" / "metrics.json"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
CONFUSION_MATRIX_PATH = PROJECT_ROOT / "models" / "confusion_matrix.png"

# Page configuration
st.set_page_config(
    page_title="Spam Mail Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    /* Modern typography and container spacing */
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .badge-pill {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    .badge-ready {
        background-color: #dcfce7;
        color: #15803d;
        border: 1px solid #86efac;
    }
    .card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.25rem;
    }
    .result-card-spam {
        background-color: #fef2f2;
        border: 1px solid #fecaca;
        border-left: 6px solid #ef4444;
        border-radius: 10px;
        padding: 1.25rem 1.5rem;
        margin-top: 1.2rem;
    }
    .result-card-ham {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-left: 6px solid #22c55e;
        border-radius: 10px;
        padding: 1.25rem 1.5rem;
        margin-top: 1.2rem;
    }
    .disclaimer-box {
        font-size: 0.85rem;
        color: #64748b;
        background-color: #f8fafc;
        border-left: 4px solid #94a3b8;
        padding: 0.75rem 1rem;
        margin-top: 1.5rem;
        border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_pipeline():
    """Load the serialized scikit-learn pipeline using Streamlit cache."""
    if not MODEL_PATH.exists():
        return None
    try:
        pipeline = joblib.load(MODEL_PATH)
        return pipeline
    except Exception as e:
        st.error(f"Error loading model from {MODEL_PATH}: {e}")
        return None


@st.cache_data
def load_metrics_and_metadata():
    """Load evaluation metrics and model metadata if available."""
    metrics = None
    metadata = None
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    return metrics, metadata


# Sample preset emails for instant testing
SAMPLE_EMAILS = {
    "Select a pre-loaded sample...": "",
    "🚨 Lottery & Prize Scam (Spam)": (
        "Subject: CONGRATULATIONS! You have been selected as the official winner of our international cash lottery! "
        "You have won $1,500,000 in cash. To claim your reward immediately, visit http://claim-cash-prize.example.com "
        "or reply with your full bank account details. Offer expires in 24 hours!"
    ),
    "🚨 Urgent Security / Account Phishing (Spam)": (
        "Subject: URGENT: Your account access has been suspended! "
        "We detected unauthorized login attempts from an unknown device. Please click http://verify-secure-login.example.com "
        "to verify your username and password immediately to prevent permanent account deletion."
    ),
    "✉️ Project Meeting & Status Update (Not Spam)": (
        "Subject: Project sync meeting agenda for Thursday afternoon\n\n"
        "Hi team, please find attached our agenda for Thursday's sprint review. "
        "We will review our Q3 metrics, discuss the model deployment timeline, and assign new action items. "
        "Let me know if you have any additional topics to include before noon."
    ),
    "✉️ Informal Coffee / Lunch Invitation (Not Spam)": (
        "Subject: Lunch catch-up tomorrow?\n\n"
        "Hey Alex, are you free to grab a quick coffee or lunch tomorrow around 12:30? "
        "I wanted to hear how your recent trip went and talk through the new documentation drafts. Let me know!"
    ),
}

# Sidebar Content
with st.sidebar:
    st.image("https://img.icons8.com/color/96/shield.png", width=64)
    st.title("System Overview")
    st.markdown(
        """
        **Spam Mail Detection System** classifies incoming email messages using natural language processing
        and probabilistic Naive Bayes classification.
        """
    )
    st.markdown("---")
    st.subheader("⚙️ Technical Architecture")
    st.markdown(
        """
        * **Classifier**: Multinomial Naive Bayes (`MultinomialNB`)
        * **Features**: TF-IDF (Term Frequency - Inverse Document Frequency)
        * **N-Grams**: Unigrams & Bigrams (1, 2)
        * **Framework**: Scikit-Learn `Pipeline`
        * **Dataset**: Enron Spam/Ham Corpus (~5,000 emails)
        """
    )

    metrics_data, metadata_data = load_metrics_and_metadata()
    if metrics_data:
        st.markdown("---")
        st.subheader("📊 Model Performance")
        st.metric("Test Accuracy", f"{metrics_data['accuracy'] * 100:.1f}%")
        st.metric("Spam Precision", f"{metrics_data['spam_precision'] * 100:.1f}%")
        st.metric("Spam Recall", f"{metrics_data['spam_recall'] * 100:.1f}%")

    st.markdown("---")
    st.caption("Built with Python 3.11, Scikit-Learn & Streamlit")


# Header
col_title, col_badge = st.columns([3, 1])
with col_title:
    st.markdown('<div class="main-header">🛡️ Spam Mail Detection System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Classify email text in real time using a Scikit-Learn TF-IDF and Naive Bayes pipeline.</div>',
        unsafe_allow_html=True,
    )
with col_badge:
    st.markdown("<br>", unsafe_allow_html=True)
    pipeline = load_pipeline()
    if pipeline is not None:
        st.markdown('<span class="badge-pill badge-ready">● Model Loaded & Active</span>', unsafe_allow_html=True)
    else:
        st.error("Model Not Found")

# Main Navigation Tabs
tab_scanner, tab_metrics, tab_architecture, tab_about = st.tabs([
    "🔍 Live Email Scanner",
    "📊 Evaluation Metrics",
    "🧠 How the Model Works",
    "ℹ️ About & Disclaimer",
])


# TAB 1: Live Email Scanner
with tab_scanner:
    if pipeline is None:
        st.warning(
            "⚠️ The trained model artifact (`models/spam_classifier.joblib`) was not found.\n\n"
            "To train and generate the model, run the following command in your terminal:\n\n"
            "```bash\npython src/train.py\n```"
        )
    else:
        st.markdown("### Test Any Email Message")

        # Quick preset selection
        def on_sample_change():
            selected_sample = st.session_state["sample_dropdown"]
            if selected_sample in SAMPLE_EMAILS and SAMPLE_EMAILS[selected_sample]:
                st.session_state["email_input"] = SAMPLE_EMAILS[selected_sample]

        col_sample, col_clear = st.columns([4, 1])
        with col_sample:
            st.selectbox(
                "Load a preset sample email:",
                options=list(SAMPLE_EMAILS.keys()),
                key="sample_dropdown",
                on_change=on_sample_change,
            )
        with col_clear:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🗑️ Clear Input", use_container_width=True):
                st.session_state["email_input"] = ""
                st.rerun()

        # Input text area
        email_text = st.text_area(
            "Paste or type email content here:",
            value=st.session_state.get("email_input", ""),
            height=180,
            placeholder="Paste raw email message body, subject line, or text here...",
            key="email_input",
        )

        btn_col, _ = st.columns([1, 3])
        with btn_col:
            predict_clicked = st.button("🚀 Analyze Email", type="primary", use_container_width=True)

        if predict_clicked:
            if not email_text or not email_text.strip():
                st.error("Please enter or paste an email message before clicking Analyze.")
            else:
                with st.spinner("Analyzing message with Naive Bayes pipeline..."):
                    prediction = pipeline.predict([email_text])[0]
                    probabilities = pipeline.predict_proba([email_text])[0]

                    # probabilities[0] = Ham (Not Spam), probabilities[1] = Spam
                    prob_ham = probabilities[0]
                    prob_spam = probabilities[1]

                    cleaned = clean_text(email_text)

                if prediction == 1:
                    st.markdown(
                        f"""
                        <div class="result-card-spam">
                            <h3 style="color: #b91c1c; margin-top: 0;">🚨 Predicted Category: SPAM MAIL</h3>
                            <p style="color: #7f1d1d; margin-bottom: 0.5rem; font-size: 1.05rem;">
                                This message exhibits linguistic, promotional, or structural patterns commonly found in unsolicited or fraudulent emails.
                            </p>
                            <p style="color: #991b1b; font-weight: 600; margin-bottom: 0;">
                                Confidence Score: {prob_spam * 100:.2f}% Spam Probability
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="result-card-ham">
                            <h3 style="color: #15803d; margin-top: 0;">✅ Predicted Category: NOT SPAM (LEGITIMATE)</h3>
                            <p style="color: #14532d; margin-bottom: 0.5rem; font-size: 1.05rem;">
                                This message appears to be normal, legitimate communication with characteristics typical of authentic correspondence.
                            </p>
                            <p style="color: #166534; font-weight: 600; margin-bottom: 0;">
                                Confidence Score: {prob_ham * 100:.2f}% Legitimate Probability
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.markdown("#### Probability Distribution")
                p_col1, p_col2 = st.columns(2)
                with p_col1:
                    st.metric("Legitimate (Not Spam) Likelihood", f"{prob_ham * 100:.2f}%")
                    st.progress(float(prob_ham))
                with p_col2:
                    st.metric("Spam Likelihood", f"{prob_spam * 100:.2f}%")
                    st.progress(float(prob_spam))

                with st.expander("🔍 Inspect Text Preprocessing & Tokens"):
                    st.markdown("**Cleaned & Normalized Representation:**")
                    st.code(cleaned if cleaned else "(empty after cleaning)")
                    st.markdown(
                        """
                        *Notice how URLs become `httpaddr`, currency symbols become `currencysymb`,
                        numbers become `numtoken`, and email headers are sanitized.*
                        """
                    )

        st.markdown(
            """
            <div class="disclaimer-box">
                <strong>Disclaimer:</strong> Predictions are statistical estimates generated by a machine-learning model.
                Always exercise human discretion before opening unknown attachments, clicking suspicious links, or sharing sensitive credentials.
            </div>
            """,
            unsafe_allow_html=True,
        )


# TAB 2: Evaluation Metrics
with tab_metrics:
    st.markdown("### Transparent Evaluation on Held-Out Test Data")
    st.markdown(
        """
        The model was evaluated on a held-out test split (20% stratified holdout)
        never seen during training or feature extraction.
        """
    )

    if metrics_data:
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.metric("Overall Accuracy", f"{metrics_data['accuracy'] * 100:.2f}%")
        with m_col2:
            st.metric("Spam Precision", f"{metrics_data['spam_precision'] * 100:.2f}%")
        with m_col3:
            st.metric("Spam Recall", f"{metrics_data['spam_recall'] * 100:.2f}%")
        with m_col4:
            st.metric("Spam F1-Score", f"{metrics_data['spam_f1'] * 100:.2f}%")

        st.markdown("---")

        cm_col, rep_col = st.columns([1, 1])
        with cm_col:
            st.markdown("#### Confusion Matrix Heatmap")
            if CONFUSION_MATRIX_PATH.exists():
                try:
                    st.image(str(CONFUSION_MATRIX_PATH), width="stretch")
                except TypeError:
                    st.image(str(CONFUSION_MATRIX_PATH), use_container_width=True)
            else:
                cm = metrics_data["confusion_matrix"]
                st.write(cm)

        with rep_col:
            st.markdown("#### Detailed Classification Breakdown")
            cm = metrics_data["confusion_matrix"]
            cm_table = pd.DataFrame(
                [
                    ["Actual Legitimate (Ham)", cm["true_negatives"], cm["false_positives"]],
                    ["Actual Spam", cm["false_negatives"], cm["true_positives"]],
                ],
                columns=["True Class", "Predicted Legitimate", "Predicted Spam"],
            )
            try:
                st.dataframe(cm_table, hide_index=True, width="stretch")
            except TypeError:
                st.dataframe(cm_table, hide_index=True, use_container_width=True)

            st.markdown(
                f"""
                - **True Negatives (TN):** `{cm['true_negatives']}` legitimate emails correctly preserved.
                - **False Positives (FP):** `{cm['false_positives']}` legitimate emails mistakenly flagged (False Alarms).
                - **False Negatives (FN):** `{cm['false_negatives']}` spam emails that bypassed the filter.
                - **True Positives (TP):** `{cm['true_positives']}` spam emails successfully caught and blocked.
                """
            )
    else:
        st.info("Metrics not found yet. Train the model using `python src/train.py` to populate evaluation statistics.")


# TAB 3: How the Model Works
with tab_architecture:
    st.markdown("### How the Machine Learning Pipeline Works")

    st.markdown(
        r"""
        #### 1. Text Preprocessing & Cleaning
        Raw email text is cleaned and standardized:
        * **Normalization**: Case folded to lowercase, redundant whitespace collapsed.
        * **Entity Abstraction**: URLs (`httpaddr`), emails (`emailaddr`), currency symbols (`currencysymb`), and numbers (`numtoken`) are normalized into generic tokens.
        * **Noise Reduction**: Header prefixes such as `Subject:` and punctuation are stripped while preserving vocabulary semantics.

        #### 2. TF-IDF Feature Extraction (Term Frequency - Inverse Document Frequency)
        Raw text is converted into high-dimensional numerical vectors:
        $$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$
        * **TF (Term Frequency)**: Frequency of term $t$ in email $d$.
        * **IDF (Inverse Document Frequency)**: Downweights ubiquitous words (like 'the', 'is', 'for') and elevates distinctive spam markers (like 'lottery', 'winner', 'urgent').
        * **Sublinear Scaling**: Logarithmic term frequency $\text{TF} = 1 + \log(\text{tf})$ prevents long emails from dominating.

        #### 3. Multinomial Naive Bayes Classification
        Naive Bayes computes the posterior probability using Bayes' Theorem under the conditional independence assumption:
        $$P(\text{Spam} \mid \mathbf{w}) \propto P(\text{Spam}) \prod_{i=1}^{n} P(w_i \mid \text{Spam})$$
        * **Laplace Smoothing ($\alpha=0.1$)**: Prevents zero-probability penalties for novel words unseen in the training vocabulary.
        * **Efficiency & Speed**: Naive Bayes offers microsecond inference latency and scales efficiently on sparse text matrices.

        #### 4. Understanding Precision vs. Recall in Spam Detection
        * **Precision (Spam)**: When the filter flags an email as Spam, how often is it truly Spam?
          $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
          *High precision is paramount to avoid sending important business or personal emails to the junk folder.*
        * **Recall (Spam)**: What fraction of all spam in circulation was intercepted?
          $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
          *High recall ensures the user's primary inbox remains clean and protected.*
        """
    )


# TAB 4: About & Disclaimer
with tab_about:
    st.markdown("### About This Project")
    st.markdown(
        """
        This system is a reconstructed, production-grade machine-learning email spam classifier.
        It features automated data preprocessing, reproducible Scikit-Learn pipelines, transparent evaluation,
        and an interactive user interface built for Streamlit Community Cloud.

        #### Project Structure
        ```text
        spamMailDetection/
        ├── data/                    # Dataset storage & documentation
        ├── models/                  # Serialized Joblib pipeline & evaluation artifacts
        ├── notebooks/               # Exploratory Data Analysis (EDA) Jupyter Notebook
        ├── src/                     # Modular Python code (preprocessing, train, evaluate)
        ├── tests/                   # Pytest automated testing suite
        ├── app.py                   # Streamlit web application
        ├── requirements.txt         # Dependencies
        └── README.md                # Documentation & deployment guide
        ```

        #### Ethical & Operational Disclaimer
        * Machine-learning classifiers provide probabilistic predictions based on historical training data.
        * Attackers continuously evolve spam templates (adversarial word mutations, homoglyphs, image-based text).
        * No single machine-learning model replaces defense-in-depth cybersecurity practices (SPF/DKIM/DMARC checks, multi-factor authentication, and safe browsing habits).
        """
    )
