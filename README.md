# Spam Mail Detection System

An end-to-end, production-minded machine learning system for detecting email spam. Recreated using **Python 3.11**, **Scikit-Learn**, and **Streamlit**, this project features a reproducible text preprocessing and classification pipeline, transparent evaluation metrics on a held-out test set, and an interactive web application ready for deployment to **Streamlit Community Cloud**.

---

## Table of Contents

- [Overview & Architecture](#overview--architecture)
- [Key Features](#key-features)
- [Dataset Details](#dataset-details)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Evaluation & Transparent Metrics](#evaluation--transparent-metrics)
  - [Understanding the Metrics](#understanding-the-metrics)
  - [Held-Out Test Results](#held-out-test-results)
  - [Why Results Differ from the Original ~98% Academic Claim](#why-results-differ-from-the-original-98-academic-claim)
- [Project Directory Structure](#project-directory-structure)
- [Local Installation & Execution](#local-installation--execution)
  - [macOS and Linux](#macos-and-linux)
  - [Windows (PowerShell)](#windows-powershell)
- [Running Automated Tests](#running-automated-tests)
- [Streamlit Community Cloud Deployment Guide](#streamlit-community-cloud-deployment-guide)
  - [Step 1: Push to GitHub](#step-1-push-to-github)
  - [Step 2: Connect to Streamlit Community Cloud](#step-2-connect-to-streamlit-community-cloud)
  - [Step 3: Configure Deployment](#step-3-configure-deployment)
  - [Step 4: Deploy & Verify](#step-4-deploy--verify)
  - [Troubleshooting Common Deployment Issues](#troubleshooting-common-deployment-issues)
- [License](#license)

---

## Overview & Architecture

Spam email filtering is one of the classic applications of machine learning. The system ingests raw email text, transforms linguistic features into numerical representations using **TF-IDF (Term Frequency - Inverse Document Frequency)**, and applies **Multinomial Naive Bayes** to calculate the posterior probability of an email being Spam vs. Legitimate (Ham).

```
┌─────────────────┐       ┌───────────────────────┐       ┌────────────────────────┐
│  Raw Email Text │ ───►  │ Text Preprocessing    │ ───►  │  TF-IDF Vectorization  │
│  (Input/Dataset)│       │ • Lowercasing         │       │  • (1, 2) N-Grams      │
└─────────────────┘       │ • URL/Email/Currency  │       │  • Sublinear TF        │
                          │ • Noise Reduction     │       │  • 10,000 Max Features │
                          └───────────────────────┘       └───────────┬────────────┘
                                                                      │
                                                                      ▼
┌──────────────────────────┐       ┌────────────────────────┐       ┌────────────────────────┐
│   Streamlit Web App      │ ◄──── │ Joblib Pipeline Export │ ◄──── │ Multinomial Naive Bayes│
│ • Real-time Scanner      │       │ (spam_classifier.joblib│       │ • Laplace Smoothing    │
│ • Probability Breakdown  │       │  fitted pipeline)      │       │   (alpha = 0.1)        │
└──────────────────────────┘       └────────────────────────┘       └────────────────────────┘
```

---

## Key Features

- **Reproducible Pipeline**: Uses `sklearn.pipeline.Pipeline` bundling text cleaning, feature extraction, and classification into a single serializable object.
- **Robust Schema Mapping**: Auto-detects text and label columns across diverse datasets (e.g. `message`, `text`, `label`, `v1`, `v2`, `label_num`).
- **Data Integrity**: Deduplicates training records to prevent data leakage and handles missing values cleanly.
- **Microsecond Inference**: Lightweight serialized Joblib pipeline (<700 KB) loads in milliseconds.
- **Modern Interactive UI**: Streamlit application with preset samples, probability meters, token inspection, evaluation charts, and architecture explainers.
- **Streamlit Community Cloud Ready**: Pinned dependencies, relative paths, and cached model loading (`@st.cache_resource`).

---

## Dataset Details

The system is trained on the publicly available **Enron Spam/Ham Dataset** (curated by Venky73 on Kaggle):
- **Source**: [Kaggle Spam Mails Dataset (Enron Corpus)](https://www.kaggle.com/datasets/venky73/spam-mails-dataset)
- **Total Records**: 5,171 emails
- **Unique Records After Deduplication**: 4,991 valid emails (178 duplicates and empty texts removed)
- **Class Distribution**:
  - **Not Spam (Ham)**: 3,531 emails (~70.7%)
  - **Spam**: 1,460 emails (~29.3%)
- **Data File**: `data/spam_ham_dataset.csv`
- See [`data/README.md`](file:///Users/sahil/Repositories/spamMailDetection/data/README.md) for full schema documentation and manual download instructions.

---

## Machine Learning Pipeline

1. **Text Preprocessing (`src/data_preprocessing.py`)**:
   - Converts text to lowercase.
   - Strips email header artifacts (e.g. `Subject:`, `Re:`, `Fwd:`).
   - Replaces URLs with a canonical token (`httpaddr`).
   - Replaces email addresses with a canonical token (`emailaddr`).
   - Replaces currency symbols (`$`, `€`, `£`, `¥`, `₹`) with `currencysymb`.
   - Replaces numeric values with `numtoken`.
   - Strips punctuation and excessive whitespace.
2. **Feature Extraction**:
   - `TfidfVectorizer` extracting word unigrams and bigrams (`ngram_range=(1, 2)`).
   - Sublinear term-frequency scaling (`sublinear_tf=True`) to dampen the effect of unusually long emails.
   - Vocabulary capped at `10,000` features.
3. **Classification**:
   - `MultinomialNB(alpha=0.1)` with additive Laplace smoothing to handle words outside the training vocabulary.
4. **Serialization**:
   - Exported using Joblib to `models/spam_classifier.joblib`.

---

## Evaluation & Transparent Metrics

### Understanding the Metrics

When evaluating email spam filters, relying solely on **Accuracy** can be misleading due to class imbalance and the differing real-world consequences of errors:

1. **Accuracy**:
   $$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{Total Predictions}}$$
   The proportion of all emails (both spam and legitimate) correctly categorized. While useful as a high-level summary, accuracy treats all errors equally.
2. **Precision (Spam Class)**:
   $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
   Out of all emails flagged by the model as *Spam*, what fraction were truly spam?
   *Why it matters:* **False Positives (FP)** occur when a legitimate, critical email (such as a job offer, invoice, or meeting request) is mistakenly labeled spam and sent to the junk folder. High precision prevents legitimate emails from being lost.
3. **Recall (Spam Class)**:
   $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
   Out of all actual spam emails in circulation, what fraction did the filter successfully catch?
   *Why it matters:* **False Negatives (FN)** represent spam emails that bypass the filter and reach the user's primary inbox.
4. **F1-Score**:
   $$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
   The harmonic mean of precision and recall, balancing false alarms against missed spam.

---

### Held-Out Test Results

The model was evaluated on a held-out test set (20% stratified split, `random_state=42`, $N=999$ test samples) that was **never exposed to the vectorizer or classifier during training**:

| Metric | Score | Description |
| :--- | :--- | :--- |
| **Overall Accuracy** | **95.10%** | 950 out of 999 emails correctly classified |
| **Spam Recall** | **96.92%** | Caught 283 out of 292 spam emails (only 9 slipped through) |
| **Spam Precision** | **87.62%** | 283 true spam out of 323 predicted spam |
| **Spam F1-Score** | **92.03%** | Harmonic balance between spam precision and recall |
| **Ham Precision** | **98.67%** | When labeled legitimate, 98.7% truly legitimate |
| **Ham Recall** | **94.34%** | 667 out of 707 legitimate emails correctly kept |

#### Confusion Matrix (Held-Out Test Split)

| | Predicted Legitimate (Ham) | Predicted Spam |
| :--- | :---: | :---: |
| **Actual Legitimate (Ham)** | **667** (True Negatives) | **40** (False Positives) |
| **Actual Spam** | **9** (False Negatives) | **283** (True Positives) |

---

### Why Results Differ from the Original ~98% Academic Claim

In many introductory academic projects, ~98% accuracy is frequently cited for Naive Bayes spam classification. Transparently examining our 95.10% result reveals several clear technical reasons:

1. **Dataset Differences (Email vs. SMS)**:
   Many academic projects achieve ~98.5% on the **SMS Spam Collection** dataset. SMS messages are brief (under 160 characters) and feature dense, highly repetitive spam triggers (`FREE`, `WIN`, `CALL NOW`). In contrast, the **Enron Email Corpus** consists of full-length corporate emails with complex conversational text, technical jargon, meeting notes, and varied vocabulary, making classification more challenging and realistic.
2. **Deduplication and Data Leakage**:
   The raw Enron dataset contains 178 duplicate messages. In standard classroom tutorials, train/test splits are often performed *without* deduplication. When identical emails exist in both train and test splits, the model memorizes test samples, artificially inflating accuracy scores to 97–98%. By deduplicating first, our evaluation strictly measures true generalization.
3. **Preprocessing & Feature Representation**:
   Differences in tokenization (handling numbers, email addresses, and punctuation), vocabulary size thresholds, and whether stopwords are filtered impact the precision-recall balance.
4. **Hyperparameter Tuning & Thresholding**:
   Our model uses Laplace smoothing $\alpha=0.1$, which prioritizes high spam recall (96.92%) so that almost no spam reaches the inbox. Increasing $\alpha$ or adjusting decision thresholds shifts the balance toward higher precision at the expense of recall.

---

## Project Directory Structure

```text
spamMailDetection/
│
├── data/
│   ├── spam_ham_dataset.csv        # Dataset (Enron corpus)
│   └── README.md                   # Dataset documentation and schema
│
├── models/
│   ├── spam_classifier.joblib      # Serialized Scikit-Learn Pipeline
│   ├── metrics.json                # Exported evaluation metrics
│   ├── model_metadata.json         # Training hyperparams and timestamps
│   └── confusion_matrix.png        # Confusion matrix heatmap plot
│
├── notebooks/
│   └── exploratory_analysis.ipynb  # Interactive EDA, plots, and feature exploration
│
├── src/
│   ├── __init__.py                 # Package initializer
│   ├── data_preprocessing.py       # Cleaning, schema mapping, and data splitting
│   ├── train.py                    # Pipeline construction, training, and export
│   └── evaluate.py                 # Evaluation metrics, confusion matrix, and reporting
│
├── tests/
│   └── test_pipeline.py            # Unit tests for preprocessing and inference
│
├── app.py                          # Streamlit interactive web application
├── requirements.txt                # Pinned dependencies
├── pytest.ini                      # Pytest configuration
├── .gitignore                      # Git ignore rules
├── LICENSE                         # MIT License
└── README.md                       # Complete documentation & deployment guide
```

---

## Local Installation & Execution

### macOS and Linux

1. **Clone the repository**:
   ```bash
   git clone https://github.com/<your-username>/spam-mail-detection.git
   cd spam-mail-detection
   ```

2. **Create and activate a virtual environment (Python 3.11 recommended)**:
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model (if retrained or downloading dataset)**:
   ```bash
   python src/train.py
   ```

5. **Launch the Streamlit web application**:
   ```bash
   streamlit run app.py
   ```
   Open your browser at `http://localhost:8501`.

---

### Windows (PowerShell)

1. **Clone the repository**:
   ```powershell
   git clone https://github.com/<your-username>/spam-mail-detection.git
   cd spam-mail-detection
   ```

2. **Create and activate a virtual environment**:
   ```powershell
   py -3.11 -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

4. **Train the model**:
   ```powershell
   python src\train.py
   ```

5. **Run the Streamlit web app**:
   ```powershell
   streamlit run app.py
   ```

---

## Running Automated Tests

Run the test suite using `pytest`:

```bash
pytest tests/ -v
```

Expected output:
```text
tests/test_pipeline.py::test_clean_text_basic PASSED
tests/test_pipeline.py::test_clean_text_edge_cases PASSED
tests/test_pipeline.py::test_detect_columns PASSED
tests/test_pipeline.py::test_normalize_labels PASSED
tests/test_pipeline.py::test_split_dataset PASSED
tests/test_pipeline.py::test_saved_model_inference PASSED
============================== 6 passed in 1.47s ===============================
```

---

## Streamlit Community Cloud Deployment Guide

Follow these steps to deploy the application for free to **Streamlit Community Cloud**:

### Step 1: Push to GitHub

1. Create a new repository on [GitHub](https://github.com/new) named `spam-mail-detection` (public repository recommended).
2. In your local project directory, initialize Git and commit the files:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: complete spam mail detection pipeline and Streamlit app"
   ```
3. Link your remote repository and push to the `main` branch:
   ```bash
   git branch -M main
   git remote add origin https://github.com/<your-username>/spam-mail-detection.git
   git push -u origin main
   ```
   > **Note on Model Size**: The trained model artifact (`models/spam_classifier.joblib`) is approximately **693 KB**, well below GitHub's 100 MB file limit. You can safely commit and push it directly without Git LFS.

---

### Step 2: Connect to Streamlit Community Cloud

1. Navigate to [share.streamlit.io](https://share.streamlit.io/).
2. Click **Sign in with GitHub** and authorize Streamlit.

---

### Step 3: Configure Deployment

1. Click the **"New app"** button in your Streamlit Cloud workspace.
2. Configure the deployment settings:
   - **Repository**: `<your-username>/spam-mail-detection`
   - **Branch**: `main`
   - **Main file path**: `app.py`
3. Click **"Advanced settings"** (optional):
   - Ensure the Python version is set to **3.11**.

---

### Step 4: Deploy & Verify

1. Click **"Deploy!"**.
2. Streamlit Cloud will install dependencies from `requirements.txt` and launch the app.
3. Once the build finishes, your app will be live with a public URL (e.g. `https://<your-app-name>.streamlit.app`).
4. Test with preset samples to verify predictions and probability scores.

---

### Troubleshooting Common Deployment Issues

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| **`ModuleNotFoundError: No module named 'src'`** | Python path does not include project root | `app.py` automatically injects the project root via `sys.path.insert(0, str(Path(__file__).resolve().parent))`. |
| **`FileNotFoundError: models/spam_classifier.joblib`** | Model file was omitted from Git commit | Verify that `models/spam_classifier.joblib` is tracked by git (`git status`). Run `python src/train.py` locally and commit the file. |
| **`InconsistentVersionWarning` or unpickling error** | Mismatched Scikit-Learn / Python versions | Ensure `requirements.txt` contains compatible versions (`scikit-learn>=1.3.0`, `joblib>=1.3.0`) and match the Streamlit Cloud Python version to 3.11. |
| **Slow startup time** | Model reloads on every user interaction | The app wraps model loading with `@st.cache_resource`, ensuring the pipeline is loaded into memory only once. |

---

## License

This project is licensed under the [MIT License](LICENSE).
