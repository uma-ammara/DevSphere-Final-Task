# SkillMatch: Resume-Based Job Category Prediction

SkillMatch is a machine learning project that reads the text of a resume and predicts the job category it belongs to. It uses **TF-IDF** for text features and compares **Multinomial Naive Bayes** and **Logistic Regression**. The best model is saved as a single pipeline and used by a **Streamlit** web app where anyone can paste a resume and get a prediction.

This project uses traditional machine learning only. No CNN or deep learning is used.

---

## Features

- Cleans raw resume text (removes HTML, links, emails, numbers and punctuation)
- Trains and compares two classifiers, plus a tuned Logistic Regression
- Evaluates with accuracy, precision, recall, F1-score and a confusion matrix
- Saves the complete TF-IDF + classifier pipeline as `skillmatch_model.pkl`
- Streamlit interface with sample resumes, `.txt` upload, top-3 predictions and confidence bars

---

## Dataset

The project uses the Resume Dataset CSV (`Resume.csv`).

| Column | Description |
|---|---|
| `ID` | Resume identifier (not used for training) |
| `Resume_str` | Plain resume text (model input) |
| `Resume_html` | Resume in HTML (not used for training) |
| `Category` | Job category label (model target) |

- **Rows loaded:** 2,484
- **Rows used after cleaning:** 2,481 (2 duplicates removed, 1 resume empty after cleaning)
- **Categories:** 24 (for example Information Technology, HR, Chef, Accountant, Engineering, Banking)
- **Small categories:** Agriculture (63), Automobile (36) and BPO (22) have far fewer resumes than the rest

---

## Methodology

1. Load `Resume.csv` and keep `Resume_str` and `Category`
2. Remove missing values and duplicate resumes
3. Clean the text and convert it to lowercase
4. Visualize the job category distribution
5. Split the data 80% training (1,984) and 20% testing (497), stratified with `random_state=42`
6. Extract features with `TfidfVectorizer` (English stop words, unigrams and bigrams, up to 20,000 features, sublinear TF)
7. Train Multinomial Naive Bayes and Logistic Regression as scikit-learn pipelines
8. Tune Logistic Regression with `GridSearchCV` (3-fold cross-validation on training data only)
9. Evaluate on the test set and plot a confusion matrix
10. Save the best pipeline with `joblib`

---

## Results

All scores are measured on the 497 held-out test resumes. Precision, recall and F1-score are weighted averages.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Multinomial Naive Bayes | 0.5352 | 0.5520 | 0.5352 | 0.4872 |
| Logistic Regression | 0.6378 | 0.6661 | 0.6378 | 0.6093 |
| **Logistic Regression (tuned)** | **0.6901** | **0.7053** | **0.6901** | **0.6814** |

**Best model:** Logistic Regression (tuned) with `C=50` and `class_weight="balanced"`. It is the model saved in `model/skillmatch_model.pkl`.

**Notes**

- Very small categories such as BPO and Automobile are hard to predict because they have few examples.
- Categories with similar wording (for example Consultant, Business Development and Sales) are harder to separate.

---

## Project Structure

```
DevSphere-Final-Task/
├── app.py                      # Streamlit web application
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── model/
│   └── skillmatch_model.pkl    # Saved TF-IDF + Logistic Regression pipeline
└── data/
    └── Resume.csv              # Dataset used for training
```

---

## Installation and Usage

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/DevSphere-Final-Task.git
cd DevSphere-Final-Task
```

### 2. Install the requirements

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

The app opens in your browser (usually at `http://localhost:8501`). Paste a resume, or pick a sample, and click **Predict Category**.

### Using the saved model directly

```python
import joblib

model = joblib.load("model/skillmatch_model.pkl")
print(model.predict(["hr manager with experience in recruitment payroll and onboarding"]))
```

For best results, clean the text first (lowercase, letters only) as done in `app.py`.

---

## Requirements

```
streamlit
scikit-learn
joblib
numpy
pandas
```

The model should be loaded with the **same scikit-learn version** that was used to train it. If you see a version warning, pin that version (for example `scikit-learn==1.6.1`) in `requirements.txt`.

---



## Tech Stack

Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn, joblib, Streamlit, Google Colab, GitHub

---

## Limitations and Future Work

- The dataset is small, and some categories have very few resumes
- Word-frequency features cannot capture meaning, so similar categories can be confused
- Possible improvements: more data for small categories, other traditional models (Linear SVM, Random Forest), and PDF resume upload

---

