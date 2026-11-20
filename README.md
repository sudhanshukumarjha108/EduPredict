# EduPredict – Student Performance Prediction and Explainable Analytics

<p align="center">
  <a href="https://edupredict-sudhanshsu.streamlit.app/">
    <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Open in Streamlit">
  </a>
</p>

<p align="center">
  An Explainable Machine Learning Application for Early Student Performance Prediction
</p>

## Live Demo

**Application:** https://edupredict-sudhanshsu.streamlit.app/

**GitHub:** https://github.com/sudhanshukumarjha108/EduPredict

## About the Project

EduPredict is a machine learning-based student performance prediction and explainable analytics application. It aims to identify students who may be at academic risk using background, family, and behavioral information available during the early stage of an academic semester.

The project compares multiple classification algorithms and uses explainable AI techniques to help interpret model predictions and influential factors.

## Key Features

- Student performance prediction: Pass / At-Risk classification.
- Early-stage prediction without using first and second-period grades (G1 and G2).
- Comparison of Logistic Regression, Decision Tree, and Random Forest.
- Model evaluation using a stratified train-test split.
- Five-fold cross-validation on training data.
- Confusion matrix and classification metrics.
- SHAP-based global and individual prediction explanations.
- Interactive Streamlit dashboard.
- Prediction for newly entered student information.

## Machine Learning Methodology

**Dataset:** UCI Student Performance – Mathematics (`student-mat.csv`).

**Target definition:**
- Pass: Final grade (G3) >= 10
- At-Risk: Final grade (G3) < 10

**Early-stage feature selection:** G1 and G2 are excluded to avoid using intermediate grades that may not be available during early-semester prediction.

**Models evaluated:**
1. Logistic Regression
2. Decision Tree
3. Random Forest

**Evaluation approach:**
- Stratified train-test split (75:25 default configuration).
- Five-fold stratified cross-validation on the training partition.
- Accuracy, balanced accuracy, precision, recall, and F1-score.

## Test Set Results

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Logistic Regression | 59.60% | 55.20% |
| Decision Tree | 61.62% | 59.00% |
| Random Forest | **70.71%** | **65.65%** |

Random Forest achieved the highest test accuracy and macro F1-score among the three evaluated models in this experiment.

## Explainable AI

EduPredict incorporates SHAP (SHapley Additive exPlanations) to interpret model predictions.

- Global explanations: Understand which features influence predictions across students.
- Individual explanations: Examine factors contributing to an individual student's prediction.

## Technology Stack

- **Programming:** Python
- **Machine Learning:** Scikit-learn
- **Explainability:** SHAP
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Application:** Streamlit

## Project Structure

```text
EduPredict_Project/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── student-mat.csv
│
└── .streamlit/
    └── config.toml
```

## Installation and Execution

### 1. Clone the repository

```bash
git clone https://github.com/sudhanshukumarjha108/EduPredict.git
cd EduPredict
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in your terminal.

## Research Context

This project is associated with the research topic:

**Explainable Machine Learning for Student Performance Prediction: A Comparative Analysis of Predictive Models and Influential Factors.**

The study investigates predictive performance and the interpretability of machine learning models for early identification of students who may be academically at risk.

## Limitations

- The experiments use the UCI Student Performance Mathematics dataset.
- Results on this dataset may not generalize to other institutions or student populations.
- Early-stage prediction excludes G1 and G2, but the target is still the final grade G3.
- Model predictions should support educational intervention, not replace academic judgment.

## Authors

- Sudhanshu Kumar Jha
- Riya Kumari
- Shivani Rana

**Galgotias University**  
School of Computer Applications and Technology

## Acknowledgements

Dataset: UCI Machine Learning Repository – Student Performance.

