# EduPredict — UCI Student Performance Mathematics

This version uses the UCI Student Performance Mathematics dataset (`student-mat.csv`), not OULAD.

## Dataset placement
Place the UCI Mathematics CSV in the `data` folder as one of:
- `data/student-mat.csv` (recommended)
- `data/student_mat.csv`
- `data/Original_Dataset.csv`

The standard UCI file is semicolon-delimited. The app reads it as such and falls back to comma-delimited parsing if needed.

## Target and predictors
- Target: `G3 >= 10` is Pass; `G3 < 10` is At-Risk.
- `G3` is not used as a predictor.
- `G1` and `G2` are used as earlier-period grades.
- The predictors include family size, parents' education (`Medu`, `Fedu`), parents' occupations (`Mjob`, `Fjob`), guardian, family support, school support, study habits, social habits, absences, and previous grades.

## Install and run
Use Python 3.12 (64-bit) and a virtual environment if possible.

```powershell
python -m pip install streamlit pandas numpy scikit-learn plotly matplotlib shap
python -m streamlit run app.py
```

## Evaluation
- Stratified train/test split (default 75:25).
- Logistic Regression, Decision Tree, Random Forest.
- Stratified cross-validation on training data only.
- Final metrics on the held-out test set.
- New manually entered students are inference-only and do not affect training or test metrics.
