# 🛡️ Self-Healing Machine Learning Platform

**An end-to-end ML platform for reliable model training, evaluation, monitoring, and automated recovery.**

The Self-Healing Machine Learning Platform aims to improve the reliability of machine learning systems through data validation, model monitoring, drift detection, controlled retraining, and safe model promotion and rollback.

**Initial Use Case:** Customer Churn Prediction using the IBM Telco Customer Churn dataset.

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-scikit--learn-orange?logo=scikitlearn)
![Testing](https://img.shields.io/badge/Testing-pytest-green?logo=pytest)
![Status](https://img.shields.io/badge/Status-Active%20Development-yellow)

---

## 📌 Project Status

**Current Phase: Core ML Pipeline and Reusable Inference Implemented**

The project currently includes data ingestion, data validation, baseline model training, multi-model comparison, hyperparameter tuning, and test-set evaluation.

Advanced monitoring, automated retraining, champion–challenger evaluation, safe model promotion, rollback, API serving, and dashboard functionality are planned for future development.

## 🎯 Objectives

- Build a reproducible end-to-end machine learning pipeline.
- Validate incoming data and identify data quality issues.
- Establish baseline model performance.
- Compare multiple ML algorithms using cross-validation.
- Optimize model performance through hyperparameter tuning.
- Evaluate models using appropriate classification metrics.
- Detect data drift and model degradation.
- Trigger controlled model retraining.
- Compare candidate models against the current champion.
- Support safe model promotion and rollback.
- Provide predictions through an API and monitoring dashboard.

## ✨ Implemented Features

### 1. Data Ingestion and Validation

- Load the IBM Telco Customer Churn dataset from the local data directory.
- Validate the dataset schema and required columns.
- Detect empty datasets and duplicate customer IDs.
- Validate target labels and numeric values.
- Identify missing values and invalid business-rule conditions.
- Handle whitespace-only values during validation.
- Apply tenure-based validation rules for missing `TotalCharges`.
- Run automated validation tests using `pytest`.

### 2. Baseline Model Training

- Formulate customer churn prediction as a binary classification problem.
- Encode the target variable as `No → 0` and `Yes → 1`.
- Exclude customer IDs from model features.
- Split data into training and testing sets using stratification.
- Build a preprocessing pipeline using `ColumnTransformer`.
- Apply median imputation and standard scaling to numerical features.
- Apply most-frequent imputation and one-hot encoding to categorical features.
- Train a class-balanced Logistic Regression model.
- Save the trained model using `joblib`.

### 3. Multi-Model Comparison

Compare six classification algorithms using five-fold stratified cross-validation.

| Algorithm | Purpose |
|---|---|
| Logistic Regression | Linear classification baseline |
| Support Vector Machine (SVM) | Margin-based classification |
| Random Forest | Ensemble of decision trees |
| Gradient Boosting | Sequential boosting ensemble |
| Extra Trees | Randomized tree ensemble |
| Decision Tree | Rule-based classification |

Models are compared using F1-score, precision, recall, ROC-AUC, and execution time.

**Output:** `reports/model_comparison.csv`

### 4. Hyperparameter Tuning

Use `RandomizedSearchCV` with stratified cross-validation to tune selected hyperparameters for:

- Logistic Regression
- Support Vector Machine
- Gradient Boosting

The tuned models and tuning results are saved for further evaluation.

**Output:** `reports/tuning_results.csv`

### 5. Test-Set Evaluation

Evaluate the baseline and tuned Logistic Regression models using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

**Output:** `reports/final_test_results.json`


### 6. Reusable Prediction Module

**File:** `src/self_healing_ml/models/predict.py`

- Load the tuned Logistic Regression pipeline.
- Accept customer details as a Python dictionary.
- Check for missing required features.
- Convert numeric inputs into numeric values.
- Return a predicted churn label (`Yes` or `No`).
- Return the predicted churn probability.

The module provides a reusable inference function through
`predict_churn(customer_data)`.



## 📊 Initial Model Results

The following results were obtained using the current stratified 80/20 train-test split.

| Metric | Baseline Logistic Regression | Tuned Logistic Regression |
|---|---:|---:|
| Accuracy | 73.81% | 74.24% |
| Precision | 50.43% | 50.95% |
| Recall | 78.34% | 78.61% |
| F1-score | 61.36% | 61.83% |
| ROC-AUC | 84.13% | 84.09% |

### Key Observations

- Hyperparameter tuning produced modest improvements in accuracy, precision, recall, and F1-score.
- ROC-AUC decreased slightly after tuning.
- The tuned Logistic Regression model is a **provisional candidate**, not an automatically promoted production model.
- Model selection involves trade-offs between identifying churn-risk customers and limiting false-positive predictions.

> **Note:** These are development-stage results. Because the same test split has been examined during development, a fresh holdout or external dataset is needed for a more rigorous final performance assessment.


## 🧪 Automated Testing

The project uses `pytest` to verify data validation and prediction behavior.

**Test files:**
- `tests/test_validation.py` — tests for dataset validation rules.
- `tests/test_predictions.py` — tests for the reusable prediction module.

Run the complete test suite from the project root:

```powershell
python -m pytest -v
```

**Latest test result:** 13 tests passed.


## 🧰 Technology Stack

### Currently Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| scikit-learn | Preprocessing, model training, cross-validation, and evaluation |
| joblib | Model serialization |
| pytest | Automated testing |
| Git and GitHub | Version control and collaboration |

### Planned Integrations

| Technology | Intended Purpose |
|---|---|
| MLflow | Experiment tracking and model management |
| Evidently | Data and model monitoring |
| FastAPI | Model serving and prediction endpoints |
| Streamlit | Monitoring dashboard |
| GitHub Actions | Continuous integration and delivery |
| Docker | Containerization and reproducible deployment |

## 🏗️ Project Architecture

```text
SelfHealingMLPlatform
        |
        v
  Data Ingestion
        |
        v
  Data Validation
        |
        v
  Data Preprocessing
        |
        v
  Baseline Model Training
        |
        v
  Model Comparison
        |
        v
  Hyperparameter Tuning
        |
        v
  Test-Set Evaluation
        |
        v
  Reusable Prediction Module
        |
        v
  Monitoring and Drift Detection
        |
        v
  Automated Retraining
        |
        v
  Champion-Challenger Evaluation
        |
        v
  Safe Model Promotion / Rollback
        |
        v
  API and Monitoring Dashboard
```

*The initial stages are implemented. The later stages represent the planned architecture.*

## 📂 Project Structure

```text
SelfHealingMLPlatform/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
├── reports/
├── logs/
├── tests/
│   ├── test_validation.py
│   └── test_predictions.py
│
├── src/
│   └── self_healing_ml/
│       ├── data/
│       │   ├── ingestion.py
│       │   └── validation.py
│       │
│       ├── models/
│       │   ├── train.py
│       │   ├── compare_models.py
│       │   ├── tune_models.py
│       │   └── evaluate_models.py
│       │   └── predict.py
│       │
│       ├── monitoring/
│       ├── orchestration/
│       └── api/
│
├── requirements.txt
├── .gitignore
└── README.md
```

*The monitoring, orchestration, and API directories are reserved for planned components.*

## 🚀 Getting Started

### Prerequisites

- Python 3.x
- Git
- Visual Studio Code or another Python-compatible editor

### 1. Clone the Repository

```bash
git clone https://github.com/SanjanaRawal/SelfHealingMLPlatform.git
cd SelfHealingMLPlatform
```

### 2. Create a Virtual Environment

For Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Add the Dataset

Place the IBM Telco Customer Churn CSV file at:

```text
data/raw/Telco-Customer-Churn.csv
```

Ensure the dataset is obtained from a suitable source and its usage terms are respected.

### 5. Run the Tests

From the project root:

```powershell
python -m pytest -v
```

### 6. Run the ML Pipeline

Run the modules from the project root, in order:

```powershell
python -m src.self_healing_ml.models.train
python -m src.self_healing_ml.models.compare_models
python -m src.self_healing_ml.models.tune_models
python -m src.self_healing_ml.models.evaluate_models
```

These commands assume the modules support `python -m` execution and that the dataset, dependencies, and required artifacts are available.

## 🗺️ Development Roadmap

- [x] Project scaffold and version control
- [x] Data ingestion and validation
- [x] Automated data validation tests
- [x] Baseline model training
- [x] Multi-model comparison using cross-validation
- [x] Hyperparameter tuning
- [x] Baseline versus tuned-model evaluation
- [x] Reusable prediction and inference module
- [ ] Prediction logging and monitoring
- [ ] Data drift detection
- [ ] Model performance monitoring
- [ ] Automated retraining triggers
- [ ] Champion–challenger model evaluation
- [ ] Model registry and controlled promotion
- [ ] Model rollback mechanism
- [ ] FastAPI prediction service
- [ ] Streamlit monitoring dashboard
- [ ] CI/CD pipeline and automated tests
- [ ] Docker-based deployment

## 🔐 Engineering Principles

- **Reproducibility:** Use controlled random seeds and consistent data splits.
- **Data Quality:** Validate data before it reaches the training pipeline.
- **Leakage Prevention:** Fit preprocessing steps within training and cross-validation pipelines.
- **Evaluation Discipline:** Use cross-validation for model selection and held-out data for evaluation.
- **Modularity:** Separate ingestion, validation, modeling, monitoring, orchestration, and serving.
- **Safe Automation:** Require evaluation criteria before promoting a candidate model.
- **Maintainability:** Keep the code testable, documented, and version-controlled.

## 🔮 Future Vision

The long-term goal is to build a machine learning platform that can identify data quality issues and changes in incoming data, monitor model behavior, trigger controlled retraining, evaluate new model candidates, and support safe promotion or rollback.

The platform is being developed incrementally, with an emphasis on reproducibility, reliability, and practical MLOps engineering.

---

**Project:** Self-Healing Machine Learning Platform  
**Use Case:** Customer Churn Prediction  
**Domain:** Machine Learning · MLOps · Model Monitoring · Automation  
**Status:** Active Development