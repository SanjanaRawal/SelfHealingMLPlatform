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

**Current Status:** Core ML Pipeline, Reusable Inference, Data Drift Detection, Performance Monitoring, Prediction Logging, Automated Retraining Trigger Logic, Champion Challenger Evaluation and Model Registtry Foundation Implemented

The platform currently supports data validation, model training and evaluation, statistical drift detection, offline performance monitoring, prediction logging, retraining recommendations, offline champion–challenger model evaluation and JSON-based model registry persistence with structural validation.

Safe model promotion, rollback, API serving, and dashboard functionality remain planned for future development.

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


### 7. Data Drift Detection

Implemented a statistical data drift detection module to monitor changes in incoming customer data compared with a reference dataset.

**Key capabilities:**
- Detects numeric feature drift using the Kolmogorov-Smirnov (KS) test.
- Detects categorical feature drift using the Chi-square test.
- Applies Bonferroni correction to adjust the significance threshold for multiple feature tests.
- Reports reference and current non-missing observation counts.
- Identifies features with statistically significant distribution changes.
- Handles invalid inputs and insufficient data cases.
- Exports drift monitoring results to a CSV report.

**Monitoring module:** `src/self_healing_ml/monitoring/drift_detection.py`

**Execution module:** `src/self_healing_ml/monitoring/monitor_data.py`

**Drift report:** `reports/data_drift_report.csv`

**Validation:** Tested using identical datasets and simulated distribution changes. The monitoring run detected the intentionally introduced drift in `MonthlyCharges`.


### 8. Model Performance Monitoring

Implemented a model performance monitoring module to evaluate predictions and identify potential model degradation.

**Key capabilities:**
- Calculates accuracy, precision, recall, F1-score, and ROC-AUC.
- Compares the current F1-score against a reference baseline.
- Supports configurable minimum F1-score and maximum F1-drop thresholds.
- Generates alerts when configured performance thresholds are violated.
- Exports performance metrics and monitoring status to a CSV report.
- Includes automated tests for metric calculations, input validation, degradation detection, and report generation.

**Monitoring module:** `src/self_healing_ml/monitoring/performance_monitor.py`

**Execution module:** `src/self_healing_ml/monitoring/monitor_model_performance.py`

**Performance report:** `reports/model_performance_report.csv`

**Validation:** The monitoring runner was executed successfully on a fixed test split, producing performance metrics and reporting no degradation against the configured reference baseline.

**Note:** This is an offline demonstration using previously evaluated data. Production monitoring requires new labelled evaluation data and a versioned performance baseline.


### 9. Prediction Logging

Implemented prediction logging to maintain a record of predictions generated by the reusable inference module.

**Key capabilities:**
- Automatically records predictions generated by `predict_churn()`.
- Stores the UTC timestamp, predicted churn label, churn probability, and model version.
- Appends records to a CSV file without overwriting existing predictions.
- Validates prediction labels, probability values, and model versions.
- Includes automated tests for valid inputs, invalid inputs, and repeated predictions.

**Logging module:** `src/self_healing_ml/monitoring/prediction_logger.py`

**Integration module:** `src/self_healing_ml/models/predict.py`

**Prediction log:** `logs/prediction_logs.csv`



### 10. Automated Retraining Triggers

Implemented an orchestration module that evaluates data drift and model performance reports to recommend when retraining should be considered.

**Key capabilities:**
- Reads the data drift and model performance reports.
- Recommends retraining when drift is detected.
- Checks whether model performance is flagged as degraded.
- Rechecks the minimum F1-score and maximum F1-drop thresholds.
- Records the decision and reasons in a JSON report.
- Validates report columns, values, and expected row counts.
- Includes automated tests for trigger conditions, invalid reports, and decision output.

**Orchestration module:** `src/self_healing_ml/orchestration/retraining_trigger.py`

**Decision report:** `reports/retraining_decision.json`

**Execution command:**

```powershell
python -m src.self_healing_ml.orchestration.retraining_trigger
```

**Safety:** The module recommends retraining but does not automatically replace or promote the existing model.

**Current limitation:** Decisions use offline monitoring reports. Live production monitoring and automated model retraining are not yet implemented.

### 11. Champion–Challenger Model Evaluation

Implemented an offline evaluation module that compares a candidate model against the current champion using the same stratified test split.

**Key capabilities:**

- Loads the current champion and challenger model artifacts.
- Evaluates accuracy, precision, recall, F1-score, and ROC-AUC.
- Calculates metric differences between the two models.
- Applies configurable acceptance criteria for F1-score improvement, recall drop, and ROC-AUC drop.
- Produces an explicit `accept_challenger` or `reject_challenger` decision.
- Saves the evaluation results and decision to a JSON report.
- Prevents automatic model promotion.

**Evaluation module:** `src/self_healing_ml/orchestration/champion_challenger.py`

**Champion model:** `models/tuned_logistic_regression.joblib`

**Challenger model:** `models/tuned_gradient_boosting.joblib`

**Decision report:** `reports/champion_challenger_report.json`

**Execution command:**

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m src.self_healing_ml.orchestration.champion_challenger
```

**Current demonstration result:** The challenger was rejected because it failed the minimum F1-score improvement and maximum recall-drop criteria. Its ROC-AUC was higher than the champion's.

**Important limitation:** This is an offline demonstration using a fixed test split that has already been examined during development. The decision is not evidence of production readiness, and no model was promoted or replaced.


### 12. Model Registry Foundation

Implemented a JSON-based model registry to maintain model metadata and track the designated champion model.

**Key capabilities:**
- Load an existing registry or initialize an empty registry.
- Persist registry data in JSON format.
- Create parent directories when saving registry data.
- Validate registry structure and required fields.
- Reject invalid model and history collections.
- Verify registry operations through automated tests.

**Registry module:** `src/self_healing_ml/orchestration/model_registry.py`

**Registry file:** `models/registry.json` (created when the registry is saved).

**Registry structure:**
- `active_champion` — identifies the designated champion model.
- `models` — stores registered model metadata.
- `history` — reserved for recording model management events.

**Current limitation:** The registry foundation is implemented, but controlled model promotion, history updates, and rollback are not yet implemented.


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

The project uses `pytest` to verify data validation, prediction behavior and data drift detection.

**Test files:**
- `tests/test_validation.py` — tests for dataset validation rules.
- `tests/test_predictions.py` — tests for the reusable prediction module.
- `tests/test_drift_detection.py` — tests for numeric and categorical drift detection, input validation, and drift report generation.
- `tests/test_performance_monitor.py` — tests for performance metrics, input validation, degradation detection, and report generation.
- `tests/test_prediction_logger.py` — tests for prediction logging, input validation, and CSV record appending.
- `tests/test_retraining_trigger.py` — tests for retraining decisions, drift and performance triggers, threshold checks, report generation, and invalid inputs.
- `tests/test_model_registry.py` — tests for registry initialization, JSON persistence, directory creation, structural validation, and invalid registry data.

**Test coverage:**
- Data validation: 7 tests
- Prediction module: 6 tests
- Data drift detection: 7 tests
- Performance monitoring: 12 tests
- Prediction logger: 15 tests
- Automated retraining triggers: 9 tests
- Model Registry : 9 tests 

Run the complete test suite from the project root:

```powershell
python -m pytest -v
```

**Latest test result:** 65 tests passed.


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
│   ├── test_predictions.py
│   ├── test_drift_detection.py
│   ├── test_performance_monitor.py
│   ├── test_prediction_logger.py
│   └── test_retraining_trigger.py
│   └── test_model_registry.py
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
│       │   ├── evaluate_models.py
│       │   └── predict.py
│       │
│       ├── monitoring/
│       │   ├── drift_detection.py
│       │   ├── monitor_data.py
│       │   ├── performance_monitor.py
│       │   ├── monitor_model_performance.py
│       │   └── prediction_logger.py
│       ├── orchestration/
│       │   └── retraining_trigger.py
│       │   └── champion_challenger.py
│       │   └── model_registry.py
│       └── api/
│
├── requirements.txt
├── .gitignore
└── README.md
```

*The API directories are reserved for planned components.*

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


### 7. Run Data Drift Monitoring

1. Ensure the reference dataset exists at `data/raw/Telco-Customer-Churn.csv`.
2. Ensure the current dataset exists at `data/processed/current_data.csv`.
3. Run the monitoring module from the project root:

```bash
python -m src.self_healing_ml.monitoring.monitor_data
```

4. Review the drift report generated at `reports/data_drift_report.csv`.

**Note:** The current dataset used during initial testing was simulated by modifying `MonthlyCharges`. This demonstrates drift detection on controlled data; it is not evidence of drift in real production data.

These commands assume the modules support `python -m` execution and that the dataset, dependencies, and required artifacts are available.


### 8. Run Model Performance Monitoring

1. Ensure the trained model exists at `models/tuned_logistic_regression.joblib`.
2. Ensure the dataset exists at `data/raw/Telco-Customer-Churn.csv`.
3. Run the monitoring module from the project root:

```powershell
python -m src.self_healing_ml.monitoring.monitor_model_performance
```

4. Review the generated report at `reports/model_performance_report.csv`.

**Note:** The current runner uses a fixed, previously evaluated test split for demonstration. It does not yet monitor live production predictions.


### 9. Run Prediction Logging

1. Ensure the trained model exists at `models/tuned_logistic_regression.joblib`.
2. Run the prediction module from the project root:

   ```powershell
   python -m src.self_healing_ml.models.predict
   ```

3. Review the prediction history at `logs/prediction_logs.csv`.

Each successful sample prediction appends a new record to the CSV file.


### 10. Run Automated Retraining Trigger Evaluation

1. Ensure the data drift report exists at `reports/data_drift_report.csv`.
2. Ensure the performance report exists at `reports/model_performance_report.csv`.
3. Run the orchestration module from the project root:

```powershell
python -m src.self_healing_ml.orchestration.retraining_trigger
```

4. Review the generated decision at `reports/retraining_decision.json`.

The module recommends retraining when a configured trigger condition is met. It does not automatically train, promote, or replace a model.


### 11. Run Champion–Challenger Evaluation

1. Ensure the champion and challenger model artifacts exist in the `models/` directory.
2. Run the evaluation module from the project root:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m src.self_healing_ml.orchestration.champion_challenger
```

3. Review the evaluation decision at `reports/champion_challenger_report.json`.

The module compares the champion and challenger using configured evaluation criteria and recommends whether the challenger should be accepted or rejected. It does not automatically promote or replace the active model.


## 🗺️ Development Roadmap

- [x] Project scaffold and version control
- [x] Data ingestion and validation
- [x] Automated data validation tests
- [x] Baseline model training
- [x] Multi-model comparison using cross-validation
- [x] Hyperparameter tuning
- [x] Baseline versus tuned-model evaluation
- [x] Reusable prediction and inference module
- [x] Prediction logging and monitoring
- [x] Data drift detection
- [x] Model performance monitoring
- [x] Automated retraining triggers
- [x] Champion–challenger model evaluation
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