# Self-Healing Machine Learning Platform

An end-to-end machine learning platform designed to detect data quality
issues and model degradation, evaluate retrained models, and support
safe model promotion and rollback.

## Project Status

🚧 Initial development — project scaffold

## Objectives

- Build a reproducible machine learning pipeline.
- Validate incoming data and detect data quality issues.
- Monitor model health and data drift.
- Automate controlled model retraining.
- Compare candidate models against the production model.
- Support safe model promotion and rollback.
- Expose predictions through an API and monitoring dashboard.

## Initial Use Case

Customer churn prediction using the IBM Telco Customer Churn dataset.

## Planned Technology Stack

- Python, Pandas, NumPy, scikit-learn
- MLflow and Evidently
- FastAPI and Streamlit
- pytest, Git, GitHub Actions, Docker

## Development Roadmap

1. Project setup and version control
2. Data ingestion and validation
3. Preprocessing and baseline model
4. Monitoring and drift detection
5. Automated retraining and model evaluation
6. Model registry, safe promotion and rollback
7. API, dashboard, testing and deployment

## Status

This project is under active development. Features will be
implemented and tested incrementally.
