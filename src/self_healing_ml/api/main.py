from fastapi import FastAPI, HTTPException
from src.self_healing_ml.models.predict import  FEATURE_COLUMNS, MODEL_PATH, predict_churn
from src.self_healing_ml.api.schemas import HealthResponse, ModelInfoResponse, PredictionRequest, PredictionResponse


# Application configuration
app = FastAPI(
    title="Self-Healing ML Platform API",
    description=("REST API for customer churn prediction using the Self-Healing ML Platform."),
    version="1.0.0",
)


MODEL_NAME = "tuned_logistic_regression"


@app.get("/")
def root():
    """Return the API welcome message."""
    return {
        "message": "Welcome to the Self-Healing ML Platform API.",
        "version": app.version,
        "documentation": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Check API and model artifact availability."""
    model_available = MODEL_PATH.is_file()
    return {
        "status": "ok" if model_available else "degraded",
        "model_available": model_available,
    }


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info():
    """Return information about the active prediction model."""
    return {
        "model_name": MODEL_NAME,
        "model_file": MODEL_PATH.name,
        "model_available": MODEL_PATH.is_file(),
        "feature_count": len(FEATURE_COLUMNS),
        "features": FEATURE_COLUMNS,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_customer_churn(request: PredictionRequest):
    """Predict customer churn using the existing ML inference pipeline."""
    customer_data = request.customer.model_dump()
    try:
        result = predict_churn(customer_data)
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Prediction model is currently unavailable.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc