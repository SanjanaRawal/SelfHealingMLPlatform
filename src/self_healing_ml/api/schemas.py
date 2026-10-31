from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class CustomerFeatures(BaseModel):
    """Validate the customer features required for churn prediction."""
    model_config = ConfigDict(extra="forbid")

    gender: str
    SeniorCitizen: Literal[0, 1]
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float | None = Field(default=None, ge=0)


class PredictionRequest(BaseModel):
    """Request body for the prediction endpoint."""
    customer: CustomerFeatures


class PredictionResponse(BaseModel):
    """Response returned after a churn prediction."""
    prediction: Literal["Yes", "No"]
    churn_probability: float = Field(ge=0, le=1)


class HealthResponse(BaseModel):
    """API health-check response."""
    status: str
    model_available: bool


class ModelInfoResponse(BaseModel):
    """Information about the model used by the API."""
    model_name: str
    model_file: str
    model_available: bool
    feature_count: int
    features: list[str]