"""
Pydantic schemas for the diabetes risk prediction API.
"""

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    age: float = Field(..., ge=1, le=13, description="Age group (1=18-24 … 13=80+)")
    bmi: float = Field(..., gt=0, le=100, description="Body Mass Index")
    bp: float = Field(..., ge=0, le=1, description="High blood pressure (1=Yes, 0=No)")
    cholesterol: float = Field(..., ge=0, le=1, description="High cholesterol (1=Yes, 0=No)")
    smoker: float = Field(..., ge=0, le=1, description="Smoker - 100+ cigarettes ever (1=Yes, 0=No)")
    activity: float = Field(..., ge=0, le=1, description="Physical activity (1=Active, 0=Inactive)")
    health: float = Field(..., ge=1, le=5, description="General health (1=Excellent … 5=Poor)")
    mental: float = Field(..., ge=0, le=30, description="Mental health - bad days in past 30 (0-30)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": 8,
                    "bmi": 32.0,
                    "bp": 1,
                    "cholesterol": 1,
                    "smoker": 0,
                    "activity": 0,
                    "health": 3,
                    "mental": 5,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    risk_percentage: float = Field(..., description="Diabetes risk as percentage (0-100)")
    risk_level: str = Field(..., description="Risk classification: Low, Moderate, or High")
