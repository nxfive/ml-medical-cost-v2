from typing import Annotated

from fastapi import Depends, FastAPI

from .prediction import get_prediction_service
from .prediction_service import PredictionService
from .schemas import (MedicalCostFeatures, PredictionManyResponse,
                      PredictionResponse)

app = FastAPI()


@app.post("/predict", response_model=PredictionResponse)
def predict(
    features: MedicalCostFeatures,
    service: Annotated[PredictionService, Depends(get_prediction_service)],
) -> dict[str, float]:
    result = service.predict(features)
    return {"charges": result}


@app.post("/predict_many", response_model=PredictionManyResponse)
def predict_many(
    features_list: list[MedicalCostFeatures],
    service: Annotated[PredictionService, Depends(get_prediction_service)],
) -> dict[str, float]:
    result = service.predict_many(features_list)
    return {"charges": result}
