import pandas as pd

from src.features.core import convert_features_type

from .bento_client import BentoClient
from .prediction_repository import PredictionRepository
from .schemas import MedicalCostFeatures


class PredictionService:
    def __init__(self, client: BentoClient, repository: PredictionRepository):
        self.client = client
        self.repository = repository

    def predict(self, features: MedicalCostFeatures) -> float:
        """
        Makes a prediction for a single set of features and saves it to the database.
        """
        payload = convert_features_type(pd.DataFrame([features.model_dump()])).to_dict(
            orient="records"
        )[0]
        prediction = self.client.predict(payload)
        self.repository.save(payload, prediction)
        return prediction

    def predict_many(self, features_list: list[MedicalCostFeatures]) -> list[float]:
        """
        Makes predictions for multiple sets of features and saves them to the database.
        """
        payload_list = [f.model_dump() for f in features_list]
        converted_list = convert_features_type(pd.DataFrame(payload_list)).to_dict(
            orient="records"
        )
        predictions = self.client.predict_many(converted_list)
        self.repository.save_many(payload_list, predictions)
        return predictions
