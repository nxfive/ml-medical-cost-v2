from db import Database
from sqlalchemy.orm import Session


class PredictionRepository:
    def __init__(self, db_session: Session):
        self.db = db_session

    def save(self, data: dict, prediction: float) -> None:
        """
        Saves a single prediction record to the database.
        """
        Database(self.db).create_record(data=data, prediction=prediction)

    def save_many(self, data_list: list[dict], predictions: list[float]) -> None:
        """
        Saves multiple prediction records to the database.
        """
        Database(self.db).create_records(data_list=data_list, predictions=predictions)
