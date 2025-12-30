from typing import Annotated

from fastapi import Depends
from prediction_repository import PredictionRepository
from sqlalchemy.orm import Session

from database.db import get_db
from services.backend.fastapi.bento_client import bento_client
from services.backend.fastapi.prediction_service import PredictionService


def get_prediction_repository(
    db: Annotated[Session, Depends(get_db)],
) -> PredictionRepository:
    """
    Creates and returns a PredictionRepository using the current DB session.
    """
    return PredictionRepository(db)


def get_prediction_service(
    repository: Annotated[PredictionRepository, Depends(get_prediction_repository)],
) -> PredictionService:
    """
    Creates and returns a PredictionService with the BentoClient and repository.
    """
    return PredictionService(bento_client, repository)
