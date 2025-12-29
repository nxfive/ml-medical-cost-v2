import requests


class BentoClient:
    def __init__(self, bento_url: str):
        self.bento_url = bento_url

    def predict(self, data: dict) -> float:
        """
        Sends a single input data dictionary to the BentoML service for prediction.
        """
        response = requests.post(
            f"{self.bento_url}/predict", json={"input_data": data}, timeout=10
        )
        response.raise_for_status()
        return response.json()["charges"]

    def predict_many(self, data_list: list[dict]) -> list[float]:
        """
        Sends multiple input data dictionaries to the BentoML service for batch 
        prediction.
        """
        response = requests.post(
            f"{self.bento_url}/predict_multiple",
            json={"input_data": data_list},
            timeout=10,
        )
        response.raise_for_status()
        return response.json()["charges"]
