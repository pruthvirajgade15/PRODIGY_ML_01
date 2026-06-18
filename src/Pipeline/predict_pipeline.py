import os
import sys
from dataclasses import dataclass

import pandas as pd

from src.exception import CustomException
from src.utils import load_object


@dataclass
class CustomData:
    area_type: str
    availability: str
    location: str
    size: str
    society: str
    total_sqft: float
    bath: float
    balcony: float

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "area_type": [self.area_type],
                "availability": [self.availability],
                "location": [self.location],
                "size": [self.size],
                "society": [self.society],
                "total_sqft": [self.total_sqft],
                "bath": [self.bath],
                "balcony": [self.balcony],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)


class PredictPipeline:
    _model = None
    _preprocessor = None

    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

    def predict(self, features):
        try:
            if PredictPipeline._model is None:
                PredictPipeline._model = load_object(file_path=self.model_path)

            if PredictPipeline._preprocessor is None:
                PredictPipeline._preprocessor = load_object(
                    file_path=self.preprocessor_path
                )

            transformed_features = PredictPipeline._preprocessor.transform(features)
            prediction = PredictPipeline._model.predict(transformed_features)

            return prediction

        except Exception as e:
            raise CustomException(e, sys)
