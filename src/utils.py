import os
import sys
import pickle
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    """
    Save a Python object to a pickle file.
    
    Args:
        file_path (str): Path where the object will be saved
        obj: The object to be saved
    
    Raises:
        CustomException: If an error occurs during saving
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        
        logging.info(f"Object saved successfully at {file_path}")
    
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """
    Load a Python object from a pickle file.
    
    Args:
        file_path (str): Path to the pickle file
    
    Returns:
        The loaded object
    
    Raises:
        CustomException: If an error occurs during loading
    """
    try:
        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)
        
        logging.info(f"Object loaded successfully from {file_path}")
        return obj
    
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models):
    """
    Evaluate multiple models and return their scores.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Testing features
        y_test: Testing target
        models (dict): Dictionary of model name and model object
    
    Returns:
        dict: Dictionary with model names and their test scores
    
    Raises:
        CustomException: If an error occurs during evaluation
    """
    try:
        report = {}
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = model.score(X_test, y_test)
            report[name] = score
            logging.info(f"Model {name} score: {score}")
        
        return report
    
    except Exception as e:
        raise CustomException(e, sys)
