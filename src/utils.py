import os
import pandas as pd
import numpy as np
import dill
from src.logger import logging
from src.exception import CustomException
import sys
def save_object(file_path, obj):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
        logging.info(f"Object saved successfully at {file_path}")
    except Exception as e:
        logging.error("Error occurred while saving object")
        raise CustomException(e, sys) from e