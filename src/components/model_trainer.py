import pandas as pd
from sklearn.linear_model import LinearRegression
from src.logger import logging
from src.exception import CustomException
import sys
from src.utils import save_object
from dataclasses import dataclass
import os
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor
@dataclass
class ModelTrainerConfig:##save the final trained model here in model.pkl
    trained_model_path: str = os.path.join("artifacts", "model.pkl")
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing input data")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "KNeighbors": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "Linear Regression": LinearRegression()
            }

            model_report = {}

            for model_name, model in models.items():
                logging.info(f"Training {model_name}")
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                r2_square = r2_score(y_test, y_pred)
                model_report[model_name] = r2_square ##This stores the r2 score for each model
                logging.info(f"{model_name} R2 Score: {r2_square}")

            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            logging.info(f"Best model found: {best_model_name} with R2 Score: {best_model_score}")

            if best_model_score < 0.6:
                raise CustomException("No best model found with acceptable accuracy")

            save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj=best_model
            )

            logging.info("Best model saved successfully")

            return best_model_score

        except Exception as e:
            logging.error("Error occurred during model training")
            raise CustomException(e, sys) from e
if __name__ == "__main__":
    pass
    