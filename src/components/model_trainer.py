import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting train and test data")

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

            params = {
                "Decision Tree": {
                    "criterion": ["squared_error", "friedman_mse"]
                },
                "Random Forest": {
                    "n_estimators": [50, 100, 200]
                },
                "KNeighbors": {
                    "n_neighbors": [5, 7, 9],
                    "weights": ["uniform", "distance"]
                },
                "XGBRegressor": {
                    "learning_rate": [0.01, 0.1],
                    "max_depth": [3, 5],
                    "n_estimators": [100, 200]
                },
                "CatBoosting Regressor": {
                    "depth": [4, 6],
                    "learning_rate": [0.01, 0.1],
                    "iterations": [100, 200]
                },
                "Linear Regression": {}
            }

            best_model_score = -float("inf")
            best_model_name = None
            best_model_obj = None

            for model_name, model in models.items():
                logging.info(f"Training {model_name}")

                if params[model_name]:
                    search = RandomizedSearchCV(
                        estimator=model,
                        param_distributions=params[model_name],
                        n_iter=10,
                        cv=3,
                        scoring="r2",
                        n_jobs=-1,
                        random_state=42
                    )
                    search.fit(X_train, y_train)
                    trained_model = search.best_estimator_
                else:
                    model.fit(X_train, y_train)
                    trained_model = model

                y_pred = trained_model.predict(X_test)
                r2 = r2_score(y_test, y_pred)

                logging.info(f"{model_name} R2 Score: {r2}")

                if r2 > best_model_score:
                    best_model_score = r2
                    best_model_name = model_name
                    best_model_obj = trained_model

            logging.info(
                f"Best model: {best_model_name} with R2 Score: {best_model_score}"
            )

            if best_model_score < 0.6:
                raise CustomException("No model achieved acceptable performance", sys)

            save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj=best_model_obj
            )

            logging.info("Best model saved successfully")

            return best_model_score

        except Exception as e:
            logging.error("Error occurred during model training")
            raise CustomException(e, sys)
