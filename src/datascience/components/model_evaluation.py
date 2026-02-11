from src.datascience.entity.config_entity import ModelEvaluationConfig
from src.datascience.constants import *
from src.datascience.utils.common import read_yaml,create_directories,save_json

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

import numpy as np
import pandas as pd
import os
import joblib
import mlflow
import mlflow.sklearn
from urllib.parse import urlparse
from pathlib import Path

import os

os.environ["MLFLOW_TRACKING_URL"]="https://dagshub.com/upendranaidu333/Datascienceproject.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"]="upendranaidu333"
os.environ["MLFLOW_TRACKING_PASSWORD"]="e653b390d070a6b1afb9ac7d4750ffd5f260c342"


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self,actual,pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual,pred)
        r2 = r2_score(actual,pred)
        return rmse,mae,r2

    def log_into_mlflow(self):

        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column],axis=1)
        test_y = test_data[[self.config.target_column]]

        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():

            predictions = model.predict(test_x)

            (rmse,mae,r2) = self.eval_metrics(test_y,predictions)

            scores = {'rmse':rmse, 'mae':mae, 'r2':r2}
            save_json(path=Path(self.config.metric_file_name),data=scores)

            mlflow.log_params(self.config.all_params)

            mlflow.log_metric('rmse',rmse)
            mlflow.log_metric('mae',mae)
            mlflow.log_metric('r2',r2)


            if tracking_url_type_store != "file":

                mlflow.sklearn.log_model(model, "model", registered_model_name="ElasticnetModel")

            else:
                mlflow.sklearn.log_model(model, "model")