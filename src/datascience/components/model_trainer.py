from src.datascience import logger
from src.datascience.entity.config_entity import ModelTrainerConfig
import os
import pandas as pd
import joblib
from sklearn.linear_model import ElasticNet


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def initiate_model_trainer(self):
        train_data = pd.read_csv(self.config.train_data_dir)
        test_data = pd.read_csv(self.config.test_data_dir)

        lr = ElasticNet(alpha=self.config.alpha,l1_ratio=self.config.l1_ratio, random_state=101)

        train_x = train_data.drop([self.config.target_name],axis=1)
        train_y = train_data[[self.config.target_name]]
        test_x = test_data.drop([self.config.target_name],axis=1)
        test_y = test_data[[self.config.target_name]]

        lr.fit(train_x,train_y)

        joblib.dump(lr, os.path.join(self.config.root_dir, self.config.model_name))