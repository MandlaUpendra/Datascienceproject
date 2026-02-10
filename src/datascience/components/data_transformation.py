import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.datascience import logger
from src.datascience.entity.config_entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def train_test_split(self):
        data = pd.read_csv(self.config.data_dir)

        train,test = train_test_split(data)

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"),index = False)
        test.to_csv(os.path.join(self.config.root_dir, 'test.csv'),index=False)

        logger.info('Spliting the data into train and test sets')
        logger.info(train.shape)
        logger.info(test.shape)

        print(train.shape)
        print(test.shape)
