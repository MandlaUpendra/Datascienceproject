from src.datascience import logger
from src.datascience.config.configuration import ConfigurationManager
from src.datascience.components.model_trainer import ModelTrainer


STAGE_NAME = "Model Trainer stage"

class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def initiate_model_trainer(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer = ModelTrainer(model_trainer_config)
        model_trainer.initiate_model_trainer()